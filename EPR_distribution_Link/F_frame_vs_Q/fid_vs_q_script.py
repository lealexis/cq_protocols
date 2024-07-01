
from qunetsim.components import Network, Host
from qunetsim.objects import Qubit, DaemonThread
import numpy as np 
import random

import time

import QuTils
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_itfc_FT_NON_IDEAL_FEU import EPR_buff_itfc
from rotational_error_class import Rot_Error
from pipelined_CQ_channel import qPipe, cPipe
from increasingFrameLen import frLenIncr

from threading import Event
import pandas as pd

INTER_CBIT_TIME = 0.1  # time between the bits of the feedback signals. 30 ms
INTER_QBIT_TIME = 0.02  # time between the qubits of the quantum frames. 18ms
EFF_LOAD = 40

start_time = 0.0
start_time_frame = 0.0
finish_time = 0.0
PROCESS = []
SimuData = pd.DataFrame({"F_est_ideal": pd.Series(dtype="float"),
                         "N": pd.Series(dtype="int"),
                         "Q": pd.Series(dtype="float"),
                         "F_est": pd.Series(dtype="float"),
                         "t_span": pd.Series(dtype="float")})

low_quality = True

if low_quality:
    base_path = "../fid_analysis/data/low_quality/"
else:
    base_path = "../fid_analysis/data/high_quality/"


def sEPR_proto(host: Host, receiver_id, epr_gen: EPR_generator, qmem_itfc: EPR_buff_itfc, q_chan: qPipe, c_chan: cPipe,
               error_gen: Rot_Error, proto_finished: Event, fr_len: frLenIncr, sRand, begin_end_simu: Event,
               qframe_received: Event, verbose_level=0):
    """Distributes an EPR-frame from Alice to Bob. Each time this method is used, an EPR-frame is
    generated at Alice and distributed to Bob and a fidelity estimation process takes place after this distribution"""

    F_ideal_BF_chann = 0  # tracks the fidelity of each generated EPR-pair before their distribution over noisy quantum
    # channel, ideal because the fidelity of each EPR pair is accessed without destroying them

    nfID = qmem_itfc.nfID_EPR_START()  # get ID of EPR-frame to be generated and distributed
    #if (verbose_level == 0) or (verbose_level == 1):
    #    print("ALICE/EPR - send EPR-Frame nfID:{id_epr}".format(id_epr=nfID))

    frame_len, count = fr_len.get_length()

    if fr_len.last_frame.is_set():
        begin_end_simu.set()

    for i in range(frame_len):
        q1, q2 = epr_gen.get_EPR_pair()  # EPR-pair is generated in loop
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)  # local EPR-pair half
        if verbose_level == 1:
            print("ALICE/EPR - send EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)  # add receiver to EPR-pair half
        error_gen.apply_error(flying_qubit=q2)  # simulating noisy channel error
        F_ideal_BF_chann += epr_gen.get_fidelity(half=q1)  # tracks fidelity before channel: NOW after channel
        q_chan.put(q2)  # passing EPR-pair half to channel
        time.sleep(INTER_QBIT_TIME)  # wait for receiver to be ready

    # delete history of epr_gen and error_gen
    epr_gen.history.drop(epr_gen.history.index, inplace=True)
    error_gen.history.drop(error_gen.history.index, inplace=True)

    feedback_to_feu = []
    F_ideal_BF_chann = (F_ideal_BF_chann / frame_len)  # Average fidelity of EPR-frame before noisy channel

    feedback_to_feu.append(F_ideal_BF_chann)
    qmem_itfc.set_F_EPR_END_PHASE(F_est=F_ideal_BF_chann, to_history=True)  # keeps tracks of fidelity before channel

    ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()  # qubit ids of local EPR-pairs halves

    #if (verbose_level == 0) or (verbose_level == 1):
    #    print("ALICE/EPR - random measurements")
    meas_amount = len(ip_qids) - qmem_itfc.eff_load  # raw load - eff load = amount of measurements
    meas_qids = sRand.sample(ip_qids, int(meas_amount))  # random qubits for fidelity estimation: X-,Y-,& Z-Meas.

    # FEU-FEEDBACK measurement order [mx , my, mz]
    # random X-Measurements
    x_qids = sRand.sample(meas_qids, int(meas_amount/3))  # 1/3 of meas_amount to be X-Measured

    # TODO: wait for qbits to be received at Bob, then measure you idiot!

    qframe_received.wait()

    for idq in x_qids:  # after loop meas_qids contains 1/3 less qubits
        mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
        mq.H()  # Apply hadamard to implement X-Meas
        bit = mq.measure()
        feedback_to_feu.append(bit)  # store measurement output
        meas_qids.remove(idq)  # remove idq from meas_qids

    # random Y-Measurements
    s_dagger_mtrx = np.array([[1, 0], [0, -1j]])  # needed for Y-Meas
    y_qids = sRand.sample(meas_qids, int(len(meas_qids)/2))  # take half of meas_qids, this is 1/3 of the original
    for idq in y_qids:  # after loop meas_qids contains tha last 1/3 of qubits to be measured
        mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
        mq.custom_gate(s_dagger_mtrx)  # Apply s_dagger and hadamard to implement Y-Meas
        mq.H()
        bit = mq.measure()
        feedback_to_feu.append(bit)  # store measurement output
        meas_qids.remove(idq)  # remove idq from measuring list

    # random Z-Measurements
    for idq in meas_qids:  # use the rest qids for Z-Meas, this is 1/3 of the original
        mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
        bit = mq.measure()
        feedback_to_feu.append(bit)  # store measurement output

    for bit in feedback_to_feu:  # send feedback to the FEU
        c_chan.put(bit)
        time.sleep(INTER_CBIT_TIME)

    c_chan.feedback_to_host.wait()
    while True:
        try:  # continuous hearing classical channel
            f_est = c_chan.out_socket.pop()
        except IndexError:
            continue
        else:  # fidelity was received
            qmem_itfc.drop_ip_frame_EPR_END_PHASE(fest=f_est)  # drop remaining EPR-pairs
            qmem_itfc.EPR_frame_history.drop(qmem_itfc.EPR_frame_history.index, inplace=True)
            qmem_itfc.In_halves_history.drop(qmem_itfc.In_halves_history.index, inplace=True)
            break
    qframe_received.clear()
    proto_finished.wait()
    return count, frame_len


def sender_protocol(host: Host, receiver_id, epr_gen: EPR_generator, qmem_itfc: EPR_buff_itfc, q_chan: qPipe,
                    c_chan: cPipe, error_gen: Rot_Error, proto_finished: Event, len_est: frLenIncr, sd_random,
                    feed_to_feu_ready: Event, finish_simu: Event, begin_end_simu: Event, qframe_received: Event,
                    verbose_level=0):
    """Alice protocol to transmit EPR- and SDC-frames depending on which kind of Job is in PROCESS"""

    global PROCESS
    global SimuData
    global start_time_frame
    sd_random.seed(1010)  # same seed used in Alice & Bob for random processes in fidelity estimation
    while True:
        try:
            process_type = PROCESS.pop()
        except IndexError:
            continue
        else:
            if process_type == "EPR":
                proto_finished.clear()  # event is set back to 0
                feed_to_feu_ready.clear()
                #print("\nALICE/EPR - Starting EPR-Frame distribution.")
                start_time_frame = time.time()
                fr_count, fr_len = sEPR_proto(host, receiver_id, epr_gen, qmem_itfc, q_chan, c_chan, error_gen,
                                              proto_finished, len_est, sd_random, begin_end_simu, qframe_received,
                                              verbose_level)
                print('\rFrame length:[{min} ---> {l} <--- {max}] ***frame {c} out of {m}***'.format(
                    min=len_est.lengths[0], l=fr_len, max=len_est.lengths[-1], c=fr_count, m=len_est.m_frames), sep='',
                    end='', flush=True)
                if fr_count == len_est.m_frames:
                    data_path = base_path + "length_" + str(fr_len) + "_" + str(len_est.m_frames) + "_frames.csv"
                    SimuData.to_csv(data_path)
                    print("\nData of " + str(len_est.m_frames) + " frames with length " + str(fr_len) + " was stored " +
                          "and took " + str(SimuData["t_span"].sum()) + "[min].")
                    SimuData = SimuData.iloc[0:0]
                continue  # go back to get next element from PROCESS
            else:  # Simulation is done
                finish_simu.set()
                break
    print("\nFinalizing simulation.")
                

def put_next_process(proto_finished: Event):
    """Whenever the protocol finishes the distribution of an EPR- or SDC-frame, signaled by the
    proto_finished event, it will be appended to PROCESS either SDC or EPR. SDC is chosen
    accordingly to the job arrival probability job_prob.
    """
    global PROCESS
    while True:
        proto_finished.wait()
        if len(PROCESS) == 0:
            PROCESS.append("EPR")
        continue


def feu(c_chan_A: cPipe, c_chan_B: cPipe, proto_finished: Event, feed_from_alice_ready: Event,  eff_N=40):

    global SimuData
    global finish_time
    global start_time_frame
    while True:
        c_chan_A.feedback_to_feu.wait()
        meas_A = []
        meas_B = []
        frame_data = []
        count = 0

        while True:
            try:  # continuous hearing of  channel
                data_A = c_chan_A.out_socket.pop()
            except IndexError:
                continue
            else:  # data from Alice was received
                count += 1
                if count != 1:  # measurement outputs alice
                    meas_A.append(data_A)
                else:  # ideal fidelity
                    frame_data.append(data_A)
                if c_chan_A.feedback_to_feu.is_set():
                    continue
                else:  # Alice's feedback received
                    n_meas = count - 1
                    n = n_meas + eff_N
                    q = n_meas / n
                    frame_data.append(n)
                    frame_data.append(q)
                    break

        feed_from_alice_ready.set()
        c_chan_B.feedback_to_feu.wait()
        while True:
            try:  # continuous hearing of  channel
                data_B = c_chan_B.out_socket.pop()
            except IndexError:
                continue
            else:  # measurement output Bob was received
                meas_B.append(data_B)
                if c_chan_B.feedback_to_feu.is_set():
                    continue
                else:  # Bob's feedback received
                    break

        # Fidelity estimation after measurements by qubit error rate for X-, Y-, and Z-measurement
        meas_count = 0
        mxyz_len = len(meas_A)  # amount of X, Y, and Z measurement outputs
        qberx = 0  # initial values of 0 for qber
        qbery = 0
        qberz = 0
        for idx in range(len(meas_A)):  # calculate different QBER looping over measurement outputs
            meas_count += 1
            if meas_count < (mxyz_len + 1):  # first 1/3 of measurements are X-Measurements outputs
                if meas_A[idx] == meas_B[idx]:  # both equal --> no error
                    continue
                else:  # different --> error
                    qberx += 1
            elif mxyz_len < meas_count < (2 * mxyz_len + 1):  # 2nd 1/3 of measurements are Y-Measurements outputs
                if meas_A[idx] == meas_B[idx]:  # both equal --> error
                    qbery += 1
                else:  # different --> no error
                    continue
            else:  # last third are Z-Measurements outputs
                if meas_A[idx] == meas_B[idx]:  # both equal --> no error
                    continue
                else:  # different --> error
                    qberz += 1
        # divide by amount of measurements and QBER is calculated for X, Y and Z measurements
        qberx = qberx / mxyz_len
        qbery = qbery / mxyz_len
        qberz = qberz / mxyz_len
        f_est_dahl = 1 - (qberx + qbery + qberz) / 2
        frame_data.append(f_est_dahl)

        c_chan_A.put(f_est_dahl)
        c_chan_B.put(f_est_dahl)
        finish_time = time.time()
        span = (finish_time - start_time_frame) / 60
        frame_data.append(span)
        SimuData = pd.concat([SimuData, pd.DataFrame([frame_data], columns=SimuData.columns)])

        proto_finished.wait()
        continue


def receiver_protocol(host:Host, qmem_itfc: EPR_buff_itfc, q_chan: qPipe, c_chan: cPipe, proto_finished: Event,
                      alice_feu_feed_ready: Event, begin_end_simu: Event, rRandom, qframe_received: Event, verbose_level=0):

    """A quantum frame is received from the qu_chann and the header is measured to decide if quantum frame is an
    EPR-frame or SDC-frame. Then the protocol runs correspondingly interpreting the quantum load as EPR or SDC and
    sending classical feedback's between Alice and Bob until the protocol is finalized and the complete quantum load is
    validated."""

    global finish_time  # to track duration of the simulation
    global PROCESS  # used to enforce finalization of the simulation

    rRandom.seed(1010)  # same seed used in Alice & Bob for random processes in fidelity estimation
    count = 0  # store amount of received qubits in a quantum frame
    f_est_ideal = 0
    #frame_id = None
    qbit = None
    while True:
        try:  # continuous hearing of quantum channel
            qbit = q_chan.out_socket.pop()
        except IndexError:
            continue
        else:  # qubit was received, proceed to interpret it
            if count == 0:
                frame_id = qmem_itfc.nfID_EPR_START()  # id of EPR-frame being received
                #if (verbose_level == 0) or (verbose_level == 1):
                #    print("BOB  /EPR - recv EPR-Frame nfID:{id_u}".format(
                #        id_u=frame_id))
            count += 1

            if verbose_level == 1:
                print("BOB  /EPR - recv EPR halve Nr: {hnum}".format(hnum=count))
            f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)  # Fidelity of EACH distributed EPR-pair --> ideal
            qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)

            if q_chan.Qframe_in_transmission.is_set():  # Quantum frame is still in transmission
                continue

            else:  # Quantum frame was  completely received
                qframe_received.set()
                f_est_ideal = f_est_ideal / count  # (ideally estimated) average Fidelity of EPR-Frame
                #print("BOB  /EPR - Ideal fidelity estimation: {}".format(f_est_ideal))

                # set ideally estimated EPR-Frame's fidelity to entanglement manager
                qmem_itfc.set_F_EPR_END_PHASE(F_est=f_est_ideal, to_history=True)
                ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()  # in process (ip) qubit identifiers (qids)

                meas_amount = len(ip_qids) - qmem_itfc.eff_load  # always divisible by 3 (X-, Y- & Z-Meas)
                            
                #if (verbose_level == 0) or (verbose_level == 1):
                #    print("BOB  /EPR - {} qubits in payload".format(count - 1))
                #    print("BOB  /EPR - {} random measurements to be performed".format(meas_amount))

                meas_qids = rRandom.sample(ip_qids, int(meas_amount))  # random qubits to be measured
                            
                # Composition of FEU-Feedback: [mx , my, mz]
                feed_to_feu = []

                # X-MEASUREMENT
                x_qids = rRandom.sample(meas_qids, int(meas_amount/3))  # random qubits for X-Measurement
                for idq in x_qids:  # perform X-Measurement and add it to epr feedback
                    mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                    mq.H()
                    bit = mq.measure()
                    feed_to_feu.append(bit)
                    meas_qids.remove(idq)  # remove measured qubit from list of qubits to measure

                # Y-MEASUREMENT
                s_dagger = np.array([[1, 0], [0, -1j]])
                y_qids = rRandom.sample(meas_qids, int(len(meas_qids)/2))  # random qubits for Y-Measurement
                for idq in y_qids:  # perform Y-Measurement and add it to epr feedback
                    mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                    mq.custom_gate(s_dagger)
                    mq.H()
                    bit = mq.measure()
                    feed_to_feu.append(bit)
                    meas_qids.remove(idq)  # remove measured qubit from list of qubits to measure

                # Z-MEASUREMENT
                # left qubits in "meas_qids" were already randomly chosen
                for idq in meas_qids:  # perform Z-Measurement and add it to epr feedback
                    mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                    bit = mq.measure()
                    feed_to_feu.append(bit)

                #if (verbose_level == 0) or (verbose_level == 1):
                #    print("BOB  /EPR - send FEU-Feedback")

                if alice_feu_feed_ready.is_set():  # start transmission to feu after Alice
                    for bit in feed_to_feu:  # send EPR-FEEDBACK [mx, my, mz]
                        c_chan.put(bit)
                        time.sleep(INTER_CBIT_TIME)  # give time to feu to process the received cbit
                else:
                    alice_feu_feed_ready.wait()
                    for bit in feed_to_feu:  # send EPR-FEEDBACK [mx, my, mz]
                        c_chan.put(bit)
                        time.sleep(INTER_CBIT_TIME)  # give time to feu to process the received cbit

                c_chan.feedback_to_host.wait()  # When feedback starts, continue

                while True:
                    try:  # continuous hearing classical channel
                        f_est = c_chan.out_socket.pop()
                    except IndexError:
                        continue
                    else:  # fidelity was received
                        qmem_itfc.drop_ip_frame_EPR_END_PHASE(fest=f_est)  # drop remaining EPR-pairs
                        qmem_itfc.EPR_frame_history.drop(qmem_itfc.EPR_frame_history.index, inplace=True)
                        qmem_itfc.In_halves_history.drop(qmem_itfc.In_halves_history.index, inplace=True)
                        break

                if begin_end_simu.is_set():
                    finish_time = time.time()
                    PROCESS.append(None)  # to stop simulation add None to process

                f_est_ideal = 0
                count = 0
                proto_finished.set()

                continue  # protocol finalized-continue with outer while loop


def main():
    global finish_time
    global low_quality
    global start_time

    start_time = time.time()

    # Establish network and nodes
    network = Network.get_instance()

    Alice = Host('A')
    Bob = Host('B')

    Alice.add_connection('B')
    Bob.add_connection('A')

    Alice.start()
    Bob.start()

    network.add_hosts([Alice, Bob])
    network.start()

    if low_quality:
        # epr fidelity
        mean_fid = 0.5
        dev_fid = 0.15
        # rotational angle
        mean_rot = 0.45
        dev_rot = 0.1
    else:
        # epr fidelity
        mean_fid = 0.99
        dev_fid = 0.015
        # rotational angle
        mean_rot = 0.05
        dev_rot = 0.03

    # fidelity
    mx_fid = mean_fid + 0.01
    mn_fid = mean_fid - 0.01
    mx_dev_fid = dev_fid + 0.01
    mn_dev_fid = dev_fid - 0.01

    epr_gen = EPR_generator(host=Alice, max_fid=mx_fid, min_fid=mn_fid, max_dev=mx_dev_fid, min_dev=mn_dev_fid,
                            typ="gaussian")
    epr_gen.start()

    # rotational angle
    mx_rot = mean_rot + 0.01
    mn_rot = mean_rot - 0.01
    mx_dev_rot = dev_rot + 0.01
    mn_dev_rot = dev_rot - 0.01

    rot_error = Rot_Error(max_rot=mx_rot, min_rot=mn_rot, max_dev=mx_dev_rot, min_dev=mn_dev_rot, typ="gaussian")
    rot_error.start_time = epr_gen.start_time

    # pipelined classical and quantum channel
    delay = 8  # 1.2 minimum delay, imply channel length in seconds
    q_channel = qPipe(delay=delay)
    c_channel_a_feu = cPipe(delay=delay)
    c_channel_b_feu = cPipe(delay=delay)

    # TODO: change Qumem ITFCs to entanglement managers
    # Qumem ITFCs

    # Alice
    qm_itfc_a = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False, eff_load=EFF_LOAD)
    qm_itfc_a.start_time = epr_gen.start_time

    # Bob
    qm_itfc_b = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True, eff_load=EFF_LOAD)
    qm_itfc_b.start_time = epr_gen.start_time

    print("\nHost Alice and Bob started. Network started.")
    print("Starting communication: Alice sender, Bob receiver.\n")
    print("mu-fid:{mf} & sig-fid:{sf}.".format(mf=mean_fid, sf=dev_fid))
    print("mu-rxy:{mr} & sig-rxy:{sr}.\n".format(mr=mean_rot, sr=dev_rot))

    # TODO: adapt min and max q
    fr_lengths = frLenIncr(min_q=0.791, max_q=0.8, frames_per_len=10)
    #fr_lengths = frLenIncr(min_q=0.724, max_q=0.728, frames_per_len=10)

    min_l = fr_lengths.lengths[0]
    max_l = fr_lengths.lengths[-1]

    print("q varying between " + str((min_l - 40) / min_l) + " and " + str((max_l - 40) / max_l) + ".\n")
    print("frame length varying between " + str(min_l) + " and " + str(max_l) + ".\n")
    # defining necessary events
    epr_frame_distributed = Event()  # proto_finished event !
    alice_feed_to_feu_ready = Event()
    begin_end_simu = Event()
    finish_simu = Event()
    qframe_received = Event()

    # to manage random choices in sender and receiver
    send_random = random.Random()
    recv_random = random.Random() 

    DaemonThread(target=receiver_protocol, args=(Bob, qm_itfc_b, q_channel, c_channel_b_feu, epr_frame_distributed,
                                                 alice_feed_to_feu_ready, begin_end_simu, recv_random, qframe_received,
                                                 2))

    DaemonThread(target=put_next_process, args=(epr_frame_distributed,))

    DaemonThread(target=feu, args=(c_channel_a_feu, c_channel_b_feu, epr_frame_distributed, alice_feed_to_feu_ready))

    DaemonThread(target=sender_protocol, args=(Alice, Bob.host_id, epr_gen, qm_itfc_a, q_channel, c_channel_a_feu,
                                               rot_error, epr_frame_distributed, fr_lengths, send_random,
                                               alice_feed_to_feu_ready, finish_simu, begin_end_simu, qframe_received,
                                               2))

    epr_frame_distributed.set()  # force beginning of communication

    finish_simu.wait()

    print("\nSimulation time duration in hours is: {t}".format(t=((finish_time - start_time) / 3600)))

    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()    


if __name__ == '__main__':
    main()