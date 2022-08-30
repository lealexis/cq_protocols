from qunetsim.components import Network, Host
from qunetsim.objects import Qubit, DaemonThread
import numpy as np 
import random
from PIL import Image
import time

import QuTils
from meas_portion import frame_length_estimator as frm_len_EST
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_itfc_FT_NON_IDEAL_FEU import EPR_buff_itfc
from rotational_error_class import Rot_Error
from pipelined_CQ_channel import QuPipe, ClassicPipe

from threading import Event 
from matplotlib import pyplot as plt
import math

INTER_CBIT_TIME = 0.01
INTER_QBIT_TIME = 0.012 # 10ms
EFF_LOAD = 35 # Load length - 560 bits to be sent
BASE_FIDELITY = 0.5

global sent_mssgs 
sent_mssgs = ""
global dcdd_mssgs
dcdd_mssgs = ""
global finish_time
finish_time = None
global PROCESS
PROCESS = []
global SAVE_MARIO
SAVE_MARIO = False

"""Definition of classical data to be sent through the network. An image 
is used to be sent over entangled-assisted communication."""

im = Image.open("./Playground_v0.8/mario_sprite.bmp")
pixels = np.array(im)
im.close()
coloumns, rows, colors = pixels.shape
dtype = pixels.dtype

hashes = []
hashes_ = []

palette = {}
indices = {}
for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        hashes.append(hashed)
        hashes_ = hashes.copy()
        palette[hashed] = color

hashes = list(set(hashes))

for i, hashed in enumerate(hashes):
    indices[hashed] = i

def binaryTupleFromInteger(i):
    return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])

def integerFromBinaryTuple(a, b):
    return a * 2 ** 1 + b * 2 ** 0

# to store the received image
global received_mario
received_mario = np.zeros((coloumns, rows, colors), dtype=dtype)

global im_2_send 
im_2_send = ""

for row in range(rows):
    for column in range(coloumns):
        color = pixels[column, row, :]
        hashed = hash(tuple(color))
        index = indices[hashed]
        b1, b2 = binaryTupleFromInteger(index)
        im_2_send += str(b1) + str(b2)

def int2bin(i,length):
    return [int(j) for j in list(bin(i)[2:].zfill(length))]

def bin2int(bin_str):
    return int(bin_str,2)

def sEPR_proto(host: Host, receiver_id , epr_gen: EPR_generator, 
                   qmem_itfc: EPR_buff_itfc, quPipeChann:QuPipe, 
                   clsicPipeChann: ClassicPipe, error_gen: Rot_Error, 
                   proto_finished:Event, on_demand:Event, epr_demand_end:Event,
                   len_est:frm_len_EST, sRand, F_thres=0.5, verbose_level=0):
    
    nfID = qmem_itfc.nfID_EPR_START()
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/EPR - send EPR-Frame nfID:{id_epr}".format(id_epr=nfID))
    
    #head_qbit = Qubit(host)
    head_qbit = QuTils.superposed_qubit(host, sigma=0.7)
    head_qbit.send_to(receiver_id=receiver_id)
    error_gen.apply_error(flying_qubit=head_qbit)        
    quPipeChann.put(head_qbit)
    frame_len = len_est.get_frame_len()

    for i in range(frame_len):
        q1, q2 = epr_gen.get_EPR_pair()
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)
        if verbose_level == 1:
            print("ALICE/EPR - send EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q2)
        quPipeChann.put(q2)
        time.sleep(0.002) # for synchronization
    
    
    # Hear classic channel for feedback
    cbit_counter = 0
    clsicPipeChann.fst_feedback_in_trans.wait()
    ip_qids = None
    recv_meas = []
    d_nack = False
    f_thres_recv = ""
    THRES = None
    while True:
        try:
            bit = clsicPipeChann.out_socket.pop()
        except IndexError:
            continue
        else:
            cbit_counter += 1
            if cbit_counter == 1:
                if bit == 0 : # D-NACK
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv D-NACK: dropping nfID and nuID")
                    qmem_itfc.drop_DNACK_EPR_END_PHASE()
                    clsicPipeChann.feedback_num = 0
                    d_nack = True
                    break

                else: # EPR-FEEDBACK
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv EPR-Feedback")
                    ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()
                    continue
            else:
                if cbit_counter < 8:
                    f_thres_recv += str(bit)
                else:
                    recv_meas.append(bit)
                if clsicPipeChann.fst_feedback_in_trans.is_set():
                    continue
                else:
                    break
    if d_nack:
        if on_demand.is_set():
            epr_demand_end.wait()
        else:
            proto_finished.wait()
    else:
        f_thres_recv = (bin2int(f_thres_recv) + 50) / 100
        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - random measurements")       
        #meas_amount is divisible by 3!
        meas_amount = len(ip_qids) - qmem_itfc.eff_load
        # random selection of qubits for FEU
        meas_qids = sRand.sample(ip_qids, int(meas_amount))
        # EPR-FEEDBACK meas order [mx , my, mz]
        local_meas = []
        # qids for X-MEAS 1/3
        x_qids = sRand.sample(meas_qids, int(meas_amount/3))
        for idq in x_qids:
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            mq.H()
            bit = mq.measure()
            local_meas.append(bit)
            meas_qids.remove(idq)
        
        s_dagger_mtrx = np.array([[1,0],[0, 1j]])
        # qids for Y-MEAS 2/3
        y_qids = sRand.sample(meas_qids, int(len(meas_qids)/2))
        for idq in y_qids:
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            mq.custom_gate(s_dagger_mtrx)
            mq.H()
            bit = mq.measure()
            local_meas.append(bit) 
            meas_qids.remove(idq)
        # qids for Z-MEAS 3/3
        for idq in meas_qids:
            mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
            bit = mq.measure()
            local_meas.append(bit)

        if (verbose_level == 0) or (verbose_level == 1):
            print("ALICE/EPR - Fidelity estimation")
        meas_count = 0 
        mxyz_len = len(meas_qids)
        qberx = 0
        qbery = 0 
        qberz = 0
        F_est_dahl = 0
        F_est_intuition = 0
        for idx in range(len(recv_meas)):
            meas_count +=1
            if recv_meas[idx] == local_meas[idx]:
                continue
            else:
                if meas_count < (mxyz_len + 1):
                    qberx += 1
                elif mxyz_len < meas_count < (2*mxyz_len + 1):
                    qbery += 1
                else:
                    qberz += 1
        qberx = qberx / mxyz_len
        qbery = qbery / mxyz_len
        qberz = qberz / mxyz_len

        # as in paper of dahlberg
        #F_est_dahl = 1 - (qberx + qbery + qberz) / 2
        #print(F_est_dahl)
        # as my intuition says
        F_est_intuition = 1 - (qberx + qbery + qberz) / 3
        #print(F_est_intuition)            
        F_send = math.trunc(F_est_intuition * 1000)
        F_EST = F_send / 1000 # to be setted on local qumem itfc
        F_send = F_send - 500

        if F_thres > f_thres_recv:
            THRES =  F_thres
        else:
            THRES = f_thres_recv
        # ACK
        if THRES <= F_EST:
            qmem_itfc.set_F_EPR_END_PHASE(F_EST)
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - uID:{id_u} - Fid:{f}".format(id_u=nfID, f=F_EST))
                print("ALICE/EPR - send EPR-ACK")
            # epr-ack = [1, F_est]
            epr_ack = [1]
            epr_ack.extend(int2bin(F_send, 9))
            for bit in epr_ack: # send epr-ack
                clsicPipeChann.put(bit)
                time.sleep(INTER_CBIT_TIME)
        # NACK
        else:
            qmem_itfc.drop_ip_frame_EPR_END_PHASE()
            if (verbose_level == 0) or (verbose_level == 1):
                print("ALICE/EPR - F_est < F_thres:{fest} < {fth}".format(fest=F_EST, fth=THRES))
                print("ALICE/EPR - send NACK")
            clsicPipeChann.put(0) # send NACK to Bob
        if on_demand.is_set():
            epr_demand_end.wait()
        else:
            proto_finished.wait()

# im_2_send string to be sent!
# SDC-encoded C-Info will always arrive and be decoded at BOB
def sSDC_proto(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, 
                      quPipeChann:QuPipe, clsicPipeChann: ClassicPipe, 
                      error_gen: Rot_Error, proto_finished:Event, 
                      verbose_level=0):
    global im_2_send
    global sent_mssgs
    sent_mssg = ""
    nuID = qmem_itfc.nuID_SDC_START()
    if (verbose_level==0) or (verbose_level==1):
        print("ALICE/SDC - send SDC-Frame nuID:{id_u}".format(id_u=nuID))
    
    #head_qbit = Qubit(host)
    head_qbit = QuTils.superposed_qubit(host, sigma=0.7)
    head_qbit.X()# --> header qubit into state "e_1"
    error_gen.apply_error(flying_qubit=head_qbit)
    head_qbit.send_to(receiver_id=receiver_id)
    quPipeChann.put(head_qbit)
    time.sleep(INTER_QBIT_TIME)
    for mi in range(qmem_itfc.eff_load):
        # taking the first two bits & deleting this two bits from im_2_send
        msg = im_2_send[0:2]
        im_2_send = im_2_send[2:]
        sent_mssg += msg
        q_sdc = qmem_itfc.pop_SDC_END_PHASE()
        QuTils.dens_encode(q_sdc, msg)
        if verbose_level==1:
            print("ALICE/SDC - send SDC-encoded epr half Nr:{q_i}".format(q_i=mi))
        q_sdc.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q_sdc)
        quPipeChann.put(q_sdc)
        time.sleep(INTER_QBIT_TIME)
    sent_mssgs += sent_mssg
    if (verbose_level==0) or (verbose_level==1):
        print("ALICE/SDC - send C-Info: {cmsg}".format(cmsg=sent_mssg))
    proto_finished.wait()


def sender_protocol(host:Host, receiver_id, epr_gen: EPR_generator, 
                    qmem_itfc:EPR_buff_itfc, QpipeChann:QuPipe, 
                    CpipeChann: ClassicPipe, error_gen: Rot_Error, 
                    proto_finished:Event, on_demand:Event, epr_demand_end:Event,
                    finish_simu:Event, len_est:frm_len_EST, sRandom, F_thres=0.5,
                    verbose_level=0):

    if (F_thres < 0.5) or (F_thres > 1):
        raise ValueError("F_thres must live in region 0.5 <= F_thres < 1")
    
    global PROCESS
    sRandom.seed(1100)
    while True:
        try:
            process_type = PROCESS.pop()
        except IndexError:
            continue
        else:
            if process_type == "EPR":
                if qmem_itfc.is_full:
                    print("ALICE/EPR - Memory interface is full.\n")
                    continue
                else:
                    proto_finished.clear()
                    print("\nALICE/EPR - Starting EPR-Frame distribution.")
                    sEPR_proto(host, receiver_id , epr_gen, qmem_itfc, 
                               QpipeChann, CpipeChann, error_gen, 
                               proto_finished, on_demand, epr_demand_end,
                               len_est, sRandom, F_thres, verbose_level)
                    continue
            elif process_type == "SDC":
                if qmem_itfc.is_empty: # ON DEMAND EPR THEN SDC
                    proto_finished.clear()

                    on_demand.set()
                    att_nr = 0

                    while True:
                        att_nr += 1 
                        print("\nALICE/SDC - ON DEMAND EPR distribution attempt {}\n".format(att_nr))
                        sEPR_proto(host, receiver_id , epr_gen, qmem_itfc, 
                                   QpipeChann, CpipeChann, error_gen, 
                                   proto_finished, on_demand, epr_demand_end,
                                   len_est, sRandom, F_thres, verbose_level)
                        epr_demand_end.clear()
                        if not qmem_itfc.is_empty:
                            break
                        else:
                            continue

                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, verbose_level)
                    on_demand.clear()
                    continue
                else: # send SDC-Frame
                    proto_finished.clear()
                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, verbose_level)
                    continue
            else:
                finish_simu.set()
                break
                

def put_next_process(proto_finished:Event, job_prob=0.5):
    global PROCESS
    while True:
        proto_finished.wait()
        # TODO: possible time.sleep() to allow readiness of receiver protocol
        if len(PROCESS) == 0:
            job_prob_var = random.random()
            if job_prob_var > (1 - job_prob):
                PROCESS.append("SDC")
            else:
                PROCESS.append("EPR")
        continue

def receiver_protocol(host:Host, qmem_itfc: EPR_buff_itfc, quPipeChann:QuPipe, 
                      clsicPipeChann: ClassicPipe, proto_finished:Event, 
                      on_demand:Event, epr_demand_end:Event, rRandom, F_thres=0.5,
                      verbose_level=0):

    if (F_thres < 0.5) or (F_thres > 1):
        raise ValueError("F_thres must live in region 0.5 <= F_thres =< 1")

    # for epr-feedback
    F_thres = math.trunc(F_thres * 100) - 50 
    F_thres = int2bin(F_thres, 6) 

    global im_2_send
    global dcdd_mssgs
    global finish_time
    global SAVE_MARIO
    global received_mario
    global PROCESS

    rRandom.seed(1100)
    count = 0
    Type = 0 # 0 means epr frame, 1 means data frame
    dcdd_mssg = None
    f_est_ideal = 0
    frame_id = None
    qbit = None
    while True:
        try:
            qbit = quPipeChann.out_socket.pop()
        except IndexError:
            continue
        else:
            count += 1
            if count == 1:
                head = qbit.measure()

                # SDC-Frame
                if head == 1: 
                    # Header was corrupted, it must have been EPR-Frame
                    if qmem_itfc.is_empty: 
                        Type = 0                        
                        frame_id = qmem_itfc.nfID_EPR_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB/EPR   - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is stored EPR-Frames, interpret payload as SDC-Frame
                    else:
                        Type=1
                        dcdd_mssg = ""
                        frame_id = qmem_itfc.nuID_SDC_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB/SDC   - recv SDC-Frame nuID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                # EPR-Frame
                else:
                    # Header was corrupted, it must have been SDC-Frame
                    if qmem_itfc.is_full:
                        Type=1
                        dcdd_mssg = ""
                        frame_id = qmem_itfc.nuID_SDC_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB/SDC   - recv SDC-Frame nuID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is space in ITFC, interpret payload as EPR-Frame
                    else:
                        Type = 0                        
                        frame_id = qmem_itfc.nfID_EPR_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB/EPR   - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
            else:
                if Type == 0: # receive epr frame
                    if verbose_level == 1: 
                        print("BOB/EPR   - recv EPR halve Nr: {hnum}".format(hnum=(count - 1)))
                    f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)
                    qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)
                else: # receive superdense coded data frame
                    if verbose_level==1:
                        print("BOB/SDC   - recv SDC-encoded epr half Nr:{q_i}".format(q_i=(count - 1)))
                    if qmem_itfc.in_process:
                        retrieved_epr_half = qmem_itfc.pop_SDC_END_PHASE()
                        decoded_string = QuTils.dense_decode(retrieved_epr_half, qbit)
                        dcdd_mssg += decoded_string
                    else:
                        # just for destroying qubit
                        qbit.measure()

                # Q-Frame is still in transmission
                if quPipeChann.Qframe_in_transmission.is_set():
                    continue

                # Q-Frame was  completely received    
                else:
                    if Type == 1: 
                        # SDC-Decoding level
                        if count == (qmem_itfc.eff_load + 1): 
                            if (verbose_level == 0) or (verbose_level == 1):   
                                    print("BOB/SDC   - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                            dcdd_mssgs += dcdd_mssg
                            # IMAGE was completely sent ---> STOP SIMULATION
                            if len(im_2_send) == 0:
                                finish_time = time.time()
                                print("SIMULATION END: Image was completely sent!")
                                received_indexes = []
                                c1 = 0 
                                c2 = 0
                                for i in range(int(len(dcdd_mssgs)/2)):
                                    c1, c2 = tuple(map(int, dcdd_mssgs[i*2:i*2 +2]))
                                    rec_index = integerFromBinaryTuple(c1, c2)
                                    received_indexes.append(rec_index)

                                for row in range(rows):
                                    for column in range(coloumns):
                                        received_hash = hashes[received_indexes[0]]
                                        received_color = palette[received_hash]
                                        received_mario[column, row, :] = received_color
                                        received_indexes.pop(0)

                                received_im = Image.fromarray(received_mario)

                                if SAVE_MARIO:
                                    received_im.save('./mario_link_varying_fid_and_channel.bmp')
                                # PROCESS=[None] stops simulation
                                PROCESS.append(None)
                            dcdd_mssg = None
                            frame_id = None
                            proto_finished.set()
                        else: # D-NACK
                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB/SDC   - recv Payload length({frln}) > eff load({eff}).".format(frln=count-1, 
                                                                                                eff=qmem_itfc.eff_load))
                                print("BOB/SDC   - ===> It was an EPR-Frame")
                                print("BOB/SDC   - dropping decoded C-Info.")
                                print("BOB/SDC   - send D-NACK.")  
                            dcdd_mssg = None
                            frame_id = None

                            # TODO: possible timming problems
                            clsicPipeChann.put(0) # D-NACK
                            if on_demand.is_set():
                                epr_demand_end.set()
                            else:
                                proto_finished.set()                        
                        count = 0 
                        continue # COMM-ENDS: CONTINUE WITH OUTER WHILE LOOP

                    else: # EPR-FEEDBACK LEVEL

                        # It was SDC-Frame
                        if count == (qmem_itfc.eff_load + 1 ):
                            dcdd_mssg = ""
                            uID = qmem_itfc.nuID_CORRECT_epr_as_sdc_EPR_PHASE_2()
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB/EPR---> BOB/SDC - recv EPR-Frame len:{fl} ==> SDC-Frame nuID:{id_u}".format(fl=(count-1), id_u=uID))

                            while qmem_itfc.in_process:
                                recv_qbit, rtrv_qbit = qmem_itfc.pop_sync_SDC_END_PHASE()
                                decoded_string = QuTils.dense_decode(rtrv_qbit, recv_qbit)
                                dcdd_mssg += decoded_string

                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB/SDC   - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                            dcdd_mssgs += dcdd_mssg
                                    
                            # IMAGE was completely sent ---> STOP SIMULATION
                            if len(im_2_send) == 0:
                                finish_time = time.time()
                                print("SIMULATION END: Image was completely sent!")
                                received_indexes = []
                                c1 = 0 
                                c2 = 0
                                for i in range(int(len(dcdd_mssgs)/2)):
                                    c1, c2 = tuple(map(int, dcdd_mssgs[i*2:i*2 +2]))
                                    rec_index = integerFromBinaryTuple(c1, c2)
                                    received_indexes.append(rec_index)

                                for row in range(rows):
                                    for column in range(coloumns):
                                        received_hash = hashes[received_indexes[0]]
                                        received_color = palette[received_hash]
                                        received_mario[column, row, :] = received_color
                                        received_indexes.pop(0)

                                received_im = Image.fromarray(received_mario)

                                if SAVE_MARIO:
                                    received_im.save('./mario_link_varying_fid_and_channel.bmp')
                                # PROCESS=[None] stops simulation
                                PROCESS.append(None)
                            f_est_ideal = 0
                            dcdd_mssg = None
                            frame_id = None
                            proto_finished.set()
                        # EPR-Frame
                        else:
                            f_est_ideal = (f_est_ideal /(count -1))
                            print("BOB/EPR   - Ideal fidelity estimation: {}".format(f_est_ideal))
                            # TODO: store ideal estimated fidelity to history 
                            # Just store frame fidelity before error on itfc history
                            #qmem_itfc.set_F_EPR_END_PHASE(F_est=F_ideal_BF_chann, to_history=True)

                            #in process qids
                            ip_qids = qmem_itfc.get_qids_EPR_PHASE_2()
                            #meas_amount is divisible by 3!
                            meas_amount = len(ip_qids) - qmem_itfc.eff_load
                            
                            if (verbose_level == 0) or (verbose_level == 1):
                                print("BOB/EPR   - {} qubits in payload".format(count -1))
                                print("BOB/EPR   - {} random measurements to be performed".format(meas_amount))
                            # random selection of qubits for FEU
                            meas_qids = rRandom.sample(ip_qids, int(meas_amount))
                            
                            # EPR-FEEDBACK [1, F_thres_B ,mx , my, mz]
                            epr_feed = [1]
                            epr_feed.extend(F_thres)

                            # qids for X-MEAS 1/3
                            x_qids = rRandom.sample(meas_qids, int(meas_amount/3))
                            for idq in x_qids:
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit)
                                meas_qids.remove(idq)
                            
                            s_dagger_mtrx = np.array([[1,0],[0, 1j]])
                            # qids for Y-MEAS 2/3
                            y_qids = rRandom.sample(meas_qids, int(len(meas_qids)/2))
                            for idq in y_qids:
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                mq.custom_gate(s_dagger_mtrx)
                                mq.H()
                                bit = mq.measure()
                                epr_feed.append(bit) 
                                meas_qids.remove(idq)

                            # qids for Z-MEAS 3/3
                            for idq in meas_qids:
                                mq = qmem_itfc.get_epr_EPR_PHASE_3(idq)
                                bit = mq.measure()
                                epr_feed.append(bit)
                                
                            if (verbose_level == 0) or (verbose_level == 1):   
                                print("BOB/EPR   - send EPR-Feedback")
                            # send EPR-FEEDBACK [1, F_thres_B, mx, my, mz]
                            for bit in epr_feed: 
                                clsicPipeChann.put(bit)
                                time.sleep(INTER_CBIT_TIME)

                            clsicPipeChann.snd_feedback_in_trans.wait()

                            F_bits = ""
                            cbit_counter = 0
                            while True:
                                try:
                                    bit = clsicPipeChann.out_socket.pop()
                                except IndexError:
                                    continue
                                else:
                                    cbit_counter += 1
                                    if cbit_counter == 1:
                                        # NACK
                                        if bit == 0: 
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB/EPR   - recv NACK")
                                            qmem_itfc.drop_ip_frame_EPR_END_PHASE()
                                            break
                                        # ACK
                                        else:
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB/EPR   - recv ACK")
                                            continue
                                    else:
                                        F_bits += str(bit)
                                        if clsicPipeChann.snd_feedback_in_trans.is_set():
                                            continue
                                        else:
                                            F_est = (bin2int(F_bits) + 500)/ 1000
                                            qmem_itfc.set_F_EPR_END_PHASE(F_est=F_est)
                                            if (verbose_level == 0) or (verbose_level == 1):
                                                print("BOB/EPR   - uID:{id_u} - Fid:{f}".format(id_u=frame_id, f=F_est))
                                            break
                            if on_demand.is_set():
                                epr_demand_end.set()
                            else:
                                proto_finished.set()   
                        f_est_ideal = 0
                        count=0
                        continue # COMM-ENDS: CONTINUE WITH OUTER WHILE LOOP

def main():
    start_time = time.time()

    VERBOSE = True
    SAVE_DATA = False

    network = Network.get_instance()
    
    Alice = Host('A')
    Bob = Host('B')

    Alice.add_connection('B')
    Bob.add_connection('A')

    Alice.start()
    Bob.start()

    network.add_hosts([Alice, Bob])
    network.start()

    # Initialize needed classes
    Alice_EPR_gen =  EPR_generator(host=Alice, min_fid=0.75, max_dev=0.05, 
                                   f_mu=(1/80), f_sig=(1/160), sig_phase=np.pi)
    Alice_EPR_gen.start()

    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False, 
                                 eff_load=EFF_LOAD)
    qumem_itfc_A.start_time = Alice_EPR_gen.start_time

    len_est = frm_len_EST(eff_load=EFF_LOAD, freq_q=(1/80), phase=np.pi)
    len_est.start_time = Alice_EPR_gen.start_time
    
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True, 
                                 eff_load=EFF_LOAD)
    qumem_itfc_B.start_time = Alice_EPR_gen.start_time

    rot_error = Rot_Error(f_mu=(1/80), f_sig=(1/160) )#, sig_phase=np.pi)
    rot_error.start_time = Alice_EPR_gen.start_time

    if VERBOSE:
        print("Host Alice and Bob started. Network started.")

    if VERBOSE:
        print("Starting communication loop: Alice sender, Bob receiver.\n")

    delay = 2  # 1.2 minimum delay
    Qpiped_channel =  QuPipe(delay=delay)
    Cpiped_channel = ClassicPipe(delay=delay)

    frame_comm_finished = Event()
    on_demand_comm = Event()
    on_demand_epr_finished = Event()
    FINALIZE_simu = Event()
    send_random = random.Random()
    recv_random = random.Random() 

    F_thres_A = 0.65
    F_thres_B = 0.65  

    DaemonThread(target=receiver_protocol, args=(Bob, qumem_itfc_B, 
                                Qpiped_channel, Cpiped_channel, 
                                frame_comm_finished, on_demand_comm, 
                                on_demand_epr_finished, recv_random, F_thres_B,
                                0)) 

    DaemonThread(target=put_next_process, args=(frame_comm_finished, 0.5))

    DaemonThread(target=sender_protocol, args=(Alice, Bob.host_id, Alice_EPR_gen,
                            qumem_itfc_A, Qpiped_channel, Cpiped_channel, rot_error,
                            frame_comm_finished, on_demand_comm, 
                            on_demand_epr_finished, FINALIZE_simu, len_est, 
                            send_random, F_thres_A, 0))

    # force beginning of communication
    frame_comm_finished.set()
    FINALIZE_simu.wait()
    global finish_time
    # TODO: wait for end of simulation to proceed with this lines, through an EVENT()

    print("Simulation time duration in seconds is: {t}".format(
                                                    t=finish_time - start_time))

    print("Alice sent the following classical bitstream:\n"
          "{bstrm}".format(bstrm=sent_mssgs))

    print("\nBob received the following classical bitstream:"
          "\n{rbstrm}".format(rbstrm=dcdd_mssgs))
    
    if SAVE_DATA:

        exp_num = 2

        base_path = "./Playground_v0.8/erp_dist_link_direct_store_data/"


        epr_hist_alice = base_path + "alice_epr_frame_history_exp_" + str(exp_num) + ".csv"
        epr_hist_bob = base_path + "bob_epr_frame_history_exp_" + str(exp_num) + ".csv"

        qumem_itfc_A.EPR_frame_history.to_csv(epr_hist_alice)
        qumem_itfc_B.EPR_frame_history.to_csv(epr_hist_bob)

        sdc_hist_alice = base_path + "alice_sdc_frame_history_exp_" + str(exp_num) + ".csv"
        sdc_hist_bob = base_path + "bob_sdc_frame_history_exp_" + str(exp_num) + ".csv"    

        qumem_itfc_A.SDC_frame_history.to_csv(sdc_hist_alice)
        qumem_itfc_B.SDC_frame_history.to_csv(sdc_hist_bob)

        input_hist_alice = base_path + "alice_in_halves_history_exp_" + str(exp_num) + ".csv"
        input_hist_bob = base_path + "bob_in_halves_history_exp_" + str(exp_num) + ".csv"

        qumem_itfc_A.In_halves_history.to_csv(input_hist_alice)
        qumem_itfc_B.In_halves_history.to_csv(input_hist_bob)

        alice_gen_hist = base_path + "alice_epr_generator_history_exp_" + str(exp_num) + ".csv"
        error_hist = base_path + "applied_error_history_exp_" + str(exp_num) + "csv"

        Alice_EPR_gen.history.to_csv(alice_gen_hist)
        rot_error.history.to_csv(error_hist)

    print(qumem_itfc_A.EPR_frame_history)
    print(qumem_itfc_B.EPR_frame_history)
    
    ax = plt.gca()
          
    Alice_EPR_gen.history.plot(x="time_gauss", y="mu", kind="scatter", ax=ax)
    rot_error.history.plot(x="time_gauss", y="mu", kind="scatter", color="red", ax=ax)
    plt.show()

    Alice_EPR_gen.history.plot(x="time_gauss", y="sig", kind="scatter")
    plt.show()
    rot_error.history.plot(x="time_gauss", y="sig", kind="scatter")
    plt.show()

    Alice_EPR_gen.history.plot(x="time_fid", y="fid", kind="scatter")
    plt.show()
    Alice_EPR_gen.history.plot(x="time_gen", y="fid_gen", kind="scatter")
    plt.show()

    rot_error.history.plot(x="time_gamm", y="gamm", kind="scatter")
    plt.show()
    
    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()    

if __name__=='__main__':
    main()