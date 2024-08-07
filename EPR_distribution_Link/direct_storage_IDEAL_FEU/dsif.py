
from qunetsim.components import Network, Host
from qunetsim.objects import Qubit, DaemonThread
import numpy as np 
import random
from PIL import Image
import time

import QuTils
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_itfc_DS_IDEAL_FEU import EPR_buff_itfc
from rotational_error_class import Rot_Error
from pipelined_CQ_channel import QuPipe, ClassicPipe

from threading import Event 
from matplotlib import pyplot as plt
import math

INTER_CBIT_TIME = 0.008 # 5ms
INTER_QBIT_TIME = 0.023 # 15ms
FRAME_LENGTH = 40 # Load length - 560 bits to be sent
SIG = 0.15 # standard deviation for rotational angle of header qubit

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

im = Image.open("../../../Data_results/mario_sprite.bmp")
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
                   frm_len=FRAME_LENGTH, verbose_level=0):
    F_ideal_BF_chann = 0
    nfID = qmem_itfc.nfID_EPR_START()
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/EPR - send EPR-Frame nfID:{id_epr}".format(id_epr=nfID))
    
    head_qbit = QuTils.superposed_qubit(host, sigma=SIG)
    head_qbit.send_to(receiver_id=receiver_id)
    error_gen.apply_error(flying_qubit=head_qbit)        
    quPipeChann.put(head_qbit)
    for i in range(frm_len):
        # TODO: A fraction of the EPR-pairs needs to be measured for end proto version
        q1, q2 = epr_gen.get_EPR_pair()
        F_ideal_BF_chann += epr_gen.get_fidelity(half=q1)
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)
        if verbose_level == 1:
            print("ALICE/EPR - send EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q2)
        quPipeChann.put(q2)
    F_ideal_BF_chann = (F_ideal_BF_chann / frm_len)
    # Just store frame fidelity before error on itfc history
    qmem_itfc.set_F_EPR_END_PHASE(F_est=F_ideal_BF_chann, to_history=True)
    
    # Hear classic channel for feedback
    cbit_counter = 0
    F_bits = ""
    clsicPipeChann.fst_feedback_in_trans.wait()
    while True:
        try:
            bit = clsicPipeChann.out_socket.pop()
        except IndexError:
            continue
        else:
            cbit_counter += 1
            if cbit_counter == 1:
                if bit == 0:  # EPR
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv EPR-Feedback")
                    continue
                else: # Bob interpreted the QFrame as SDC -> CORRECT IT !
                    ipID, nuID = qmem_itfc.drop_ipID_and_nuID_EPR_END_PHASE()
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - recv SDC-Feedback: dropping (nfID,nuID)=({ip},{u})".format(ip=ipID, u=nuID))
                        print("ALICE/EPR - send EPR-ACK")
                    clsicPipeChann.put(0) # SDC-NACK = EPR-ACK               
                    break
            else:
                F_bits += str(bit)
                if cbit_counter == 11:
                    F_est = bin2int(F_bits) / 1000
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/EPR - uID:{id_u} - Fid:{f}".format(id_u=nfID, f=F_est))
                        print("ALICE/EPR - send EPR-ACK")
                    clsicPipeChann.put(0) # Send EPR-ACK 
                    qmem_itfc.set_F_EPR_END_PHASE(F_est=F_est)
                    break
                else:
                    continue
    if on_demand.is_set():
        epr_demand_end.wait()
    else:
        proto_finished.wait()

# im_2_send string to be sent!
# SDC-encoded C-Info will always arrive and be decoded at BOB
def sSDC_proto(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, 
                      quPipeChann:QuPipe, clsicPipeChann: ClassicPipe, 
                      error_gen: Rot_Error, proto_finished:Event, frm_len=FRAME_LENGTH, 
                      verbose_level=0):
    global im_2_send
    global sent_mssgs
    sent_mssg = ""
    nuID = qmem_itfc.nuID_SDC_START()
    if (verbose_level==0) or (verbose_level==1):
        print("ALICE/SDC - send SDC-Frame nuID:{id_u}".format(id_u=nuID))
    
    head_qbit = QuTils.superposed_qubit(host, sigma=SIG)
    head_qbit.X()# --> header qubit into state "e_1"
    error_gen.apply_error(flying_qubit=head_qbit)
    head_qbit.send_to(receiver_id=receiver_id)
    quPipeChann.put(head_qbit)
    time.sleep(INTER_QBIT_TIME)
    for mi in range(frm_len):
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
    if (verbose_level == 0) or (verbose_level == 1):
        print("ALICE/SDC - send C-Info: {cmsg}".format(cmsg=sent_mssg))
    
    # Hear classic channel for feedback
    cbit_counter = 0
    clsicPipeChann.fst_feedback_in_trans.wait()
    while True:
        try:
            bit = clsicPipeChann.out_socket.pop()
        except IndexError:
            continue
        else:
            cbit_counter += 1 
            if cbit_counter == 1:            
                if bit == 1:  # SDC-Feedback
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/SDC - recv SDC-Feedback")
                        print("ALICE/SDC - send SDC-ACK")
                    clsicPipeChann.put(1)
                    break
                else:  # EPR-Feedback
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/SDC - recv EPR-Feedback")
                    continue
            else:
                if (cbit_counter < 11) and clsicPipeChann.fst_feedback_in_trans.is_set():
                    # TODO: possibly time.sleep()
                    continue
                else:
                    if (verbose_level == 0) or (verbose_level == 1):
                        print("ALICE/SDC - send SDC-ACK")
                    clsicPipeChann.put(1)
                    # possible 
                    # TODO: wait timeout so that Bob can perform sync SDC decoding if needed
                    break
    proto_finished.wait()
    qmem_itfc.finish_SDC()

def sender_protocol(host:Host, receiver_id, epr_gen: EPR_generator, 
                    qmem_itfc: EPR_buff_itfc, QpipeChann: QuPipe,
                    CpipeChann: ClassicPipe, error_gen: Rot_Error, 
                    proto_finished: Event, on_demand: Event, epr_demand_end: Event,
                    finish_simu: Event, frm_len=FRAME_LENGTH, verbose_level=0):
    global PROCESS
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
                               frm_len, verbose_level)
                    continue
            elif process_type == "SDC":
                if qmem_itfc.is_empty:  # ON DEMAND EPR THEN SDC
                    proto_finished.clear()
                    on_demand.set()
                    
                    print("\nALICE/SDC - ON DEMAND EPR distribution\n")
                    sEPR_proto(host, receiver_id , epr_gen, qmem_itfc, 
                               QpipeChann, CpipeChann, error_gen, 
                               proto_finished, on_demand, epr_demand_end,
                               frm_len, verbose_level)
                    
                    epr_demand_end.clear()
                    
                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, frm_len, verbose_level)
                    on_demand.clear()
                    continue
                else:  # send SDC-Frame
                    proto_finished.clear()
                    print("\nALICE/SDC - Starting SDC-Frame communication.")
                    sSDC_proto(host, receiver_id, qmem_itfc, QpipeChann, CpipeChann, 
                               error_gen, proto_finished, frm_len, verbose_level)
                    continue
            else:
                finish_simu.set()
                break
                

def put_next_process(proto_finished: Event, job_prob=0.5):
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

def receiver_protocol(host: Host, qmem_itfc: EPR_buff_itfc, quPipeChann: QuPipe,
                      clsicPipeChann: ClassicPipe, proto_finished: Event,
                      on_demand: Event, epr_demand_end: Event,
                      frm_len=FRAME_LENGTH, verbose_level=0):
    global im_2_send
    global dcdd_mssgs
    global finish_time
    global SAVE_MARIO
    global received_mario
    global PROCESS

    count = 0
    Type = 0  # 0 means epr frame, 1 means data frame
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
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is stored EPR-Frames, interpret payload as SDC-Frame
                    else:
                        Type=1
                        dcdd_mssg = ""
                        frame_id = qmem_itfc.nuID_SDC_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}".format(
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
                            print("BOB  /SDC - recv SDC-Frame nuID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
                    # There is space in ITFC, interpret payload as EPR-Frame
                    else:
                        Type = 0                        
                        frame_id = qmem_itfc.nfID_EPR_START()
                        if (verbose_level == 0) or (verbose_level == 1):    
                            print("BOB  /EPR - recv EPR-Frame nfID:{id_u}".format(
                                                    id_u=frame_id))
                        continue
            else:
                if Type == 0:  # receive epr frame
                    if verbose_level == 1: 
                        print("BOB  /EPR - recv EPR halve Nr: {hnum}".format(hnum=(count - 1)))
                    f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)
                    qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)
                else: # receive superdense coded data frame
                    if verbose_level==1:
                        print("BOB  /SDC - recv SDC-encoded epr half Nr:{q_i}".format(q_i=(count - 1)))
                    retrieved_epr_half = qmem_itfc.pop_SDC_END_PHASE()
                    decoded_string = QuTils.dense_decode(retrieved_epr_half, qbit)
                    dcdd_mssg += decoded_string

                # Q-Frame is still in transmission
                if quPipeChann.Qframe_in_transmission.is_set():
                    continue

                # Q-Frame was  completely received
                # Send Type dependent Feedback and wait for last ACK/NACK LEVEL    
                else:
                    count = 0 
                    if Type == 1:  # SDC-FEEDBACK LEVEL
                        if (verbose_level == 0) or (verbose_level == 1):   
                            print("BOB  /SDC - send SDC-Feedback")
                        clsicPipeChann.put(1)  # send SDC feedback [1]
                        clsicPipeChann.snd_feedback_in_trans.wait()
                        while True:  # LAST SDC/EPR-ACK LEVEL
                            try:
                                bit = clsicPipeChann.out_socket.pop()
                            except IndexError:
                                continue
                            else:
                                if bit == 1:  # SDC-ACK decoded C-Info is valid
                                    if (verbose_level == 0) or (verbose_level == 1):   
                                        print("BOB  /SDC - recv SDC-ACK C-Info is valid.")
                                        print("BOB  /SDC - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                                    dcdd_mssgs += dcdd_mssg
                                    qmem_itfc.finish_SDC(val_C_info=int(1))

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
                                        PROCESS.append(None)  # Force simulation end
                                    dcdd_mssg = None
                                    frame_id = None
                                    proto_finished.set()
                                    break  # COMM-ENDS: BREAKING INNER WHILE LOOP

                                else:  # EPR-ACK ---> decoded C-Info is invalid
                                    if (verbose_level == 0) or (verbose_level == 1):   
                                        print("BOB  /SDC - recv EPR-ACK C-Info is invalid.")
                                        print("BOB  /SDC - dropping decoded C-Info.")  
                                    qmem_itfc.finish_SDC(val_C_info=int(0))
                                    dcdd_mssg = None
                                    frame_id = None
                                    if on_demand.is_set():
                                        epr_demand_end.set()
                                    else:
                                        proto_finished.set()                        
                                    break  # COMM-ENDS: BREAKING INNER WHILE LOOP
                        continue  # COMM-ENDS: CONTINUE WITH OUTER WHILE LOOP

                    else:  # EPR-FEEDBACK LEVEL
                        if (verbose_level == 0) or (verbose_level == 1):   
                            print("BOB  /EPR - send EPR-Feedback")
                        f_est_ideal = (f_est_ideal / frm_len)
                        f_est_ideal = math.trunc(f_est_ideal * 1000)
                        
                        
                        epr_feed = [0]
                        epr_feed.extend(int2bin(f_est_ideal, 10))
                        f_est_ideal = (f_est_ideal / 1000)
                        for bit in epr_feed:  # send EPR feedback [0, f_est_ideal]
                            time.sleep(INTER_CBIT_TIME) 
                            clsicPipeChann.put(bit) 
                            # TODO: adjust time.sleep()
                        clsicPipeChann.snd_feedback_in_trans.wait()
                        while True:
                            try:
                                bit = clsicPipeChann.out_socket.pop()
                            except IndexError:
                                continue
                            else:
                                if bit == 0: # EPR-ACK
                                    if (verbose_level == 0) or (verbose_level == 1):   
                                        print("BOB  /EPR - recv EPR-ACK")
                                        print("BOB  /EPR - uID:{id_u} - Fid:{f}".format(id_u=frame_id, f=f_est_ideal))
                                    qmem_itfc.set_F_EPR_END_PHASE(f_est_ideal)
                                    f_est_ideal = 0
                                    frame_id = None
                                    if on_demand.is_set():
                                        epr_demand_end.set()
                                    else:
                                        proto_finished.set() 
                                    break  # COMM-ENDS: BREAKING INNER WHILE LOOP
                                else:  # SDC-ACK - Correct EPR as SDC
                                    if (verbose_level == 0) or (verbose_level == 1):
                                        print("BOB  /EPR - recv SDC-ACK")
                                        print("BOB  /EPR - correct as SDC")   
                                    dcdd_mssg = ""
                                    uID = qmem_itfc.nuID_CORRECT_epr_as_sdc_EPR_PHASE_2()
                                    if (verbose_level == 0) or (verbose_level == 1):
                                        print("BOB/EPR---> BOB/SDC - nuID:{id_u}".format(id_u=uID))

                                    while qmem_itfc.in_process:
                                        recv_qbit, rtrv_qbit = qmem_itfc.pop_sync_SDC_END_PHASE()
                                        decoded_string = QuTils.dense_decode(rtrv_qbit, recv_qbit)
                                        dcdd_mssg += decoded_string

                                    if (verbose_level == 0) or (verbose_level == 1):   
                                        print("BOB  /SDC - recv C-Info: {cmsg}".format(cmsg=dcdd_mssg))
                                    dcdd_mssgs += dcdd_mssg
                                    #qmem_itfc.finish_SDC(val_C_info=int(1))
                                    
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
                                    break  # COMM-ENDS: BREAKING INNER WHILE LOOP
                        continue  # COMM-ENDS: CONTINUE WITH OUTER WHILE LOOP
                        
def main():
    start_time = time.time()

    VERBOSE = True
    SAVE_DATA = False
    FINAL_EXP = True
    EXP_1 = False
    PRINT_HIST = True

    network = Network.get_instance()
    
    Alice = Host('A')
    Bob = Host('B')

    Alice.add_connection('B')
    Bob.add_connection('A')

    Alice.start()
    Bob.start()

    network.add_hosts([Alice, Bob])
    network.start()

    if FINAL_EXP:
        if EXP_1:
            # use min_fid = 0.5
            freq_mu = 1/300
            freq = 1/300
            mu_phi = 0
            gamm_mu_phi = np.pi
            phi = np.pi
        else:
            # use min_fid = 0.8
            freq_mu = 1/300
            freq = 1/600
            mu_phi = np.pi
            gamm_mu_phi = np.pi
            phi = np.pi

    # Initialize needed classes
    Alice_EPR_gen = EPR_generator(host=Alice, max_fid=0.99, min_fid=0.5,
                                   max_dev=0.15, min_dev=0.015, f_mu=freq_mu, 
                                   f_sig=freq, mu_phase=mu_phi, sig_phase=phi)
    Alice_EPR_gen.start()

    rot_error = Rot_Error(max_rot=0.45, min_rot=0.05, max_dev=0.1, min_dev=0.03, 
                          f_mu=freq_mu, f_sig=freq, mu_phase=gamm_mu_phi, 
                          sig_phase=phi)
    rot_error.start_time = Alice_EPR_gen.start_time

    delay = 4  # 1.2 minimum delay
    Qpiped_channel = QuPipe(delay=delay)
    Cpiped_channel = ClassicPipe(delay=delay)
    
    # Qumem ITFCs

    # Alice
    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False, 
                                 eff_load=FRAME_LENGTH)
    qumem_itfc_A.start_time = Alice_EPR_gen.start_time
    
    # Bob
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True, 
                                 eff_load=FRAME_LENGTH)
    qumem_itfc_B.start_time = Alice_EPR_gen.start_time


    if VERBOSE:
        print("Host Alice and Bob started. Network started.")
        print("Starting communication.\n")

    frame_comm_finished = Event()
    on_demand_comm = Event()
    on_demand_epr_finished = Event()
    FINALIZE_simu = Event()  

    Job_arrival_prob = 0.35 

    DaemonThread(target=receiver_protocol, args=(Bob, qumem_itfc_B, 
                                Qpiped_channel, Cpiped_channel, 
                                frame_comm_finished, on_demand_comm, 
                                on_demand_epr_finished, FRAME_LENGTH, 0)) 

    DaemonThread(target=put_next_process, args=(frame_comm_finished, Job_arrival_prob))

    DaemonThread(target=sender_protocol, args=(Alice, Bob.host_id, Alice_EPR_gen,
                            qumem_itfc_A, Qpiped_channel, Cpiped_channel, rot_error,
                            frame_comm_finished, on_demand_comm, 
                            on_demand_epr_finished, FINALIZE_simu, FRAME_LENGTH, 0))

    # force beginning of communication
    frame_comm_finished.set()
    # waiting for end of simulation
    FINALIZE_simu.wait()
    global finish_time

    print("\nSimulation time duration in seconds is: {t}".format(
                                                    t=finish_time - start_time))

    print("\nAlice sent the following classical bitstream:\n"
          "{bstrm}".format(bstrm=sent_mssgs))

    print("\nBob received the following classical bitstream:"
          "\n{rbstrm}".format(rbstrm=dcdd_mssgs))
    # TODO: Actualize path
    BASE_PATH = "./Analysis_plots/Proto_experiments_&_Plots/"
    
    if SAVE_DATA:
        if FINAL_EXP:
            if EXP_1:
                DATA_PATH = BASE_PATH + "final_exps/EXP_1/DS_IF/data/"
                exp_typ = 1
            else:
                DATA_PATH = BASE_PATH + "final_exps/EXP_2/DS_IF/data/"
                exp_typ = 2
        else:
            DATA_PATH = BASE_PATH + "preliminar_exps/DS_IF/data/"
            exp_typ = "PRE"
        
        # QUMEM ITFCs HISTORY
        epr_hist_alice = DATA_PATH + "alice_epr_frame_history_exp_" + str(exp_typ) + ".csv"
        epr_hist_bob = DATA_PATH + "bob_epr_frame_history_exp_" + str(exp_typ) + ".csv"
        sdc_hist_alice = DATA_PATH + "alice_sdc_frame_history_exp_" + str(exp_typ) + ".csv"
        sdc_hist_bob = DATA_PATH + "bob_sdc_frame_history_exp_" + str(exp_typ) + ".csv"  
        input_hist_alice = DATA_PATH + "alice_in_halves_history_exp_" + str(exp_typ) + ".csv"
        input_hist_bob = DATA_PATH + "bob_in_halves_history_exp_" + str(exp_typ) + ".csv"

        qumem_itfc_A.EPR_frame_history.to_csv(epr_hist_alice)
        qumem_itfc_B.EPR_frame_history.to_csv(epr_hist_bob)
        qumem_itfc_A.SDC_frame_history.to_csv(sdc_hist_alice)
        qumem_itfc_B.SDC_frame_history.to_csv(sdc_hist_bob)
        qumem_itfc_A.In_halves_history.to_csv(input_hist_alice)
        qumem_itfc_B.In_halves_history.to_csv(input_hist_bob)

        # EPR GEN & APPLIED ERROR HISTORIES
        alice_gen_hist = DATA_PATH + "alice_epr_generator_history_exp_" + str(exp_typ) + ".csv"
        error_hist = DATA_PATH + "applied_error_history_exp_" + str(exp_typ) + ".csv"

        Alice_EPR_gen.history.to_csv(alice_gen_hist)
        rot_error.history.to_csv(error_hist)
    
    if PRINT_HIST:
        print(qumem_itfc_A.EPR_frame_history)
        print(qumem_itfc_B.EPR_frame_history)
        print(qumem_itfc_A.SDC_frame_history)
        print(qumem_itfc_B.SDC_frame_history)
        print(qumem_itfc_A.In_halves_history)
        print(qumem_itfc_B.In_halves_history)
    
    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()    

if __name__=='__main__':
    main()