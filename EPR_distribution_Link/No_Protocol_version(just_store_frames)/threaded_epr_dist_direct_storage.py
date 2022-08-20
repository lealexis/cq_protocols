from qunetsim.components import Network, Host
from qunetsim.objects import Qubit, DaemonThread
import numpy as np 
import random
from PIL import Image
import QuTils
import time
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_interface_class import EPR_buff_itfc
from rotational_error_class import Rot_Error
from channel_pipelines_class import QuPipe
from threading import Event 
from matplotlib import pyplot as plt

DISCRETE_TIME_STEP = 0.05 # 20ms
FRAME_LENGTH = 35 # Load length
mean_gamma = np.pi / 16
channel = True
global Out_QF_receptor
Out_QF_receptor = ([],[])

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

def send_epr_frame(host: Host, receiver_id , epr_gen: EPR_generator, 
                   qmem_itfc: EPR_buff_itfc, error_gen: Rot_Error, 
                   pipe_out:QuPipe, frm_len=FRAME_LENGTH, head_len = 8, 
                   verbose=False):
    f_est_ideal = 0
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=head_qbit)        
        pipe_out.put(head_qbit)
        time.sleep(DISCRETE_TIME_STEP)
    for i in range(frm_len):
        # TODO: A fraction of the EPR-pairs needs to be measured in order to
        if verbose:
            print("Generating EPR pair Nr: {pnum} with Fidelity {F}".format(
                                                              pnum=i+1, F=Fid))
        q1, q2 = epr_gen.get_EPR_pair()
        f_est_ideal += epr_gen.get_fidelity(half=q1)
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)
        if verbose:
            print("Sending EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q2)
        pipe_out.put(q2)
        time.sleep(DISCRETE_TIME_STEP)
    f_est_ideal = (f_est_ideal / frm_len)
    qmem_itfc.set_F_est_EPR_END_PHASE(F_est=f_est_ideal, to_history=True)

# im_2_send string to be sent!
def send_sdense_frame(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, 
                      error_gen: Rot_Error, pipe_out:QuPipe, 
                      frm_len=FRAME_LENGTH, head_len=8, verbose=False):
    global im_2_send
    sent_mssg = ""
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.X()# --> header qubit into state "e_1"
        error_gen.apply_error(flying_qubit=head_qbit)
        head_qbit.send_to(receiver_id=receiver_id)
        pipe_out.put(head_qbit)
        time.sleep(DISCRETE_TIME_STEP)
    for mi in range(frm_len):
        # taking the first two bits & deleting this two bits from im_2_send
        msg = im_2_send[0:2]
        im_2_send = im_2_send[2:]
        sent_mssg += msg
    
        q_sdc = qmem_itfc.pop_SDC_END_PHASE()
        if verbose:
            print("message to be sent is:" + msg)
        QuTils.dens_encode(q_sdc, msg)
        if verbose:
            print("Sending superdense coded epr half Nr:{q_i}".format(q_i=mi))
        q_sdc.send_to(receiver_id=receiver_id)
        error_gen.apply_error(flying_qubit=q_sdc)
        pipe_out.put(q_sdc)
        time.sleep(DISCRETE_TIME_STEP)
    return sent_mssg

def QF_receptor(qf_received:Event, pipe_out:QuPipe, host:Host, 
                qmem_itfc: EPR_buff_itfc, frm_len=FRAME_LENGTH, head_len = 8,
                verbose=False):

    count = 0
    Type = 0 # 0 means epr frame, 1 means data frame
    SDC_count = 0
    EPR_count = 0
    epr_halves = None 
    received_mssg = None
    f_est_ideal = 0
    rec_fr_id = ""
    frame_id = None
    qbit = None
    while True:
        try:
            qbit = pipe_out.out_socket.pop()
        except IndexError:
            continue
        else:
            if count < head_len:
                head = qbit.measure()
                if head == 1: # SDC_count
                    SDC_count += 1
                else: # EPR_count
                    EPR_count += 1 
                count += 1
                if count == head_len:
                    if SDC_count == EPR_count:
                        # Neg_ack can not determine if frame is DATA or EPR
                        # TODO: receiver(host) must communicate failure of the frame
                        raise Exception("Frame not recognized")
                        pass
                        SDC_count = 0
                        EPR_count = 0
                    elif SDC_count > EPR_count: # Superdense data frame
                        Type=1
                        received_mssg = ""
                        frame_id = qmem_itfc.nxt_uID_SDC_START()
                        SDC_count = 0
                        EPR_count = 0
                        if verbose:      
                            print("Receiving a Superdense coded data frame")
                    else: # EPR halves frame
                        Type = 0                        
                        frame_id = qmem_itfc.nxt_fID_EPR_START()
                        SDC_count = 0
                        EPR_count = 0
                        if verbose:
                            print("Receiving an EPR halves frame")
                continue
                
            else:
                if Type == 0: # receive epr frame
                    if verbose: 
                        print("Receiving EPR halve Nr: {hnum}".format(hnum=(count - 
                                                                    head_len + 1)))
                    f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)
                    qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)
                else: # receive superdense coded data frame
                    retrieved_epr_half = qmem_itfc.pop_SDC_END_PHASE()
                    if retrieved_epr_half is None:
                        print("Retrieved Pair was None!")
                    if qbit.id == retrieved_epr_half.id:
                        if verbose:
                            print("Received superdense coded qubit has the same id" 
                                  " as the used epr half!")
                        decoded_string = QuTils.dense_decode(
                                                stored_epr_half=retrieved_epr_half, 
                                                received_qubit=qbit)
                        received_mssg += decoded_string
                count +=1

                if count==(frm_len + head_len):
                    if verbose:
                        print("Frame successfully Received!")
                    global Out_QF_receptor
                    if Type == 1:
                        count = 0
                        Out_QF_receptor[0].append(received_mssg)
                        Out_QF_receptor[1].append(frame_id)
                        qf_received.set()
                        #return received_mssg, frame_id
                    else:
                        count = 0
                        f_est_ideal = (f_est_ideal / frm_len)
                        Out_QF_receptor[0].append(f_est_ideal)
                        Out_QF_receptor[1].append(frame_id)
                        qf_received.set()
                        qmem_itfc.set_F_est_EPR_END_PHASE(F_est=f_est_ideal)
                        f_est_ideal = 0
                        #return f_est_ideal, frame_id
                    continue
                else:
                    continue
                    
def main():
    start_time = time.time()

    VERBOSE = True
    save = False

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

    Alice_EPR_gen =  EPR_generator(host=Alice, min_fid=0.75, max_dev=0.05, 
                                   f_mu=(1/80), f_sig=(1/160), sig_phase=np.pi)
    Alice_EPR_gen.start()

    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False, 
                                 eff_load=FRAME_LENGTH)
    qumem_itfc_A.start_time = Alice_EPR_gen.start_time
    
    
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True, 
                                 eff_load=FRAME_LENGTH)
    qumem_itfc_B.start_time = Alice_EPR_gen.start_time

    rot_error = Rot_Error(f_mu=(1/80), f_sig=(1/160) )#, sig_phase=np.pi)
    rot_error.start_time = Alice_EPR_gen.start_time

    


    if VERBOSE:
        print("Host ALice and Bob started. Network started.")

    choice = ["data", "idle"]
    
    comm_length = 200

    sent_mssgs = ""
    dcdd_mssgs = ""

    if VERBOSE:
        print("Starting communication loop: Alice sender, Bob receiver.")
    epr_frame_counter = 0 
    data_frame_counter = 0


    QF_rcvd = Event()
    global Out_QF_receptor
    piped_channel =  QuPipe(delay=0.5) # 50 ms
    DaemonThread(target=QF_receptor, args=(QF_rcvd, piped_channel, Bob, 
                                           qumem_itfc_B)) 
    for step in range(comm_length):
        state = random.choice(choice)

        if state == "idle": # distribute entanglement
            epr_frame_counter += 1
            
            fIDa = qumem_itfc_A.nxt_fID_EPR_START()
            if VERBOSE:
                print("IDLE - Alice sending EPR Frame using ID:{nr_eprf}".format(
                                                    nr_eprf=fIDa))
            send_epr_frame(Alice, Bob.host_id, Alice_EPR_gen, qumem_itfc_A, 
                                 rot_error, piped_channel)
            QF_rcvd.wait()
            fest = Out_QF_receptor[0].pop()
            fIDb =  Out_QF_receptor[1].pop()
            QF_rcvd.clear()

            if VERBOSE:
                print("IDLE - Bob received EPR Frame using ID:{nr_eprf}".format(
                                                    nr_eprf=fIDb))
            qumem_itfc_A.set_F_est_EPR_END_PHASE(F_est=fest)

        else: # send data frame
            data_frame_counter += 1
            if Alice.shares_epr(Bob.host_id): # just send data

                uIDa = qumem_itfc_A.nxt_uID_SDC_START()
                # sending data frame
                if VERBOSE:
                    print("COMM - Alice sending Data frame using ID:{df} to SDC-encode".format(
                                                        df=uIDa))

                clsic_mssg = send_sdense_frame(Alice, Bob.host_id, qumem_itfc_A, 
                                               rot_error, piped_channel)
                if VERBOSE:
                    print("COMM - Alice did sent {cmsg}".format(
                                                            cmsg=clsic_mssg))
                sent_mssgs +=  clsic_mssg

                QF_rcvd.wait()
                dcdd_mssg = Out_QF_receptor[0].pop()
                uIDb =  Out_QF_receptor[1].pop()
                QF_rcvd.clear()
                
                if VERBOSE:
                    print("COMM - Bob used Frame with ID:{uidb} to SDC-decode".format(uidb=uIDb))
                    print("COMM - Bob received   {cmsg}".format(cmsg=dcdd_mssg))
                dcdd_mssgs += dcdd_mssg
                
            else: # generate epr frame and then consume it
                epr_frame_counter += 1


                fIDa = qumem_itfc_A.nxt_fID_EPR_START()
                if VERBOSE:
                    print("COMM - On demand EPR generation\nAlice is sending EPR"
                          " frame using ID:{ef}".format(ef=fIDa))
               
                send_epr_frame(Alice, Bob.host_id, Alice_EPR_gen, qumem_itfc_A, 
                               rot_error, piped_channel)
                QF_rcvd.wait()
                fest = Out_QF_receptor[0].pop()
                fIDb =  Out_QF_receptor[1].pop()
                QF_rcvd.clear()
                
                if VERBOSE:
                    print("IDLE - Bob received EPR Frame using ID:{idb}".format(
                                                    idb=fIDb))
                qumem_itfc_A.set_F_est_EPR_END_PHASE(F_est=fest)

                uIDa = qumem_itfc_A.nxt_uID_SDC_START()
             
                if VERBOSE:
                    print("COMM - Alice sending Data frame using ID:{df} to SDC-encode".format(
                                                        df=uIDa))

                clsic_mssg = send_sdense_frame(Alice, Bob.host_id, qumem_itfc_A,
                                               rot_error, piped_channel)
                if VERBOSE:
                    print("COMM - Alice did sent {cmsg}".format(
                                                            cmsg=clsic_mssg))
                sent_mssgs +=  clsic_mssg
                
                QF_rcvd.wait()
                dcdd_mssg = Out_QF_receptor[0].pop()
                uIDb =  Out_QF_receptor[1].pop()
                QF_rcvd.clear()
            
                if VERBOSE:
                    print("COMM - Bob used Frame with ID:{uidb} to SDC-decode".format(uidb=uIDb))
                    print("COMM - Bob received   {cmsg}".format(cmsg=dcdd_mssg))
                dcdd_mssgs += dcdd_mssg

            if len(im_2_send) == 0:

                finish_time = time.time()
                print("Image was completely sent")
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
                
                if save:
                    received_im.save('./mario_link_varying_fid_and_channel.bmp')
                break

    print("Simulation time duration in seconds is: {t}".format(
                                                    t=finish_time - start_time))
    print("A total of {nrp} EPR pairs was generated".format(
                                            nrp=epr_frame_counter*FRAME_LENGTH))

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