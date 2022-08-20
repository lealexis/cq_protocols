from qunetsim.components import Network, Host
from qunetsim.objects import Qubit
import numpy as np 
import random
from PIL import Image
import QuTils
import time
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_interface_class import EPR_buff_itfc
from rotational_error_class import Rot_Error
from matplotlib import pyplot as plt

DISCRETE_TIME_STEP = 0.1
FRAME_LENGTH = 35 # Load length
mean_gamma = np.pi / 16
channel = True

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
                   qmem_itfc: EPR_buff_itfc, error_gen: Rot_Error ,frm_len=FRAME_LENGTH,  
                   head_len = 8, verbose=False):
    frame = []
    frame_id = qmem_itfc.nxt_fID_EPR_START()
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.send_to(receiver_id=receiver_id)
        
        error_gen.apply_error(flying_qubit=head_qbit)
        #QuTils.gaussian_xy_rotation_channel(flying_qubit=head_qbit, mu=mean_gamma)
        
        frame.append(head_qbit)
        time.sleep(DISCRETE_TIME_STEP)
    f_est_ideal = 0
    for i in range(frm_len):
        # TODO: A fraction of the EPR-pairs needs to be measured in order to
        if verbose:
            print("Generating EPR pair Nr: {pnum} with Fidelity {F}".format(
                                                              pnum=i+1, F=Fid))
        q1, q2 = epr_gen.get_EPR_pair()

        frame.append(q2)
        f_est_ideal += epr_gen.get_fidelity(half=q1)
        qmem_itfc.store_EPR_PHASE_1(epr_half=q1)

        #sending the qubit q2 to receiver
        if verbose:
            print("Sending EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)

        error_gen.apply_error(flying_qubit=q2)
        #QuTils.gaussian_xy_rotation_channel(flying_qubit=q2, mu=mean_gamma)
        time.sleep(DISCRETE_TIME_STEP)
    f_est_ideal = (f_est_ideal / frm_len)
    qmem_itfc.set_F_est_EPR_END_PHASE(F_est=f_est_ideal, to_history=True)
    return frame

# im_2_send string to be sent!
def send_sdense_frame(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, error_gen: Rot_Error,
                      frm_len=FRAME_LENGTH, head_len=8, verbose=False):
    global im_2_send
    sdense_frame = []
    frame_id = qmem_itfc.nxt_uID_SDC_START()
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.X()# --> header qubit into state "e_1"
        head_qbit.send_to(receiver_id=receiver_id)

        error_gen.apply_error(flying_qubit=head_qbit)
        #QuTils.gaussian_xy_rotation_channel(flying_qubit=head_qbit, mu=mean_gamma)

        sdense_frame.append(head_qbit) # control over sent superdense coded frame
        time.sleep(DISCRETE_TIME_STEP)
    sent_mssg = ""
    for mi in range(frm_len):
        # taking the first two bits & deleting this two bits from im_2_send
        msg = im_2_send[0:2]
        im_2_send = im_2_send[2:]
        sent_mssg += msg
    
        qu_i = qmem_itfc.pop_SDC_END_PHASE()
        if verbose:
            print("message to be sent is:" + msg)
        QuTils.dens_encode(qu_i, msg)
        if verbose:
            print("Sending superdense coded epr half Nr:{q_i}".format(q_i=mi))
        qu_i.send_to(receiver_id=receiver_id)

        error_gen.apply_error(flying_qubit=qu_i)
        #QuTils.gaussian_xy_rotation_channel(flying_qubit=qu_i, mu=mean_gamma)
        sdense_frame.append(qu_i)
        time.sleep(DISCRETE_TIME_STEP)
    return sdense_frame, sent_mssg

def receive_frame(host:Host, qmem_itfc: EPR_buff_itfc, Frame:list,
                  frm_len=FRAME_LENGTH, head_len = 8,verbose=False):
    count = 0
    Type = 0 # 0 means epr frame, 1 means data frame
    SDC_count = 0
    EPR_count = 0
    epr_halves = None 
    received_mssg = None

    f_est_ideal = 0
    rec_fr_id = ""
    frame_id = None
    for qbit in Frame:
        if count < head_len:
            head = qbit.measure()
            if head == 1: # SDC_count
                SDC_count += 1
            else: # EPR_count
                EPR_count += 1 
            count += 1
            #time.sleep(DISCRETE_TIME_STEP)
            if count == head_len:
                if SDC_count == EPR_count:
                    # Neg_ack can not determine if frame is DATA or EPR
                    # TODO: receiver(host) must communicate failure of the frame
                    raise Exception("Frame not recognized")
                    pass
                elif SDC_count > EPR_count: # Superdense data frame
                    Type=1
                    received_mssg = ""
                    frame_id = qmem_itfc.nxt_uID_SDC_START()
                    if verbose:      
                        print("Receiving a Superdense coded data frame")
                else: # EPR halves frame
                    if verbose:
                        print("Receiving an EPR halves frame")
                    frame_id = qmem_itfc.nxt_fID_EPR_START()
        else:
            if Type == 0: # receive epr frame
                if verbose: 
                    print("Receiving EPR halve Nr: {hnum}".format(hnum=(count - 
                                                                head_len + 1)))
                f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)
                qmem_itfc.store_EPR_PHASE_1(epr_half=qbit)
                time.sleep(DISCRETE_TIME_STEP + 0.035)
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
                time.sleep(DISCRETE_TIME_STEP +0.035)
            count +=1
            
    if Type == 0:
        f_est_ideal = (f_est_ideal / frm_len)
        qmem_itfc.set_F_est_EPR_END_PHASE(F_est=f_est_ideal)

    if verbose:
        if ((frm_len + head_len + qmem_itfc.n_exp - 1) == count):
            print("Frame successfully Received!")
    if Type == 1:
        return received_mssg
    else:
        return f_est_ideal

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

    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id, is_receiver=False)
    qumem_itfc_A.start_time = Alice_EPR_gen.start_time
    
    
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id, is_receiver=True)
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

    for step in range(comm_length):
        state = random.choice(choice)

        if state == "idle": # distribute entanglement
            epr_frame_counter += 1
            if VERBOSE:
                print("IDLE - Alice sending EPR Frame Nr:{nr_eprf}".format(
                                                    nr_eprf=epr_frame_counter))
            EPR_frame = send_epr_frame(host=Alice, receiver_id=Bob.host_id, 
                                       epr_gen=Alice_EPR_gen,
                                       qmem_itfc=qumem_itfc_A, error_gen=rot_error)
            if VERBOSE:
                print("IDLE - Bob receiving EPR Frame Nr:{nr_eprf}".format(
                                                    nr_eprf=epr_frame_counter))
            
            fest = receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, Frame=EPR_frame)
            qumem_itfc_A.set_F_est_EPR_END_PHASE(F_est=fest)
        else: # send data frame
            data_frame_counter += 1
            if Alice.shares_epr(Bob.host_id): # just send data

                # sending data frame
                if VERBOSE:
                    print("COMM - Alice sending Data frame Nr:{df}".format(
                                                        df=data_frame_counter))

                SDC_frame, classic_mssg = send_sdense_frame(host=Alice, 
                                                        receiver_id=Bob.host_id,
                                                        qmem_itfc=qumem_itfc_A,
                                                        error_gen=rot_error)
                if VERBOSE:
                    print("COMM - Alice is sending {cmsg}".format(
                                                            cmsg=classic_mssg))
                sent_mssgs +=  classic_mssg

                # receiving qu data frame
                dcdd_mssg = receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, 
                                          Frame=SDC_frame)
                
                if VERBOSE:
                    print("COMM - Bob received {cmsg}".format(cmsg=dcdd_mssg))
                dcdd_mssgs += dcdd_mssg
                
            else: # generate epr frame and then consume it
                epr_frame_counter += 1
                if VERBOSE:
                    print("COMM - On demand EPR generation\nAlice is sending EPR"
                          " frame Nr:{ef}".format(ef=epr_frame_counter))
                # sending and receiving epr_frame
                EPR_frame = send_epr_frame(host=Alice, receiver_id=Bob.host_id, 
                                           epr_gen=Alice_EPR_gen, 
                                           qmem_itfc=qumem_itfc_A, error_gen=rot_error)
                if VERBOSE:
                    print("COMM - On demand EPR generation\nBob is receiving "
                          "EPR frame Nr:{ef}".format(ef=epr_frame_counter)) 
        
                fest = receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, Frame=EPR_frame)
                qumem_itfc_A.set_F_est_EPR_END_PHASE(F_est=fest)

                SDC_frame, classic_mssg = send_sdense_frame(host=Alice, 
                                                        receiver_id=Bob.host_id,
                                                        qmem_itfc=qumem_itfc_A,
                                                        error_gen=rot_error)
                if VERBOSE:
                    print("COMM - Alice is sending {cmsg}".format(
                                                            cmsg=classic_mssg))
                sent_mssgs +=  classic_mssg
                # receiving qu data frame
                if VERBOSE:
                    print("COMM - Bob is receiving Data frame"
                          " Nr:{df}".format(df=data_frame_counter))
            
                dcdd_mssg = receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, 
                                          Frame=SDC_frame)

                if VERBOSE:
                    print("COMM - Bob received {cmsg}".format(cmsg=dcdd_mssg))
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