
# export PYTHONPATH=$PYTHONPATH:$PWD/QuNetSim/
from qunetsim.components import Network, Host
from qunetsim.objects import Qubit
import numpy as np 
import random
from PIL import Image
import QuTils
import time
from epr_gen_class import EPR_generator, EPR_Pair_fidelity
from ent_buff_interface_class import EPR_buff_itfc
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
                   qmem_itfc: EPR_buff_itfc, frm_len=FRAME_LENGTH,  
                   head_len = 8, verbose=False):
    frame = []
    frame_id = qmem_itfc.get_next_free_frID()
    bin_frame_id = int2bin(i=frame_id, length=qmem_itfc.n_exp)
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.send_to(receiver_id=receiver_id)
        
        QuTils.gaussian_xy_rotation_channel(flying_qubit=head_qbit, mu=mean_gamma)
        
        frame.append(head_qbit)
        time.sleep(DISCRETE_TIME_STEP)
    for bit in bin_frame_id:
        id_qubit = Qubit(host)
        if bit==1:
            id_qubit.X()
        id_qubit.send_to(receiver_id=receiver_id)

        QuTils.gaussian_xy_rotation_channel(flying_qubit=id_qubit, mu=mean_gamma)

        frame.append(id_qubit)
        time.sleep(DISCRETE_TIME_STEP)
    f_est_ideal = 0
    for i in range(frm_len):
        # TODO: A fraction of the EPR-pairs needs to be measured in order to
        if verbose:
            print("Generating EPR pair Nr: {pnum} with Fidelity {F}".format(
                                                              pnum=i+1, F=Fid))
        q1, q2 = epr_gen.get_EPR_pair()

        frame.append(q2)
        f_est_ideal += epr_gen.get_fidelity(half=q2)
        qmem_itfc.store_epr_half_in_fr_ID(frame_id=frame_id, epr_half=q1)

        #sending the qubit q2 to receiver
        if verbose:
            print("Sending EPR halve Nr:{pnum}".format(pnum=i+1))
        q2.send_to(receiver_id=receiver_id)

        QuTils.gaussian_xy_rotation_channel(flying_qubit=q2, mu=mean_gamma)
        time.sleep(DISCRETE_TIME_STEP)
    f_est_ideal = (f_est_ideal / frm_len)
    qmem_itfc.set_F_est(F_est=f_est_ideal, frame_id=frame_id)
    return frame

# im_2_send string to be sent!
def send_sdense_frame(host:Host, receiver_id, qmem_itfc: EPR_buff_itfc, 
                      frm_len=FRAME_LENGTH, head_len=8, verbose=False):
    global im_2_send
    sdense_frame = []
    frame_id = qmem_itfc.get_next_used_frID()
    bin_frame_id = int2bin(i=frame_id, length=qmem_itfc.n_exp)
    for _ in range(head_len):
        head_qbit = Qubit(host)
        head_qbit.X()# --> header qubit into state "e_1"
        head_qbit.send_to(receiver_id=receiver_id)

        QuTils.gaussian_xy_rotation_channel(flying_qubit=head_qbit, mu=mean_gamma)

        sdense_frame.append(head_qbit) # control over sent superdense coded frame
        time.sleep(DISCRETE_TIME_STEP)
    for bit in bin_frame_id:
        id_qubit = Qubit(host)
        if bit==1:
            id_qubit.X()
        id_qubit.send_to(receiver_id=receiver_id)

        QuTils.gaussian_xy_rotation_channel(flying_qubit=id_qubit, mu=mean_gamma)

        sdense_frame.append(id_qubit)
        time.sleep(DISCRETE_TIME_STEP)
    sent_mssg = ""
    
    for mi in range(frm_len):
        # taking the first two bits & deleting this two bits from im_2_send
        msg = im_2_send[0:2]
        im_2_send = im_2_send[2:]
        sent_mssg += msg
    
        qu_i = qmem_itfc.pop_oldest_epr_half_from_Frame(frame_id=frame_id)
        if verbose:
            print("message to be sent is:" + msg)
        QuTils.dens_encode(qu_i, msg)
        if verbose:
            print("Sending superdense coded epr half Nr:{q_i}".format(q_i=mi))
        qu_i.send_to(receiver_id=receiver_id)

        QuTils.gaussian_xy_rotation_channel(flying_qubit=qu_i, mu=mean_gamma)
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
            if count == head_len:
                if SDC_count == EPR_count:
                    # Neg_ack can not determine if frame is DATA or EPR
                    # TODO: receiver(host) must communicate failure of the frame
                    raise Exception("Frame not recognized")
                    pass
                elif SDC_count > EPR_count: # Superdense data frame
                    Type=1
                    received_mssg = ""
                    if verbose:      
                        print("Receiving a Superdense coded data frame")
                else: # EPR halves frame
                    if verbose:
                        print("Receiving an EPR halves frame")
                    pass   
        elif count < (head_len + qmem_itfc.n_exp):
            bit_id = qbit.measure()
            rec_fr_id += str(bit_id)
            count += 1
            if count == (head_len + qmem_itfc.n_exp):
                rec_fr_id = bin2int(rec_fr_id)
                if Type == 0: 
                    frame_id = qmem_itfc.get_next_free_frID()
                else:
                    frame_id = qmem_itfc.get_next_used_frID()
                if frame_id == rec_fr_id:
                    print("Received frame id and from qmem interface getted next to use frame id are equal.")
                else:
                    print("Received frame id is {rid} and frame id selected from interface is {iid}.".format(
                        rid=rec_fr_id, iid=frame_id)) 
        else:
            if Type == 0: # receive epr frame
                if verbose: 
                    print("Receiving EPR halve Nr: {hnum}".format(
                                  hnum=count - head_len - qmem_itfc.n_exp + 1))
                f_est_ideal += EPR_Pair_fidelity(epr_halve=qbit)
                qmem_itfc.store_epr_half_in_fr_ID(frame_id=frame_id, 
                                                                 epr_half=qbit)
            else: # receive superdense coded data frame
                retrieved_epr_half = qmem_itfc.pop_oldest_epr_half_from_Frame(
                                                             frame_id=frame_id)
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
            
    if Type == 0:
        f_est_ideal = (f_est_ideal / frm_len)
        qmem_itfc.set_F_est(F_est=f_est_ideal, frame_id=frame_id)

    if verbose:
        if ((frm_len + head_len + qmem_itfc.n_exp - 1) == count):
            print("Frame successfully Received!")
    if Type == 1:
        return received_mssg

def main():
    start_time = time.time()

    VERBOSE = True
    save = False

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
                                   f_mu=(1/60), f_sig=(1/30))
    qumem_itfc_A = EPR_buff_itfc(Alice, Bob.host_id)
    qumem_itfc_B = EPR_buff_itfc(Bob, Alice.host_id)

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
                                       qmem_itfc=qumem_itfc_A)
            if VERBOSE:
                print("IDLE - Bob receiving EPR Frame Nr:{nr_eprf}".format(
                                                    nr_eprf=epr_frame_counter))
            
            receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, Frame=EPR_frame)

        else: # send data frame
            data_frame_counter += 1
            if Alice.shares_epr(Bob.host_id): # just send data

                # sending data frame
                if VERBOSE:
                    print("COMM - Alice sending Data frame Nr:{df}".format(
                                                        df=data_frame_counter))

                SDC_frame, classic_mssg = send_sdense_frame(host=Alice, 
                                                        receiver_id=Bob.host_id,
                                                        qmem_itfc=qumem_itfc_A)
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
                                           qmem_itfc=qumem_itfc_A)
                if VERBOSE:
                    print("COMM - On demand EPR generation\nBob is receiving "
                          "EPR frame Nr:{ef}".format(ef=epr_frame_counter)) 
        
                receive_frame(host=Bob, qmem_itfc=qumem_itfc_B, Frame=EPR_frame)

                SDC_frame, classic_mssg = send_sdense_frame(host=Alice, 
                                                        receiver_id=Bob.host_id,
                                                        qmem_itfc=qumem_itfc_A)
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

    
    print("\nFinishing simulation!")
    Alice.stop()
    Bob.stop()
    network.stop()    
    
    Alice_EPR_gen.history.plot(x="time_gauss", y="mu", kind="scatter")
    plt.show()
    Alice_EPR_gen.history.plot(x="time_gauss", y="sig", kind="scatter")
    plt.show()
    Alice_EPR_gen.history.plot(x="time_fid", y="fid", kind="scatter")
    plt.show()
    Alice_EPR_gen.history.plot(x="time_gen", y="fid_gen", kind="scatter")
    plt.show()
if __name__=='__main__':
    main()