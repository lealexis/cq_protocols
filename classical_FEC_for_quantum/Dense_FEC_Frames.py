import random

from qunetsim.components import Host
from qunetsim.components import Network
from qunetsim.objects import Logger, Qubit
from PIL import Image
import numpy as np
import unireedsolomon as rs
from threading import Thread
import time
import depolarizing_channel

q1_id = ''
q2_id = ''
time_delay = 0
DISCRETE_TIME_STEP = 0
flag_sent_message_odd = 0

Logger.DISABLED = False
im = Image.open("received_mario_sprite.bmp")
pixels = np.array(im)
im.close()

frame = []
total_message = ''

'Reed Solomon Type Setup'
N = 90
K = 40
P = N - K
coder = rs.RSCoder(N, K)


def encode(coder, msg):
    """Convert the msg to UTF-8-encoded bytes and encode with "coder".  Return as bytes."""
    return coder.encode(msg.encode('utf8')).encode('latin1')


def decode(coder, encoded):
    """Decode the encoded message with "coder", convert result to bytes and decode UTF-8.
    """
    return coder.decode(encoded)[0].encode('latin1').decode('latin1')


row_counter = 0

'Frame length is decided by the no. of rows there are in the image: 20 Rows = 20 Frames each of '

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

received_mario = np.zeros((coloumns, rows, colors), dtype=dtype)

# get integer from binary tuple


c1 = 0
c2 = 0

recc_mess = []


def gen_epr_pair(host: Host):
    halve_1st = Qubit(host)
    halve_2nd = Qubit(host, q_id=halve_1st.id)
    halve_1st.H()
    halve_1st.cnot(halve_2nd)
    return halve_1st, halve_2nd


def send_epr_frames(host: Host, receiver: str):
    host_to_keep_epr = []
    frame_to_send_epr = []
    epr_counter = 0
    global total_message
    size = int(len(total_message) / 2)
    print('---- Generating EPR frames before beginning communication ---- \n')

    'EPR frames is currently set to a safety value of 350. It will be adapted'
    ' according to the required frame - later '
    print('Sending EPR frames ... ')
    for i in range(360):
        q1, q2 = gen_epr_pair(host=host)
        host_to_keep_epr.append(q1)
        frame_to_send_epr.append(q2)
        host.add_epr(receiver, qubit=q1, q_id=q1.id)

        "Sending other half of EPR to receiver"
        q2.send_to(receiver)
        # time.sleep(DISCRETE_TIME_STEP)
        epr_counter += 1
    print('A total of {} EPR frame has been sent \n\n'.format(epr_counter))
    return host_to_keep_epr, frame_to_send_epr


def rec_epr_frames(host: Host, sender: Host, epr_frame: list):
    for qubit in epr_frame:
        # print('Receiving EPR frame no: {} \n'.format(i))
        host.add_epr(host_id=sender.host_id, qubit=qubit, q_id=qubit.id)
        # time.sleep(DISCRETE_TIME_STEP)


def send_dense(host: Host, receiver: Host, msg: str, host_buffer: list):
    """
    host_buffer(list): It is a list of lists, where each element is a list
                       storing the EPR halves shared within a single EPR Frame.
    """
    # dense_frame = []  # List to store the dense messages.
    # print('The two parties already share an EPR')
    # print('Dense protocol can begin now ... \n')
    other_half = host_buffer.pop(0)
    # other_half_frame = host_buffer.pop(0)
    retrieved_epr = host.get_epr(receiver.host_id, q_id=other_half.id)
    # retrieved_epr = host.get_epr(receiver.host_id, q_id=other_half_frame.pop(0).id)
    q_encode = dens_encode(retrieved_epr, msg)
    # dense_frame.append(q_encode)
    q_encode.send_to(receiver_id=receiver.host_id)
    return q_encode


def dens_encode(q: Qubit, bits: str):
    """From Simons code"""
    if bits == "00":
        q.I()
    elif bits == "10":
        q.Z()
    elif bits == "01":
        q.X()
    elif bits == "11":
        q.X()
        q.Z()
    else:
        raise Exception('Bad input')  # Later add qubit erasure.
    return q


# FROM SIMONS gewiJN
def dense_decode(stored_epr_half: Qubit, received_qubit: Qubit):
    received_qubit.cnot(stored_epr_half)
    received_qubit.H()
    meas = [None, None]
    meas[0] = received_qubit.measure()
    meas[1] = stored_epr_half.measure()
    return str(meas[0]) + str(meas[1])


def str_to_byte(padded):
    byte_array = padded.encode('utf-8', errors='replace')
    binary_int = int.from_bytes(byte_array, "big")
    binary_string = bin(binary_int)

    without_b = binary_string[2:]
    return without_b


def byte_to_str(without_b):
    binary_int = int(without_b, 2)
    byte_number = binary_int.bit_length() + 7 // 8

    binary_array = binary_int.to_bytes(byte_number, "big")
    # ascii_text = binary_array.decode(errors='replace')
    padded_char = binary_array
    return padded_char


def binaryTupleFromInteger(i):
    return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])


def integerFromBinaryTuple(a, b):
    return a * 2 ** 1 + b * 2 ** 0


def send_pixels_dense(sender: Host, receiver: Host, host_buffer: list):
    global c1, c2, total_message, flag_sent_message_odd
    flag_sent_message_odd = 0

    for row in range(1):
        for column in range(coloumns):
            color = pixels[column, row + row_counter, :]
            hashed = hash(tuple(color))
            index = indices[hashed]
            b1, b2 = binaryTupleFromInteger(index)
            message = str(b1) + str(b2)
            frame.append(message)

    'RS - Encoding portion -----------------------------------------------------------------------------------'

    concat_message = ''.join(frame)
    rs_message = encode(coder, concat_message)
    # print(rs_message)
    # parity_bits = rs_message[K:]
    print('SENDER: RS encoded message:', rs_message)
    'Done for sending via dense'
    bytes_as_bits = ''.join(format(byte, '08b') for byte in rs_message)
    print('SENDER: Entire message as bits:', bytes_as_bits)

    total_message = bytes_as_bits  # Variable quantity
    pair_to_send = []
    print('SENDER: prepared total message : ', total_message)
    print('SENDER: length of the prepared message: ', len(total_message))

    'Pairing up to send as superdense'
    if len(total_message) % 2 != 0:
        total_message = total_message + '0'
        flag_sent_message_odd += 1

    message_frame = []
    for i in range(int(len(total_message) / 2)):
        pair_to_send.append(total_message[2 * i: 2 * i + 2])
    counter = 0
    for p in pair_to_send:
        # print('Sending dense-ly: {}'.format(counter))
        q = send_dense(sender, receiver, p, host_buffer)
        message_frame.append(q)
        # time.sleep(DISCRETE_TIME_STEP)
        counter += 1
    frame.clear()
    print("SENDER: Frame has been sent! \n\n")
    return message_frame


received_indexes = []


def rec_pixels_dense(sender: Host, receiver: Host, frame: list, host_buffer: list):
    global flag_sent_message_odd

    print("RECEIVER:  receiving data frame -------------------")
    messages = []
    for rec_half in frame:
        # half = host_buffer.pop(0)
        half = rec_half
        # half_frame = host_buffer.pop(0)
        ret_half = receiver.get_epr(host_id=sender.host_id, q_id=half.id)
        # ret_half = receiver.get_epr(host_id=sender.host_id, q_id=half_frame.pop(0).id)

        message = dense_decode(stored_epr_half=ret_half, received_qubit=half)
        messages.append(message)

    'RS - Decoding portion -----------------------------------------------------------------------------------'

    concat_rec_message = ''.join(messages)  # Concatenating to form an entire string
    if flag_sent_message_odd == 1:
        concat_rec_message = concat_rec_message[:-1]
    else:
        pass
    # rec_parity = concat_rec_message[K:]  # Collecting the parity for conversion as per unireedsolomon
    rec_parity = byte_to_str(concat_rec_message)
    print('RECEIVER: received total message:', concat_rec_message)
    print('RECEIVER: length of the message:', len(concat_rec_message))
    rec_real_parity = bytearray(rec_parity)

    'rec_real_parity is a byte array. Needs to convert it back to bitstrings'
    rec_final_message = rec_real_parity

    decoded_message = decode(coder, rec_final_message)
    # print('decoded message:', decoded_message)
    '-----------------------------------------------------------------------'
    new_message = []
    count = 0
    binary = '01'
    choices = ['0', '1']
    for i in decoded_message:
        if i not in binary:
            val = random.choices(choices)
            i = val[0]
            new_message.append(i)
        else:
            new_message.append(i)
    decoded_message = ''.join(new_message)
    '-----------------------------------------------------------------------'
    pairwise_indexes = []
    for g in range(int(len(decoded_message) / 2)):
        pairwise_indexes.append(decoded_message[2 * g: 2 * g + 2])
    # print('pairwise index:', pairwise_indexes)

    for zz in range(len(pairwise_indexes)):
        c1, c2 = tuple(map(int, pairwise_indexes[zz]))
        rec_index = integerFromBinaryTuple(c1, c2)
        received_indexes.append(rec_index)
    # print(received_indexes)

    for row in range(1):
        for column in range(coloumns):
            received_hash = hashes[received_indexes[0]]
            received_color = palette[received_hash]
            received_mario[column, row + row_counter, :] = received_color
            received_indexes.pop(0)
    print('RECEIVER: The whole frame has been received :)')
    print('RECEIVER: message:', concat_rec_message)
    print('\n')


def main():
    alice_buffer = []
    bob_buffer = []

    VERBOSE = True

    # backend = QuTipBackend()

    network = Network.get_instance()
    network.delay = 0
    nodes = ["Alice", "Bob"]
    network.start(nodes)

    host_alice = Host('Alice')
    host_alice.add_connection('Bob')
    host_alice.start()

    host_bob = Host('Bob')
    host_bob.add_connection('Alice')
    host_bob.start()

    network.add_host(host_alice)
    network.add_host(host_bob)
    t1 = time.time()
    while True:
        global row_counter
        if row_counter == 14:
            break
        print('Alice sending EPR pairs before Dense coding \n')
        alice_half, sent_frame = send_epr_frames(host_alice, host_bob.host_id)
        alice_buffer = alice_half
        print('Bob receiving sent-EPR frames from Alice \n')
        rec_epr_frames(host_bob, host_alice, epr_frame=sent_frame)
        bob_buffer = sent_frame
        if host_alice.shares_epr(host_bob.host_id):
            print('There is EPR shared among communicating nodes \n')
            pixel_frame = send_pixels_dense(host_alice, host_bob, host_buffer=alice_buffer)
            channel_error = []
            '---- Applying depolarizing error ----'

            # for a in range(len(pixel_frame)):
            #     erroneous_qubit = depolarizing_channel.depolarizing_channel(pixel_frame[a], 0.20)    # d should be
            #     # below 0.75
            #     channel_error.append(erroneous_qubit)
            # '-------------------------------------'

            rec_pixels_dense(host_alice, host_bob, pixel_frame, host_buffer=bob_buffer)
        # send_qubit(host_alice, host_bob)
        # rec_qubit(host_alice, host_bob)
        print('Pixel frame no. {} has been sent. Sending the next frame {}'.format(row_counter, row_counter + 1))
        row_counter += 1
    no = len(host_alice.get_epr_pairs(host_bob.host_id))
    print('No. of EPR pairs: ', no)
    t2 = time.time()

    received_im = Image.fromarray(received_mario)
    received_im.save('mario_link_error_nointer.bmp')

    print('total time elapsed: ', t2 - t1)

    host_alice.stop()
    host_bob.stop()
    network.stop()


if __name__ == '__main__':
    main()
