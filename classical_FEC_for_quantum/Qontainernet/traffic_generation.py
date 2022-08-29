"""
Artificial simple traffic generation script
Script input variables:
- packet size
- time between packets
"""
from scapy.all import *
import random
import time
import sys
import unireedsolomon as uni
from PIL import Image
import numpy as np

"-----------------------Mario-----------------------------------"


def binaryTupleFromInteger(i):
    return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])


def integerFromBinaryTuple(a, b):
    return a * 2 ** 1 + b * 2 ** 0


im = Image.open("received_mario_sprite.bmp")
pixels = np.array(im)
im.close()

coloumns, rows, colors = pixels.shape
dtype = pixels.dtype

'Size of rows: 14 ; Size of coloumns: 20'

hashes = []
hashes_ = []

palette = {}
indices = {}

frame = []
binary_pair = []

frame_index = 0

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

"-----------------------Mario-----------------------------------"

DESTINATION = str(sys.argv[1])
ROUNDS = int(sys.argv[2])
EPR_NUM = int(sys.argv[3])
PACKET_NUM = int(sys.argv[4])

PACKET_SIZE = 0
if len(sys.argv) > 5:
    PACKET_SIZE = int(sys.argv[5])

TYPE = "periodic"
if len(sys.argv) > 6:
    TYPE = sys.argv[6]

PROBABILITY = 0.5
if len(sys.argv) > 7:
    PROBABILITY = float(sys.argv[7])


def generate_packet(packet_size):
    """------------------------------------Mario transmission -----------------------------------------------"""
    global frame_index
    for row in range(rows):
        for column in range(coloumns):
            color = pixels[column, row, :]
            hashed = hash(tuple(color))
            index = indices[hashed]
            b1, b2 = binaryTupleFromInteger(index)
            message = str(b1) + str(b2)
            binary_pair.append(message)

    message_string = ''.join(binary_pair)

    bytegroup = [''.join(binary_pair[x:x + 4]) for x in range(0, len(binary_pair), 4)]
    frame = [bytegroup[f:f + 7] for f in range(0, len(bytegroup), 7)]
    byte_list = [hex(int(x, 2)) for x in frame[frame_index]]
    result_bytes = bytes([int(x, 0) for x in byte_list])

    # print('mario byte:', result_bytes)
    """------------------------------------Mario transmission -----------------------------------------------"""

    packet = IP(dst=DESTINATION, src="11.0.0.1")
    packet_size = packet_size - 48
    rs = uni.rs.RSCoder(packet_size, 7)  # variable quantity: always check before running simulation.
    # load = bytearray([1] * 7)
    load = bytearray(result_bytes)

    encoded = rs.encode(load).encode('latin1')
    print(encoded)
    print(list(encoded))
    if packet_size < 0:
        packet_size = 0
    load = bytearray(encoded)
    packet = packet / Raw(load=load)
    frame_index += 1
    if frame_index == 9:
        frame_index = 0
    return packet


def generate_epr_packet():
    # \x19 is first unused option
    packet = IP(dst=DESTINATION, options='\x19')

    return packet


def generate_traffic_periodic():
    if ROUNDS == -1:
        while True:
            generate_round()
    else:
        for r in range(ROUNDS):
            generate_round()


def generate_traffic_random():
    if ROUNDS == -1:
        while True:
            generate_random_round()
    else:
        for r in range(ROUNDS):
            generate_random_round()


def generate_random_round():
    packet = None
    if random.choices([True, False], [PROBABILITY, 1 - PROBABILITY])[0]:
        packet = generate_packet(PACKET_SIZE)
    else:
        packet = generate_epr_packet()
    send(packet)
    time.sleep(2)


def generate_round():
    for i in range(EPR_NUM):
        packet = generate_epr_packet()
        send(packet)
        time.sleep(1)
    for i in range(PACKET_NUM):
        packet = generate_packet(PACKET_SIZE)
        send(packet)
        time.sleep(1)


if __name__ == "__main__":
    if TYPE == "periodic":
        generate_traffic_periodic()
    else:
        generate_traffic_random()
