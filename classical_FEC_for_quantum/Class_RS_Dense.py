import random
import numpy as np
import unireedsolomon as rs

flag_sent_message_odd = 0


class RS_Dense_Coding:
    def __init__(self, N=60, K=40, message=''):
        self.N = N
        self.K = K
        self.parity = int(N - K)
        self.coder = rs.RSCoder(N, K)
        self.type = ''
        self.frame_size = int
        self.message = message

    def encode(self, coder, msg):
        """Private function only used by the encoding function"""
        """Convert the msg to UTF-8-encoded bytes and encode with "coder".  Return as bytes."""
        return coder.encode(msg.encode('utf8')).encode('latin1')

    def decode(self, coder, encoded):
        """Private function only used by the encoding function"""
        """Decode the encoded message with "coder", convert result to bytes and decode UTF-8."""
        return coder.decode(encoded)[0].encode('latin1').decode('latin1')

    def str_to_byte(self, padded):
        """Private function only used by the encoding function"""

        byte_array = padded.encode('utf-8', errors='replace')
        binary_int = int.from_bytes(byte_array, "big")
        binary_string = bin(binary_int)

        without_b = binary_string[2:]
        return without_b

    def byte_to_str(self, without_b):
        """Private function only used by the decoding function"""

        binary_int = int(without_b, 2)
        byte_number = binary_int.bit_length() + 7 // 8

        binary_array = binary_int.to_bytes(byte_number, "big")
        # ascii_text = binary_array.decode(errors='replace')
        padded_char = binary_array
        return padded_char

    def binaryTupleFromInteger(self, i):
        """Private function only used by the encoding function"""

        return tuple([int(j) for j in list(bin(i)[2:].zfill(2))])

    def integerFromBinaryTuple(self, a, b):
        """Private function only used by the encoding function"""

        return a * 2 ** 1 + b * 2 ** 0

    def encoded_message_frame(self):
        """The encoding function returns two values: 1. The total message as a concatenation of the entire frame. 2."""
        'Paired up values of SDC protocol '
        global flag_sent_message_odd
        message = self.message
        if message is None or message == '':
            random_input_vector = np.random.randint(2, size=self.K)
            message = random_input_vector.tolist()

        string_rand_vec = [str(integer) for integer in message]

        concat_message = ''.join(string_rand_vec)
        rs_message = self.encode(self.coder, concat_message)

        'RS encoder encodes byte-wise and the characters appended as parity are utf-8 based.'

        bytes_as_bits = ''.join(format(byte, '08b') for byte in rs_message)

        total_message = bytes_as_bits  # Variable quantity

        self.frame_size = len(total_message)

        pair_to_send = []

        'Pairing up to send as superdense'
        if len(total_message) % 2 != 0:
            total_message = total_message + '0'
            flag_sent_message_odd = 1

        message_frame = []

        for i in range(int(len(total_message) / 2)):
            pair_to_send.append(total_message[2 * i: 2 * i + 2])
        # print('sent pair:', pair_to_send)
        return total_message, pair_to_send

    def decode_message_frame(self):
        global flag_sent_message_odd
        frame = self.encoded_message_frame()

        'RS - Decoding portion -----------------------------------------------------------------------------------'

        concat_rec_message = ''.join(frame)  # Concatenating to form an entire string
        if flag_sent_message_odd == 1:
            concat_rec_message = concat_rec_message[:-1]
        else:
            pass
        # rec_parity = concat_rec_message[K:]  # Collecting the parity for conversion as per unireedsolomon
        rec_parity = self.byte_to_str(concat_rec_message)
        # print('RECEIVER: received total message:', concat_rec_message)
        rec_real_parity = bytearray(rec_parity)

        decoded_message = self.decode(self.coder, rec_real_parity)
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

        return decoded_message, pairwise_indexes

