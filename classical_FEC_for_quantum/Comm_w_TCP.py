import time
import sys
from qunetsim.components import Host, Network
from qunetsim.objects import Logger
import Class_RS_Dense
from Class_Quantum_Protocols import quantum_protocols
from qunetsim.objects import Qubit

thread_1_return = None
thread_2_return = None

SYN = '10'
SYN_ACK = '11'
ACK = '01'
WAIT_TIME = 30
MAX_NUM_OF_TRANSMISSIONS = 10
Logger.DISABLED = False
SENDER_BUFFER = []
RECEIVER_BUFFER = []
N = 40
K = 20
message = ''
qprotocol_obj: quantum_protocols = quantum_protocols()


def handshake_sender(host, receiver_id):
    """
    Establishes a classical TCP-like handshake with the receiver .
    If successful starts the transmission of qubits , otherwise terminated the connection.
    :param host: Sender of qubits
    :param receiver_id: ID of the receiver
    :return: If successful returns True, otherwise False
    """

    # Create an EPR pair.
    qa_1 = Qubit(host)
    qa_2 = Qubit(host)

    qa_1.H()
    qa_1.cnot(qa_2)

    # Send a half of EPR pair and the SYN message to Bob.
    _, ack_received = host.send_qubit(receiver_id, qa_2, await_ack=True)
    if ack_received is False:
        print('ACK is not received')
        return False
    ack_received = host.send_classical(receiver_id, SYN, await_ack=True)
    if ack_received is False:
        print('ACK is not received')
        return False

    syn_seq_num = host.get_sequence_number_receiver(receiver_id)

    # Receive the qubits Bob has sent (qubit 2 and qubit 3) for SYN-ACK.
    qb_2 = host.get_data_qubit(receiver_id, wait=WAIT_TIME)
    if qb_2 is None:
        return False

    qb_3 = host.get_data_qubit(receiver_id, wait=WAIT_TIME)
    if qb_3 is None:
        return False

    # Receive the classical message Bob has sent for SYN-ACK.
    message_recv = host.get_classical(receiver_id, syn_seq_num + 2, wait=WAIT_TIME)
    if message_recv is None:
        return False

    if message_recv.content == '11':
        print("SYN-ACK is received by Alice")
    else:
        print('Connection terminated - 1 ')
        return False

    # Make a Bell State measurement on qubit 1 and qubit 2.
    qa_1.cnot(qb_2)
    qa_1.H()
    qa_1_check = qa_1.measure()
    qb_2_check = qb_2.measure()

    # If measurement results are as expected, send Bob a ACK message and the qubit 3 that he has sent previously.
    # Else report that there is something wrong.
    if qa_1_check == 0 and qb_2_check == 0:
        ack_received = host.send_classical(receiver_id, ACK, await_ack=True)
        if ack_received is False:
            print('ACK is not received')
            return False
        _, ack_received = host.send_qubit(receiver_id, qb_3, await_ack=True)
        if ack_received is False:
            print('ACK is not received')
            return False
        return True
    else:
        print("Something is wrong.")
        return False


def handshake_receiver(host, sender_id):
    """
    Establishes a classical TCP-like handshake with the sender . From QuNetSim
    If successful starts to receive the qubits , otherwise terminated the connection.
    :param host: Receiver host
    :param sender_id: ID of the sender
    :return: If successful returns True, otherwise False
    """
    latest_seq_num = host.get_sequence_number_receiver(sender_id)

    # Receive the EPR half of Alice and the SYN message
    qb_2 = host.get_data_qubit(sender_id, wait=WAIT_TIME)
    if qb_2 is None:
        print('qb_2 is None')
        return False

    message_recv = host.get_classical(sender_id, (latest_seq_num + 1), wait=WAIT_TIME)
    if not message_recv:
        print('No message has arrived')
        return False

    message_recv = message_recv.content

    if message_recv == '10':
        print("SYN is received by Bob")
    else:
        return False

    # Create an EPR pair.
    qb_3 = Qubit(host)
    qb_4 = Qubit(host)
    qb_3.H()
    qb_3.cnot(qb_4)

    # Send half of the EPR pair created (qubit 3) and send back the qubit 2 that Alice has sent first.
    _, ack_received = host.send_qubit(sender_id, qb_2, await_ack=True)
    if ack_received is False:
        print('ACK is not received')
        return False

    _, ack_received = host.send_qubit(sender_id, qb_3, await_ack=True)
    if ack_received is False:
        print('ACK is not received')
        return False

    # Send SYN-ACK message.
    host.send_classical(sender_id, SYN_ACK, True)
    latest_seq_num = host.get_sequence_number_receiver(sender_id)

    # Receive the ACK message.
    message = host.get_classical(sender_id, latest_seq_num, wait=WAIT_TIME)
    if message is None:
        print('ACK was not received by Bob')
        return False

    if message.content == '01':
        print('ACK was received by Bob')

    # Receive the qubit 3.
    qa_3 = host.get_data_qubit(sender_id, wait=WAIT_TIME)
    if qa_3 is None:
        return False

    # Make a Bell State measurement in qubit 3 and qubit 4.
    qa_3.cnot(qb_4)
    qa_3.H()

    qa_3_check = qa_3.measure()
    qb_4_check = qb_4.measure()

    # If measurement results are as expected , establish the TCP connection. From QuNetSim
    # Else report that there is something wrong.
    if qa_3_check == 0 and qb_4_check == 0:
        print("TCP connection established.")
        return True
    else:
        print("Something is wrong.")
        return False


def qtcp_sender(host, q_size, receiver_id, checksum_size_per_qubit):
    """
    Establishes a handshake and sends the data qubits to the receiver if handshake is successful. From QuNetSim
    :param host: Sender of qubits
    :param q_size: Number of qubits to be sent
    :param receiver_id: ID of the receiver
    :param checksum_size_per_qubit: Checksum qubit per data qubit size
    :return:
    """
    global thread_1_return
    tcp_connection = handshake_sender(host, receiver_id)
    if not tcp_connection:
        print('Connection terminated.')
        thread_1_return = False
        return
    else:
        print('Sender acknowledges the connection ...')


def qtcp_receiver(host, q_size, sender_id, checksum_size_per_qubit):
    """
    Establishes a handshake and receives the data qubits from the sender if handshake is successful. From QuNetSim
    :param host: Receiver host
    :param q_size: Qubit size to be received
    :param sender_id: ID of the sender
    :param checksum_size_per_qubit: Checksum qubit per data qubit size
    :return:
    """
    tcp_connection = handshake_receiver(host, sender_id)
    global thread_2_return
    if not tcp_connection:
        print('Connection terminated.')
        thread_2_return = False
        return
    else:
        print('... Receiver acknowledges the connection')


def sender_protocol(sender: Host, receiver: Host, host_buffer: list):
    """Sender protocol to transmit frames as SDC or sequentially to compare the varying channel capacity."""
    print('SENDER: Sending Protocol-------------------------------------------------------------------------')
    frames_sent = 0
    frame_to_send = []

    epr_buffer = len(sender.get_epr_pairs(receiver.host_id))
    "RS Encoding step --------------------------------------------------------------------------------------"
    rs_object = Class_RS_Dense.RS_Dense_Coding(N, K, message)
    encoded_bits, pairwise_encoded = rs_object.encoded_message_frame()
    if sender.shares_epr(receiver.host_id):
        if epr_buffer > len(encoded_bits)/2:
            print('SENDER: Sending frame using SDC protocol. \n')
            for p in pairwise_encoded:
                q = qprotocol_obj.send_dense(sender, receiver, p, host_buffer)
                frame_to_send.append(q)
            frames_sent += 1
        print('SENDER: Frame(s) sent {}'.format(frames_sent))
        print('SENDER: Message contained in the frame: {}'.format(encoded_bits))
    else:
        print('SENDER: No EPR pairs')

def receiver_protocol(sender: Host, receiver: Host, host_buffer: list):
    """Receiver protocol to receive frames as SDC or sequentially."""
    print('RECEIVER: Receiving Protocol----------------------------------------------------------------------')


def main():
    global thread_1_return
    global thread_2_return

    network = Network.get_instance()
    nodes = ["Alice", "Bob"]
    # back = CQCBackend()
    network.start(nodes)
    network.delay = 0.0

    host_alice = Host('Alice')
    host_alice.add_connection('Bob')
    host_alice.max_ack_wait = 30
    host_alice.delay = 0.0
    host_alice.start()

    host_bob = Host('Bob')
    host_bob.max_ack_wait = 30
    host_bob.delay = 0.0
    host_bob.add_connection('Alice')
    host_bob.start()

    network.add_host(host_alice)
    network.add_host(host_bob)

    network.x_error_rate = 0
    network.packet_drop_rate = 0

    q_size = 6
    checksum_per_qubit = 2

    t1 = host_alice.run_protocol(qtcp_sender, (q_size, host_bob.host_id, checksum_per_qubit))
    t2 = host_bob.run_protocol(qtcp_receiver, (q_size, host_alice.host_id, checksum_per_qubit))

    t1.join()
    t2.join()

    # while thread_1_return is None or thread_2_return is None:
    #     if thread_1_return is False or thread_2_return is False:
    #         print('TCP Connection not successful : EXITING')
    #         sys.exit(1)
    #     pass

    # while thread_1_return is True and thread_2_return is True:
    #     print('--- Starting communication between Alice and Bob ---')
    #     host_alice.run_protocol()
    #     host_bob.run_protocol()

    # start_time = time.time()
    # while time.time() - start_time < 150:
    #     pass

    network.stop(stop_hosts=True)
    exit()


if __name__ == '__main__':
    main()