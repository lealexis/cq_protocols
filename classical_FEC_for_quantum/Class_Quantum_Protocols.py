import time

from qunetsim.components import Host
from qunetsim.components import Network
from qunetsim.objects import Logger, Qubit


class quantum_protocols:

    def gen_epr_pair(self, host: Host):
        halve_1st = Qubit(host)
        halve_2nd = Qubit(host, q_id=halve_1st.id)
        halve_1st.H()
        halve_1st.cnot(halve_2nd)
        return halve_1st, halve_2nd

    def send_epr_frames(self, host: Host, receiver: str, epr_frame_length: int):
        host_to_keep_epr = []
        frame_to_send_epr = []
        epr_counter = 0
        global total_message
        size = int(len(total_message) / 2)
        print('---- Generating EPR frames before starting communication ---- \n')
        print('Sending EPR frames ... ')
        for i in range(epr_frame_length):
            q1, q2 = self.gen_epr_pair(host=host)
            host_to_keep_epr.append(q1)
            frame_to_send_epr.append(q2)
            host.add_epr(receiver, qubit=q1, q_id=q1.id)
            "Sending other half of EPR to receiver"
            q2.send_to(receiver)
            # time.sleep(DISCRETE_TIME_STEP)
            epr_counter += 1
        print('EPR Generator: A total of {} EPR pairs has been sent in the frame. \n\n'.format(epr_counter))
        return host_to_keep_epr, frame_to_send_epr

    def rec_epr_frames(self, host: Host, sender: Host, epr_frame: list):
        for qubit in epr_frame:
            host.add_epr(host_id=sender.host_id, qubit=qubit, q_id=qubit.id)

    def revamped_EPR(self, sender: Host, receiver: Host, amount=int):
        id_pair = []
        for _ in range(amount):
            q1 = Qubit(sender)
            q2 = Qubit(receiver)

            q1.H()
            q1.cnot(q2)

            sender.add_epr(receiver.host_id, qubit=q1, q_id=q1.id)
            receiver.add_epr(sender.host_id, qubit=q2, q_id=q1.id)
            id_pair = [q1.id, q2.id]
        time.sleep(5)
        return id_pair

    def send_dense(self, host: Host, receiver: Host, msg: str, host_buffer: list):

        other_half = host_buffer.pop(0)
        retrieved_epr = host.get_epr(receiver.host_id, q_id=other_half.host_id)
        q_encode = self.dens_encode(retrieved_epr, msg)
        q_encode.send_to(receiver_id=receiver.host_id)
        return q_encode

    def dens_encode(self, q: Qubit, bits: str):
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
    def dense_decode(self, stored_epr_half: Qubit, received_qubit: Qubit):
        received_qubit.cnot(stored_epr_half)
        received_qubit.H()
        meas = [None, None]
        meas[0] = received_qubit.measure()
        meas[1] = stored_epr_half.measure()
        return str(meas[0]) + str(meas[1])
