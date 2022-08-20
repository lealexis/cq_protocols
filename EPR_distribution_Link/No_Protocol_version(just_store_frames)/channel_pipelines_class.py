from queue import Queue
from qunetsim.objects import Qubit
from threading import Event, Timer 

class NotQubitException(Exception):
    pass

class NotStringException(Exception):
    pass

class OutSocketFull(Exception):
    pass

class QuPipe(object):
    """Quantum pipeline object simulating the pipeline nature of a channel on 
    which qubits fly from sender to receiver a predefined *delay* time. The 
    passed delay time must be bigger than the inter-qubit time. 
    """
    def __init__(self, delay=None):
        self._delay = delay
        self._queue = Queue()
        self.Qframe_in_transmission = Event()
        self.pipe_load = 0
        self.out_socket = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        self._delay = delay

    def put(self, flying_qubit:Qubit):

        #if isinstance(flying_qubit,Qubit):
        if self.pipe_load == 0:
            self.Qframe_in_transmission.set()
        self.pipe_load += 1 
        t_get = Timer(self._delay, self.get)
        self._queue.put(flying_qubit)
        t_get.start()
            
        #raise NotQubitException("A non Qubit type object has attempted to enter"
        #                        " the quantum pipeline.")

    def get(self):
        if len(self.out_socket) == 0 :
            self.out_socket.append(self._queue.get())
            self.pipe_load -= 1 
            if self.pipe_load == 0:
                self.Qframe_in_transmission.clear()
        else:
            raise OutSocketFull("Qubit has not yet been taked from output socket.")


class ClassicPipe(object):
    """ Pipelined non-simultaneous duplex channel for transmission of the 
    classical feedbacks between the comunicating parties. Delay should be 
    bigger than the delay of the Quantum pipelined channel.
    """
    def __init__(self, delay):
        self._delay = delay
        self.fst_feedback_in_trans =  Event()
        self.snd_feedback_in_trans =  Event()
        self._queue = Queue()
        self.pipe_load = 0
        self.feedback_num = 0
        self.out_socket = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        self._delay = delay
    
    def put(self, cbit:str):
        # TODO: check which type is the classic message to be sent is
        if type(cbit) is str:
            if (self.pipe_load == 0) and (self.feedback_num == 0):
                self.fst_feedback_in_trans.set()
                self.feedback_num += 1
            if (self.pipe_load == 0) and (self.feedback_num == 1):
                self.snd_feedback_in_trans.set()
                self.feedback_num += 1
            self.pipe_load += 1
            t_get = Timer(self.delay, self.get)
            self._queue.put(cbit)
            t_get.start()
        else:
            raise NotStringException("Classic bit must be a string.")

    def get(self):
        if len(self.out_socket) == 0:
            self.out_socket.append(self._queue.get())
            self.pipe_load -= 1
            if (self.pipe_load == 0) and (self.feedback_num == 1):
                self.fst_feedback_in_trans.clear()
            if (self.pipe_load == 0) and (self.feedback_num == 2):
                self.snd_feedback_in_trans.clear()
                self.feedback_num = 0 
        else:
            raise OutSocketFull("C-bit has not yet been taked from output socket.")