from qunetsim.objects import Qubit
from qunetsim.components import Host
import numpy as np

class HostNotStartedException(Exception):
    pass

class HostsNotConnectedException(Exception):
    pass

class BadInputException(Exception):
    pass

class FidelitySettingException(Exception):
    pass

class EPR_buff_itfc(object):
    
    def __init__(self, host: Host, partner_host_id=None, n_exp=None, eff_load=None):

        if host._queue_processor_thread is None:
            raise HostNotStartedException("host must be started before"
                                          " initializing its EPR buffer "
                                          "interface.")
        if (partner_host_id is None) or (type(partner_host_id) is not str):
            raise ValueError("partner_host_id must be a string.")

        host_partner_conn = partner_host_id in host.quantum_connections
        
        if not (host_partner_conn):
            raise HostsNotConnectedException("host and partner_host must be"
                                             " connected.")

        self.host = host
        self.partner_host_id = partner_host_id
        
        if n_exp is None:
            self.n = 2**6
            self.n_exp = 6
        else:
            # TODO: check validity of argument frame_capacity 
            self.n = 2**n_exp
            self.n_exp = n_exp

        if eff_load is None:
            self.eff_load = 35
        else:
            self.eff_load = eff_load 
        
        self.buffer_info = {key:({"f_est":[]},{"qubit_ids":[]}) for key in  np.linspace(0, (self.n - 1), self.n, dtype=int)}
        self.is_empty = True
        self.has_space = True
        self.is_full = False
        self.free_frames_IDs = set(np.linspace(0, (self.n-1), self.n, dtype=int))
        self.used_frames_IDs = set()
        self.fr_ID_in_process = {}
        self.is_in_process = False 

    def _get_tuple_from_fr_ID(self, id=None):
        
        if id is None:
            raise BadInputException("A valid id for the frame must be passed.")
        try:
            frame_info = self.buffer_info[id]
        except KeyError:
            print("The id {k} is not a valid frame id.".format(k=id))
        else:
            return frame_info
    
    def _get_F_est_from_fr_ID(self, id=None):
        
        fr_info = self._get_tuple_from_fr_ID(id=id)
        return fr_info[0]["f_est"]

    def _get_qubit_ids_from_fr_ID(self, id=None):
        
        fr_info =  self._get_tuple_from_fr_ID(id=id)
        return fr_info[1]["qubit_ids"]

    def store_epr_half_in_fr_ID(self, epr_half: Qubit, frame_id=None):
        
        if type(epr_half) is not Qubit:
            raise BadInputException("A valid epr_half must be passed.")
        qids = self._get_qubit_ids_from_fr_ID(id=frame_id)
        
        q_id = self.host.add_epr(host_id=self.partner_host_id, qubit=epr_half)        
        if len(qids)==0:
            self.is_in_process = True
            self.fr_ID_in_process = {frame_id:"EPR-storing"}
        qids.append(q_id)

    def get_epr_half_from_Frame(self, qubit_id: str, frame_id=None):
        """get the EPR half *qubit_id*. This method is solely used for getting
        the EPR halves to be measured in the FEU(fidelity estimation unit). """

        if type(qubit_id) is not str:
            raise BadInputException("A valid qubit_id must be passed.")
        qids = self._get_qubit_ids_from_fr_ID(id=frame_id)
        try:
            idx = qids.index(qubit_id)
        except ValueError:
            print("the qubit id: {id} is corrupted or is not in frame {FrID}".format(id=qubit_id, FrID=frame_id))
        else:
            epr_half = self.host.get_epr(host_id=self.partner_host_id, q_id=qubit_id)
            del(qids[idx])
            if len(qids)==self.eff_load:
                self.is_in_process = True
                self.fr_ID_in_process = {frame_id:"EPR:FEU-validating"}
            return epr_half

    def pop_oldest_epr_half_from_Frame(self, frame_id=None):
        
        qids = self._get_qubit_ids_from_fr_ID(id=frame_id)
        if len(qids)==self.eff_load:
            self.is_in_process = True
            self.fr_ID_in_process = {frame_id:"SDC:epr-retrieving"}
        qid = qids[0]
        epr_half = self.host.get_epr(host_id=self.partner_host_id, q_id=qid)
        del(qids[0])
        if len(qids) == 0:
            self._clear_frame_info(frame_id=frame_id)
            if len(self.free_frames_IDs)==0:
                self.is_full = False
                self.has_space = True
            self.free_frames_IDs.add(frame_id)
            
            if len(self.free_frames_IDs)==self.n:
                self.is_empty = True
            self.fr_ID_in_process = {}
            self.is_in_process = False
        return epr_half
    
    def get_qids_in_Frame(self, frame_id=None):

        qids = self._get_qubit_ids_from_fr_ID(id=frame_id)
        self.is_in_process = True
        self.fr_ID_in_process = {frame_id:"FEU-measuring"}
        qubit_ids = qids.copy()
        return qids
    
    def _clear_frame_info(self, frame_id=None):

        frame_info = self._get_tuple_from_fr_ID(id=frame_id)
        frame_info[0]["f_est"].clear()
        frame_info[1]["qubit_ids"].clear()

    def drop_Frame(self, frame_id=None):
        
        qids = self._get_qubit_ids_from_fr_ID(id=frame_id)
        for qid in qids:
            self.host.drop_epr(self.partner_host_id, qid)
        self._clear_frame_info(frame_id=frame_id)
        if len(self.free_frames_IDs)==0:
            self.is_full = False
            self.has_space = True
        self.free_frames_IDs.add(frame_id)
        self.is_in_process = False
        self.fr_ID_in_process = {}
        if len(self.free_frames_IDs)==self.n:
            self.is_empty = True
        self.used_frames_IDs.discard(frame_id)  

    def set_F_est(self, F_est=None, frame_id=None):
        f = self._get_F_est_from_fr_ID(id=frame_id)
        if len(f)==0:
            f.append(F_est)
            self.is_in_process = False
            self.fr_ID_in_process = {}
            if len(self.used_frames_IDs)==0:
                self.is_empty = False
            self.used_frames_IDs.add(frame_id)
            if len(self.used_frames_IDs)==self.n:
                self.is_full = True
                self.has_space = False
        else:
            raise FidelitySettingException("Frame {iD} has already a f_est.".format(iD=frame_id))

    def get_Fest_and_IDs_by_Fthres(self, f_thres=None):

        if (f_thres is None) or (f_thres < 0) or (f_thres > 1):
            raise ValueError("pass a valid fidelity threshold.")
        
        above_frames = []
        bellow_frames = []
        for fr_id in self.used_frames_IDs:
            f_est = self._get_F_est_from_fr_ID(id=fr_id).copy()
            if len(f_est)==0:
                continue
            if f_est[0] < f_thres:
                bellow_frames.append((f_est[0], fr_id))
            else:
                above_frames.append((f_est[0], fr_id))
        return above_frames, bellow_frames 

    def get_next_free_frID(self):
        free_list = self.free_frames_IDs.copy()
        free_list = list(free_list)
        nxt_ID = free_list[0]
        
        self.free_frames_IDs.remove(nxt_ID)
        self.is_in_process = True
        self.fr_ID_in_process = {nxt_ID:"EPR-started"}
        return nxt_ID

    def get_next_used_frID(self):
        used_list = self.used_frames_IDs.copy()
        used_list = list(used_list)
        nxt_ID = used_list[0]
        
        self.used_frames_IDs.remove(nxt_ID)
        self.is_in_process = True
        self.fr_ID_in_process = {nxt_ID:"SDC-started"}
        return nxt_ID
