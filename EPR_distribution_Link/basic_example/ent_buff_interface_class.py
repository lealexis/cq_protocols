from qunetsim.objects import Qubit
from qunetsim.components import Host
from QuTils import EPR_Pair_fidelity
import numpy as np
import pandas as pd 
import time

class HostNotStartedException(Exception):
    pass

class HostsNotConnectedException(Exception):
    pass

class BadInputException(Exception):
    pass

class FidelitySettingException(Exception):
    pass

class InterfaceIsInProcessException(Exception):
    pass

class BadPhaseCallException(Exception):
    pass

class FrameCompletelyStored(Exception):
    pass

class EPR_buff_itfc(object):
    
    def __init__(self, host: Host, partner_host_id=None, n_exp=None, 
                 eff_load=None, is_receiver=None, use_max_Fest=None):

        if host._queue_processor_thread is None:
            raise HostNotStartedException("host must be started before"
                                          " initializing its EPR buffer "
                                          "interface.")
        if (partner_host_id is None) or (type(partner_host_id) is not str):
            raise ValueError("partner_host_id must be a string.")

        if not (partner_host_id in host.quantum_connections):
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
        self.is_full = False
        self.f_IDs = set(np.linspace(0, (self.n-1), self.n, dtype=int))
        self.u_IDs = []
        self.ID_in_process = ([],[])
        self.in_process = False
        self._start_time = None
        self.history = False
        if is_receiver is None:
            self.is_receiver=False
        else:
            self.is_receiver= is_receiver
        self.started = False
        if use_max_Fest is None:
            # use the uID with highest fidelity first
            self.use_max_Fest =  True
        else:
            # use the oldest uID first
            self.use_max_Fest = use_max_Fest

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        if self.history is False:
            self._start_history()
            self.history = True
        if not self.started:
            self.started = True
        self._start_time = start_time

    def _start_history(self):
        if self.is_receiver:
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "t_end": pd.Series(dtype="float"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "t_est": pd.Series(dtype="float")})
        else:
            self.EPR_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                                   "t_init": pd.Series(dtype="float"),
                                                   "t_end": pd.Series(dtype="float"),
                                                   "F_est_b_xy": pd.Series(dtype="float"),
                                                   "F_est": pd.Series(dtype="float"),
                                                   "t_est": pd.Series(dtype="float")})

        self.SDC_frame_history = pd.DataFrame({"ID": pd.Series(dtype="int"),
                                               "t_init": pd.Series(dtype="float"),
                                               "t_end": pd.Series(dtype="float"),
                                               "F_est": pd.Series(dtype="float")})

        self.In_halves_history = pd.DataFrame({"t_in": pd.Series(dtype="float"),
                                               "Fid_in": pd.Series(dtype="float")})
    
    def _start(self):
        self._start_history()
        self.started = True
        self._start_time = time.perf_counter()

    def start(self):
        self._start()

    def _get_tuple_from_fr_ID(self, id_f=None):
        
        if id_f is None:
            raise BadInputException("A valid id for the frame must be passed.")
        try:
            frame_info = self.buffer_info[id_f]
        except KeyError:
            print("The id {k} is not a valid frame id.".format(k=id_f))
        else:
            return frame_info
    
    def _get_F_est_from_fr_ID(self, id_f=None):
        
        fr_info = self._get_tuple_from_fr_ID(id_f=id_f)
        return fr_info[0]["f_est"]

    def _get_qubit_ids_from_fr_ID(self, id_f=None):
        
        fr_info =  self._get_tuple_from_fr_ID(id_f=id_f)
        return fr_info[1]["qubit_ids"]

    def _clear_frame_info(self, frame_id=None):

        frame_info = self._get_tuple_from_fr_ID(id_f=frame_id)
        frame_info[0]["f_est"].clear()
        frame_info[1]["qubit_ids"].clear()

    def _append_id_to_ID_ip(self, id_f=None):
        if id_f is None:
            raise ValueError("pass a valid id_f")
        else:
            self.ID_in_process[0].append(id_f)

    def _set_mssg_to_ID_ip(self, mssg=None):
        if self.in_process:
            if len(self.ID_in_process[1]) == 1:
                self.ID_in_process[1].clear()
                self.ID_in_process[1].append(mssg)
            else:
                self.ID_in_process[1].clear()
                for m in mssg:
                    self.ID_in_process[1].append(m)
        else:   
            self.ID_in_process[1].append(mssg)

    def _get_ID_in_process(self):
        ID_ip = None
        if len(self.ID_in_process[0]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.ID_in_process[0]) == 1:
            ID_ip = self.ID_in_process[0][0]
        else:
            ID_ip = self.ID_in_process[0].copy()
        return ID_ip

    def _get_MSSG_in_process(self):
        MSSG_ip = None
        if len(self.ID_in_process[1]) == 0:
            raise ValueError("No frame ID is being processed.")
        elif len(self.ID_in_process[1]) == 1:
            MSSG_ip = self.ID_in_process[1][0]
        else:
            MSSG_ip = self.ID_in_process[1].copy()
        return MSSG_ip

    def _reset_is_IN_PROCESS(self):
        if self.in_process:
            self.in_process = False
            self.ID_in_process[0].clear()
            self.ID_in_process[1].clear()
        else:
            raise ValueError("No Frame is being processed. Imposible to reset"
                             " in_process.")

    def _actualize_ITFC_after_pop_ID(self, id_f):
        self._clear_frame_info(frame_id=id_f)
        if len(self.f_IDs)==0:
            self.is_full = False
        self.f_IDs.add(id_f)
        if len(self.f_IDs)==self.n:
            self.is_empty = True

    def _actualize_histories(self, df_to_add, kind:str):

        if kind == "epr":
            if self.EPR_frame_history.size == 0:
                self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
            else:
                if self.EPR_frame_history.iloc[-1].isnull().values.any():
                    boolcols = self.EPR_frame_history.columns.isin(df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.EPR_frame_history.iloc[-1, boolcols] = vals
                else:
                    self.EPR_frame_history = pd.concat([self.EPR_frame_history, df_to_add])
        elif kind == "sdc":
            if self.SDC_frame_history.size == 0:
                self.SDC_frame_history = pd.concat([self.SDC_frame_history, df_to_add])
            else:
                if self.SDC_frame_history.iloc[-1].isnull().values.any():
                    boolcols = self.SDC_frame_history.columns.isin(df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.SDC_frame_history.iloc[-1, boolcols] = vals
                else:
                    self.SDC_frame_history = pd.concat([self.SDC_frame_history, df_to_add])
        else:
            if self.In_halves_history.size == 0:
                self.In_halves_history = pd.concat([self.In_halves_history, df_to_add])
            else:
                if self.In_halves_history.iloc[-1].isnull().values.any():
                    boolcols = self.In_halves_history.columns.isin(df_to_add.columns.values)
                    vals = df_to_add.values
                    vals = vals.reshape((vals.shape[1],))
                    self.In_halves_history.iloc[-1, boolcols] = vals
                else:
                    self.In_halves_history = pd.concat([self.In_halves_history, df_to_add])

    def _get_nxt_uID(self):
        nxt_ID = None
        if self.use_max_Fest:
            nxt_ID = self._get_uID_with_max_Fest()
        else:
            nxt_ID = self._get_oldest_uID()
        return nxt_ID

    def _get_oldest_uID(self):
        return self.u_IDs.pop(0)
    
    def _get_uID_with_max_Fest(self):
        F_mx_id = None
        f_max = 0 
        for fr_id in self.u_IDs:
            f_est = self._get_F_est_from_fr_ID(id_f=fr_id)
            if f_est[0] > f_max:
                f_max = f_est[0]
                F_mx_id = fr_id
        self.u_IDs.remove(F_mx_id)
        return F_mx_id

    # ***************************************************
    # ** Methods for Phases of EPR-Frame Communication **
    # ***************************************************

    # valid for sender and receiver
    def nxt_fID_EPR_START(self):
        if not self.in_process:
            free_list = self.f_IDs.copy()
            free_list = list(free_list)
            nxt_ID = free_list[0]
            actualize_df = pd.DataFrame([[nxt_ID]], columns=["ID"])
            self._actualize_histories(df_to_add=actualize_df, kind="epr")
            self.f_IDs.remove(nxt_ID)
            self._append_id_to_ID_ip(id_f=nxt_ID)
            self._set_mssg_to_ID_ip(mssg="EPR:started")
            self.in_process = True
            return nxt_ID
        else:
            raise InterfaceIsInProcessException("EPR_START cannot be called. "
                                                "Interface is already processing"
                                                " a frame.")
    # valid for sender and receiver
    def store_EPR_PHASE_1(self, epr_half: Qubit):
        if type(epr_half) is not Qubit:
            raise BadInputException("A valid epr_half must be passed.")
        mssg = self._get_MSSG_in_process()
        valid_mssg = ((mssg == "EPR:started") or (mssg == "EPR:storing")) 
        if self.in_process and valid_mssg: 
            qids = self._get_qubit_ids_from_fr_ID(id_f=self._get_ID_in_process())
            if len(qids) == self.eff_load:
                raise FrameCompletelyStored("EPR frame was already completely "
                                            "received.")
            ti = time.perf_counter() - self.start_time
            if len(qids)==0:
                self._set_mssg_to_ID_ip("EPR:storing")
                actualize_df = pd.DataFrame([[ti]], columns=["t_init"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
            fest =  EPR_Pair_fidelity(epr_half)
            actualize_df = pd.DataFrame([[ti, fest]], columns=["t_in", "Fid_in"])
            self._actualize_histories(df_to_add=actualize_df, kind="in")

            # store epr half
            q_id = self.host.add_epr(host_id=self.partner_host_id, qubit=epr_half)

            qids.append(q_id)
            if len(qids) == self.eff_load:
                self._set_mssg_to_ID_ip("EPR:wait_set_F")
                tf = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[tf]], columns=["t_end"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
        else:
            raise BadPhaseCallException("EPR_PHASE_1 can repeatedly be called "
                                        "only after EPR_START until the frame "
                                        "was completely received.")
    # valid for sender and receiver
    def set_F_est_EPR_END_PHASE(self, F_est=None, to_history=False):
        if self.in_process and self._get_MSSG_in_process() == "EPR:wait_set_F":
            if to_history:
                actualize_df = pd.DataFrame([[F_est]], columns=["F_est_b_xy"])
                self._actualize_histories(df_to_add=actualize_df, kind="epr")
            else:
                ip_ID = self._get_ID_in_process()
                f = self._get_F_est_from_fr_ID(id_f=ip_ID)
                if len(f)==0:
                    f.append(F_est)
                    fid_t = time.perf_counter() - self.start_time
                    actualize_df = pd.DataFrame([[F_est, fid_t]], columns=["F_est", "t_est"])
                    self._actualize_histories(df_to_add=actualize_df, kind="epr")
                    if len(self.u_IDs)==0:
                        self.is_empty = False
                    self.u_IDs.append(ip_ID)
                    if len(self.u_IDs)==self.n:
                        self.is_full = True
                    self._reset_is_IN_PROCESS()
                else:
                    raise FidelitySettingException("Frame {iD} has already a f_est.".format(iD=ip_ID))
        else:
            raise BadPhaseCallException("EPR_END_PHASE can be called only after"
                                        " EPR_PHASE_1.")
   
    # ***************************************************
    # ** Methods for Phases of SDC-Frame Communication **
    # ***************************************************
    
    # valid for sender and receiver
    def nxt_uID_SDC_START(self):
        if not self.in_process:
            nxt_ID = self._get_nxt_uID()
            actualize_df = pd.DataFrame([[nxt_ID]], columns=["ID"])
            self._actualize_histories(df_to_add=actualize_df, kind="sdc")
            self._append_id_to_ID_ip(id_f=nxt_ID)
            self._set_mssg_to_ID_ip(mssg="SDC:started")
            self.in_process = True
            return nxt_ID
        else:
            raise InterfaceIsInProcessException("SDC_START cannot be called. "
                                                "Interface is already processing"
                                                " a frame.")
    
    #  valid for sender and receiver 
    def pop_SDC_END_PHASE(self):
        if (self.in_process and ((self._get_MSSG_in_process()=="SDC:started")
                                    or (self._get_MSSG_in_process()=="SDC:epr-retrieving"))):
            ip_ID = self._get_ID_in_process()
            qids = self._get_qubit_ids_from_fr_ID(id_f=ip_ID)
            
            if len(qids)==self.eff_load:
                self._set_mssg_to_ID_ip(mssg="SDC:epr-retrieving")
                ti = time.perf_counter() - self.start_time
                actualize_df = pd.DataFrame([[ip_ID, ti]], columns=["ID","t_init"])
                self._actualize_histories(df_to_add=actualize_df, kind="sdc")
            qid = qids.pop(0)
            epr_half = self.host.get_epr(host_id=self.partner_host_id, q_id=qid)
            
            if len(qids) == 0:
                tf = time.perf_counter() - self.start_time
                f = self._get_F_est_from_fr_ID(id_f=self._get_ID_in_process())
                actualize_df = pd.DataFrame([[tf, f[0]]], columns=["t_end", "F_est"])
                self._actualize_histories(df_to_add=actualize_df, kind="sdc")
                self._actualize_ITFC_after_pop_ID(id_f=ip_ID)
                self._reset_is_IN_PROCESS()
            return epr_half
        else:
            raise BadPhaseCallException("SDC_PHASE_1 can be repeatedly called"
                                        " only after SDC_START until stored "
                                        "frame was consumed.")