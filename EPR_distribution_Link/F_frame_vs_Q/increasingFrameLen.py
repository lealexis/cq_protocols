from threading import Event
import numpy as np
from math import trunc


class frLenIncr(object):
    def __init__(self, max_q=0.9, min_q=0.05, frames_per_len=100, eff_load=None):
        self.max_q = max_q
        self.min_q = min_q
        self.m_frames = frames_per_len

        if eff_load is None:
            self._eff_load = 40
        else:
            self._eff_load = eff_load

        min_len = trunc(self._eff_load / (1 - self.min_q))

        max_len = trunc(self._eff_load / (1 - self.max_q))

        if min_len <= self._eff_load:
            min_len = self._eff_load + 3
        while (min_len - self._eff_load) % 3 != 0:
            min_len += 1
        while (max_len - self._eff_load) % 3 != 0:
            max_len += 1

        self.min_len = min_len
        self.max_len = max_len
        self.lengths = np.arange(min_len, max_len+1, 3)

        self._frame_count = 0
        self._len_id = 0

        self.last_frame = Event()

    @property
    def eff_load(self):
        return self._eff_load
    @property
    def frame_count(self):
        return self._frame_count
    @frame_count.setter
    def frame_count(self, new_count):
        self._frame_count = new_count

    def _increase_fr_count(self):
        self.frame_count = self.frame_count + 1
        if (self.frame_count == self.m_frames) & (self._len_id == (len(self.lengths) - 1)):
            self.last_frame.set()

    def _reset_fr_count(self):
        self._frame_count = 0

    def _increase_len_id(self):
        self._len_id = self._len_id + 1

    def get_length(self):
        self._increase_fr_count()
        count = self.frame_count
        fr_ln = self.lengths[self._len_id]
        if (self.frame_count == self.m_frames) & (self._len_id != (len(self.lengths) - 1)):
            self._reset_fr_count()
            self._increase_len_id()
        return fr_ln, count


