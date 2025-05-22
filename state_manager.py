import tempfile
import numpy as np
from ultralytics.engine import results
from constants import CONSTANTS
from utils import draw_prediction_squares, sort_result


class StateManager:
    def __init__(self):
        self.current = 0
        self.STATES = {}
        self.SORTED_STATES = set()

    def save_state(self, state: results.Results, img: np.ndarray):
        sorted_state = sort_result(state)
        if tuple(sorted_state.cls.tolist()) in self.SORTED_STATES:
            return False
        else:
            fp = tempfile.TemporaryFile()
            np.save(fp, img)
            self.SORTED_STATES.add(tuple(sorted_state.cls.tolist()))
            self.STATES[self.current] = (sorted_state, fp)
            self.current += 1
            return True

    def clear_states(self):
        self.STATES = {}
        self.SORTED_STATES = set()
        self.current = 0

    def draw_states(self, img):
        x, y = 0, 0
        for state, _ in self.STATES.values():
            draw_prediction_squares(img, state, start_x=x, start_y=y)
            x += CONSTANTS.face_size
            if x > img.shape[1]:
                x = 0
                y += CONSTANTS.face_size
                if y > img.shape[0]:
                    break
