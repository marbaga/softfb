import numpy as np


class Bandit():

    def __init__(self,  *args):
        self.observation_spec = lambda : {"observations": np.zeros(2)}
        self.action_spec = lambda : np.zeros(2)

    def reset(self, **kwargs):
        self.x = np.zeros(2, dtype=np.float32)
        return self.x.copy(), {}

    def step(self, action):
        self.x = action.copy() if np.linalg.norm(self.x) < 0.1 else self.x
        return self.x.copy(), 0.0, False, False, {}

    def render(self):
        frame = np.zeros((9, 9, 3))
        idxs = ((self.x + 1) * 4.5).round().clip(0, 8)
        frame[int(idxs[0].item()), int(idxs[1].item()), 2] = 255
        return frame.astype(np.uint8)
