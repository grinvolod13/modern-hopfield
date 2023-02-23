import numpy as np

class Hopfield():
    def __init__(self, X: np.ndarray) -> None:
        self.X = X
    
    def _step(self, E: np.ndarray, b=4):
        t = b * (self.X.T @ E)
        t = t - np.max(t)
        t = np.exp(t)
        E = self.X @ (t/np.sum(t))
        return E
    
    def run(self, E: np.ndarray, b=4):
        return self._step(E, b)
