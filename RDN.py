"""
RDN Class

Dedicated class for Random Dynamical Nudge (RDN) methods.
Each method computes a nudge value from a set of opinions.

Usage:
    from RDN import RDN
    
    rdn = RDN(method=1, D=1.0, n=30, topic=0)
    nudge = rdn.calculate(opinions, global_mean)
"""

import numpy as np


class RDN:
    """
    Random Dynamical Nudge (RDN) calculator.
    
    Supports multiple RDN methods selectable by `method`.
    Implemented methods:
        method = 0: no RDN (returns 0)
        method = 1: simple sample mean (paper formulation)

    Other method numbers raise NotImplementedError so unsupported variants
    cannot silently run as if they were implemented.
    """

    def __init__(self, 
                 method: int = 1,
                 D: float = 1.0,
                 n: int = 30,
                 topic: int = 0):
        """
        Parameters:
            method: RDN method (0=off, 1=simple, ...)
            D: RDN strength
            n: sample size
            topic: 0=topic1, 1=topic2 (for simple RDN)
        """
        self.method = method
        self.D = np.float32(D)
        self.n = n
        self.topic = topic

    # ============================================================
    # PUBLIC API
    # ============================================================

    def calculate(self, opinions: np.ndarray, global_mean: float) -> np.ndarray:
        """
        Calculate the RDN value for the given opinions.
        
        Parameters:
            opinions: (N, 2) array of current opinions
            global_mean: scalar — global mean for the chosen topic
        
        Returns:
            nudge: array shaped like opinions (broadcastable) or scalar
        """
        match self.method:
            case 0:
                return 0.0
            case 1:
                return self._simple(opinions, global_mean)
            case _:
                raise NotImplementedError(
                    f"RDN method {self.method} is not implemented; supported methods are 0 and 1."
                )

    # ============================================================
    # METHOD 1: SIMPLE SAMPLE MEAN (paper formulation)
    # ============================================================

    def _simple(self, opinions: np.ndarray, global_mean: float) -> float:
        """
        R = sqrt(n) * (sample_mean - global_mean)
        """
        N = len(opinions)
        sample_indices = np.random.choice(N, size=self.n, replace=False)
        sample_opinions = opinions[sample_indices, self.topic]
        sample_mean = np.mean(sample_opinions)
        return np.sqrt(self.n) * (sample_mean - global_mean)

    # ============================================================
    # SETTERS (for mid-run changes)
    # ============================================================

    def set_method(self, method: int):
        self.method = method

    def set_D(self, D: float):
        self.D = np.float32(D)

    def set_n(self, n: int):
        self.n = n

    def set_topic(self, topic: int):
        self.topic = topic

    # ============================================================
    # REPR
    # ============================================================

    def __repr__(self):
        return (f"RDN(method={self.method}, D={self.D}, "
                f"n={self.n}, topic={self.topic})")