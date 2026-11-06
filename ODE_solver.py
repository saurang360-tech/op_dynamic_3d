"""
ODE Solver Module

Contains the social_network class with both Euler and RK4 integration,
plus optional RDN (Random Dynamical Nudge) support.

Usage:
    from ODE_solver import social_network
    
    sn = social_network(N=1000, RDN_select=0)
    sn.set_activity()
    sn.set_social_interactions()
    sn.run_network(dt=0.05, t_dur=10.0)
"""

import warnings

import numpy as np
from RDN import RDN

# Initial opinions have standard deviation 3.0; this floor is one float32
# precision unit at that scale and only regularizes numerically tiny distances.
_DISTANCE_EPSILON = np.float32(3.0 * np.finfo(np.float32).eps)


class social_network:
    """
    this class will create the agents and the socal network with 2 issues
    """

    def __init__(self, 
                 N: int = 1000,
                 m: int = 5,
                 K: float = 3.0,
                 alpha: float = 2.0,
                 alpha2: float = 2.0,
                 delta: float = 45,
                 beta: float = 2.0,
                 r: float = 0.5,
                 gamma: float = 2.1,
                 epsilon: float = 0.01,
                 RDN_select: int = 0,      # 0=off, 1=simple, 2=2D, 3=hybrid
                 RDN_topic: int = 0,       # 0=topic1, 1=topic2, 2=both
                 RDN_D: float = 0.5,       # RDN strength
                 RDN_n: int = 30           # RDN sample size
                ):           
        
        # Store all parameters
        self.N: int = N
        self.m: int = m
        self.K: np.float32 = np.float32(K)
        self.alpha: np.float32 = np.float32(alpha)
        self.alpha2: np.float32 = np.float32(alpha2)
        self.beta: np.float32 = np.float32(beta)
        self.r: np.float32 = np.float32(r)
        self.gamma: np.float32 = np.float32(gamma)
        self.epsilon: np.float32 = np.float32(epsilon)
        self.delta: np.float32 = np.float32(delta)
        
        # RDN configuration (stored for reference)
        self.RDN_select = RDN_select
        self.RDN_topic = RDN_topic
        self.RDN_D = np.float32(RDN_D)
        self.RDN_n = RDN_n
        
        # Create RDN instance
        self.rdn = RDN(
            method=RDN_select,
            D=RDN_D,
            n=RDN_n,
            topic=RDN_topic
        )

        self._opinions = np.random.normal(0, 3, (N, 2)).astype(np.float32)
        self.activities = np.zeros(N, dtype=np.float32)
        
        self.initalise_phi(self.delta)
        
        self.p_conn = None
        # Refresh cadence is measured in op_dynamics calls (RK4 stages count
        # separately), and the counter intentionally persists across runs.
        self._p_conn_dynamics_call_counter = 0
        self._p_conn_refresh_every_dynamics_calls = 10
        
        self._adj = np.zeros((N, N), dtype=np.int8)

    def initalise_phi(self, delta: np.float32):
        """Create Phi and warn when its metric is singular or ill-conditioned."""
        if not np.isfinite(delta):
            raise ValueError("delta must be a finite angle in degrees")

        matrix = np.ones((2, 2), dtype=np.float32)
        num = np.cos(np.radians(delta))
        matrix[0, 1] = num
        matrix[1, 0] = num
        self.phi = matrix

        condition_number = np.linalg.cond(matrix.astype(np.float64))
        if not np.isfinite(condition_number) or condition_number > 1e6:
            warnings.warn(
                f"delta={float(delta)} degrees makes the opinion metric "
                f"singular or ill-conditioned (condition number={condition_number:.3g})",
                RuntimeWarning,
                stacklevel=2,
            )

    def dist(self, A1: np.ndarray, A2: np.ndarray) -> float:
        """
        Calculate the distance in opinion from one agent to the next agent
        """
        Dir_vec = A1 - A2
        matrix_product = self.phi @ Dir_vec
        return np.sqrt((matrix_product.T @ matrix_product).item())

    def connect_prob_matrix(self, beta: float) -> np.ndarray:
        """
        Calculate the connection probability matrix using Equation (3).
        """
        N = self.N
        opinions = self._opinions
        
        diff = opinions[:, None, :] - opinions[None, :, :]
        
        distances = np.empty((N, N), dtype=np.float32)
        # Bound the temporary transformed-difference array while reducing the
        # Python overhead of processing one source row at a time.
        block_size = 64
        for start in range(0, N, block_size):
            stop = min(start + block_size, N)
            transformed = diff[start:stop] @ self.phi
            distances[start:stop] = np.sqrt(
                np.sum(transformed * transformed, axis=2)
            )
        
        np.fill_diagonal(distances, np.nan)
        
        p_conn = np.zeros((N, N), dtype=np.float32)
        # Keep inverse-power weights finite for coincident or numerically
        # indistinguishable opinions, without perturbing ordinary distances.
        distances = np.maximum(distances, _DISTANCE_EPSILON)
        with np.errstate(divide='ignore', over='ignore'):
            p_conn = distances ** (-beta)
        
        np.fill_diagonal(p_conn, 0)
        
        row_sums = np.nansum(p_conn, axis=1, keepdims=True)
        
        bad_rows = ~np.isfinite(row_sums) | (row_sums == 0)
        good_rows = ~bad_rows[:, 0]

        if np.any(bad_rows):
            p_conn[bad_rows[:, 0]] = 1.0 / (N - 1)
            np.fill_diagonal(p_conn, 0)

        # Normalize every valid row independently, even when another row
        # needed the uniform fallback.
        p_conn[good_rows] /= row_sums[good_rows]
        
        return p_conn.astype(np.float32)

    def set_activity(self, gamma=None, epsilon=None):
        """
        this method calculates the probability of activity for each agent 
        it uses equation 2
        """
        if gamma is None:
            gamma = self.gamma
        if epsilon is None:
            epsilon = self.epsilon
            
        r = np.random.random(self.N).astype(np.float32)
        first = epsilon ** (1 - gamma)
        second = (1 - first) * r
        exp = 1 / (1 - gamma)
        self.activities = np.power(first + second, exp).astype(np.float32)

    def set_social_interactions(self, beta=None, r=None):
        """set up the social interactions"""
        if beta is None:
            beta = self.beta
        if r is None:
            r = self.r
            
        self.beta = beta
        self.r = r
        self.p_conn = self.connect_prob_matrix(beta)

    def create_adj_matrix(self):
        """
        create the adjacency matrix using the connect_prob_matrix
        """
        N = self.N
        adj = self._adj
        adj.fill(0)
        
        active_threshold = np.random.random(N).astype(np.float32)
        active_agents = np.where(self.activities >= active_threshold)[0]
        
        active_list = active_agents.tolist()
        
        for agent in active_list:
            probs = self.p_conn[agent]
            
            if np.count_nonzero(probs) < self.m:
                probs = np.ones(N, dtype=np.float32) / (N - 1)
                probs[agent] = 0
            
            m_local = self.m
            chosen = np.random.choice(
                N,
                size=m_local,
                replace=False,
                p=probs
            )
            adj[chosen, agent] = 1
        
        return adj

    # ============================================================
    # OP DYNAMICS
    # ============================================================

    def op_dynamics(self, t, A_j, N=None, D=None):
        """
        Calculate dx/dt using Equation (1) - 2D Topic Space Version.
        RDN is delegated to the RDN class.
        """
        # Recalculate p_conn periodically
        self._p_conn_dynamics_call_counter += 1
        if (
            self._p_conn_dynamics_call_counter
            % self._p_conn_refresh_every_dynamics_calls
            == 0
        ):
            self.p_conn = self.connect_prob_matrix(self.beta)

        # Each RHS call samples a fresh network. In RK4 this intentionally
        # gives each of the four stages its own stochastic adjacency matrix.
        adj = self.create_adj_matrix()

        transformed = A_j @ self.phi
        transformed_topic1 = transformed[:, 0]
        transformed_topic2 = transformed[:, 1]

        influence_topic1 = np.tanh(self.alpha * transformed_topic1)
        influence_topic2 = np.tanh(self.alpha2 * transformed_topic2)

        influence = np.column_stack([influence_topic1, influence_topic2])

        social_influence = adj @ influence

        # Base dynamics
        dy = -A_j + self.K * social_influence

        # Apply RDN via the RDN class
        if self.rdn.method != 0:
            # Compute global mean for the topic the RDN uses
            topic = self.rdn.topic
            global_mean = np.mean(A_j[:, topic])
            
            # Get the nudge
            nudge = self.rdn.calculate(A_j, global_mean)
            
            # Apply to the correct topic
            if self.rdn.method == 1:
                dy[:, topic] += self.rdn.D * nudge
            # Future methods handled inside the RDN class

        return dy

    # ============================================================
    # RUN METHODS
    # ============================================================

    def run_network(self, dt=0.05, t_dur=50.0):
        """
        Run the simulation using Euler integration.
        """
        dt = np.float32(dt)
        t_dur = np.float32(t_dur)
        num_steps = int(t_dur / dt)

        time_points = np.zeros(num_steps, dtype=np.float32)
        opinions_history = np.zeros((self.N, 2, num_steps), dtype=np.float32)

        y = self._opinions.copy()
        t = 0.0

        dt_local = dt
        op_dynamics = self.op_dynamics

        for i in range(num_steps):
            time_points[i] = t
            opinions_history[:, :, i] = y

            dy = op_dynamics(t, y)
            y += dy * dt_local
            t += dt_local

        self.result = type('obj', (object,), {
            't': time_points,
            'y': opinions_history
        })()

        self._opinions = y

    def run_network_rk4(self, dt=0.05, t_dur=50.0, N=30, D=0.5):
        """
        Run the simulation using 4th order Runge-Kutta (RK4) integration.
        """
        dt = np.float32(dt)
        t_dur = np.float32(t_dur)
        num_steps = int(t_dur / dt)

        time_points = np.zeros(num_steps, dtype=np.float32)
        opinions_history = np.zeros((self.N, 2, num_steps), dtype=np.float32)

        y = self._opinions.copy()
        t = 0.0

        op_dynamics = self.op_dynamics

        for i in range(num_steps):
            time_points[i] = t
            opinions_history[:, :, i] = y

            k1 = op_dynamics(t, y, N=N, D=D)
            k2 = op_dynamics(t + dt/2, y + dt/2 * k1, N=N, D=D)
            k3 = op_dynamics(t + dt/2, y + dt/2 * k2, N=N, D=D)
            k4 = op_dynamics(t + dt, y + dt * k3, N=N, D=D)

            dy = (k1 + 2*k2 + 2*k3 + k4) / 6
            y += dy * dt
            t += dt

        self.result = type('obj', (object,), {
            't': time_points,
            'y': opinions_history
        })()

        self._opinions = y

    # ============================================================
    # GETTERS
    # ============================================================

    def get_N(self) -> int:
        return self.N
    
    def get_m(self) -> int:
        return self.m
    
    def get_K(self) -> float:
        return self.K
    
    def get_alpha(self) -> float:
        return self.alpha
    
    def get_beta(self) -> float:
        return self.beta
    
    def get_r(self) -> float:
        return self.r
    
    def get_gamma(self) -> float:
        return self.gamma
    
    def get_epsilon(self) -> float:
        return self.epsilon

    def get_opinions(self) -> np.ndarray:
        return self._opinions
    
    def get_activities(self) -> np.ndarray:
        return self.activities