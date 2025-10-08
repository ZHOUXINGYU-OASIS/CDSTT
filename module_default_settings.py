import numpy as np
import random
import multiprocessing
import scipy.io as scio

"""Parameters for the numerical propagation"""
RelTol: float = 1.0e-12
AbsTol: float = 1.0e-12

"""Parameters for the Earth-Moon system"""
MIU_EM: float = 0.0121505839705277  # gravitational constant of the Earth-Moon system
UNIT_L: float = 384400.0
UNIT_V: float = 1.02454629434750
UNIT_T: float = 375190.464423878

"""Parameters of the orbit states"""
MAT_DATA = scio.loadmat("NRHO_DATA.mat")  # load the data of the 9:2 NRHO
INITIAL_STATE: np.array = MAT_DATA["x0"].T[0]  # initial state of the 9:2 NRHO
PERIOD: float = MAT_DATA["period"][0, 0]  # orbital period of the 9:2 NRHO
NAV_TIMES: float = MAT_DATA["nav_time"][0]

INITIAL_COVARIANCE = np.diag(
    np.array([2.5e-5, 2.5e-5, 2.5e-5, 1.0e-6, 1.0e-6, 1.0e-6]) ** 2  # initial covariance
)

DIM_R: int = 1

"""Parameters of the measurements"""
STD: float = 1.0e-3 / UNIT_L
# STD: float = 1.0e-4 / UNIT_L

"""Parameters of the Monte Carlo simulation"""
SEED: int = 42  # random seed

def setup_seed(
        seed: int = SEED,
):
    """Set the random seed"""
    np.random.seed(seed)
    random.seed(seed)

"""Parameters of the adaptive strategy"""
EPS_DSTT: float = 1.0e2
EPS_CDSTT: float = 1.0e-10

"""Parameters for parallel computing"""
N_JOBS: int = max(1, (multiprocessing.cpu_count() * 2) // 3)
