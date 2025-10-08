import time
import numpy as np
import scipy.io as scio
from multiprocessing import Pool
from tqdm import tqdm
from module_ekf import ekf, sekf, dstt_sekf, cdstt_sekf, mdstt_sekf, adaptive_cdstt_sekf
from module_default_settings import setup_seed

"""Load the default parameters"""
from module_default_settings import MIU_EM, UNIT_L, UNIT_V, SEED, STD, INITIAL_STATE, INITIAL_COVARIANCE, \
    NAV_TIMES, DIM_R, RelTol, AbsTol, N_JOBS

def one_run_simulation(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        method: str = "EKF",
        dim_R: int = DIM_R,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
        if_save: bool = False,
) -> tuple[np.array, np.array, np.array, np.array]:
    """One-run simulation"""
    """Implement different filter algorithms"""
    if method == "EKF":
        """Extended Kalman filter"""
        error, time_costs, cov_data, md_data = ekf(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=t_series,  # time series
            P0=P0,  # initial covariance
            std=std,  # measurement standard
            mu=mu,
            if_set_seed=if_set_seed,
            seed=seed,
            if_print=if_print,
        )
        eigenvalues, eigenvectors = [], []
    elif method == "SEKF":
        """Second-order extended Kalman filter"""
        error, time_costs, cov_data, md_data = sekf(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=t_series,  # time series
            P0=P0,  # initial covariance
            std=std,  # measurement standard
            mu=mu,
            if_set_seed=if_set_seed,
            seed=seed,
            if_print=if_print,
        )
        eigenvalues, eigenvectors = [], []
    elif method == "DSTT_SEKF":
        """Second-order directional extended Kalman filter"""
        error, time_costs, cov_data, md_data, eigenvalues, eigenvectors = dstt_sekf(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=t_series,  # time series
            P0=P0,  # initial covariance
            dim_R=dim_R,
            std=std,  # measurement standard
            mu=mu,
            if_set_seed=if_set_seed,
            seed=seed,
            if_print=if_print,
        )
    elif method == "CDSTT_SEKF":
        """Second-order covariance-directional extended Kalman filter"""
        error, time_costs, cov_data, md_data, eigenvalues, eigenvectors = cdstt_sekf(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=t_series,  # time series
            P0=P0,  # initial covariance
            dim_R=dim_R,
            std=std,  # measurement standard
            mu=mu,
            if_set_seed=if_set_seed,
            seed=seed,
            if_print=if_print,
        )
    elif method == "MDSTT_SEKF":
        """MDSTT-based second-order extended Kalman filter"""
        error, time_costs, cov_data, md_data = mdstt_sekf(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=t_series,  # time series
            P0=P0,  # initial covariance
            std=std,  # measurement standard
            mu=mu,
            if_set_seed=if_set_seed,
            seed=seed,
            if_print=if_print,
        )
        eigenvalues, eigenvectors = [], []
    else:
        raise ValueError("No such method. Please select from EKF/SEKF/DSTT_SEKF/CDSTT_SEKF/MDSTT_SEKF")
    """Save data"""
    if if_save is True:
        if method == "DSTT_SEKF" or method == "CDSTT_SEKF":
            filepath = f"./data/one_results_{method}_{dim_R}.mat"
        else:
            filepath = f"./data/one_results_{method}.mat"
        print(f"Writing file: {filepath}")
        scio.savemat(
            filepath, {
                "x0r": x0r,
                "x0e": x0e,
                "t_series": t_series,
                "P0": P0,
                "std": std,
                "seed": seed,
                "error": error,
                "time_costs": time_costs,
                "cov_data": cov_data,
                "md_data": md_data,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
            },
        )
    """Return results"""
    return error, time_costs, cov_data, md_data

if __name__ == '__main__':
    """Main implementation"""
    """Parameters setting"""
    errR = 2.5e-5
    errV = 1.0e-6
    x0r = INITIAL_STATE
    x0e = x0r + np.concatenate((
        np.ones(3) * errR, np.ones(3) * errV,
    ))
    if_print = False
    dim_R = DIM_R
    """Implement one-run simulations"""
    method_lists = ["EKF", "SEKF", "DSTT_SEKF", "CDSTT_SEKF", "MDSTT_SEKF"]
    for method in method_lists:
        error, time_costs, cov_data, md_data = one_run_simulation(
            x0r=x0r,  # true orbit state
            x0e=x0e,  # estimated orbit state
            t_series=NAV_TIMES,  # time series
            P0=INITIAL_COVARIANCE,  # initial covariance
            std=STD,  # measurement standard
            mu=MIU_EM,
            method=method,
            dim_R=dim_R,
            if_set_seed=True,
            seed=SEED,
            if_print=True,
            if_save=True,
        )
