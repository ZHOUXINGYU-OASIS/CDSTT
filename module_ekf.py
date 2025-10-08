"""Import Python toolbox"""
import numpy as np
from numpy.linalg import inv
import time
import scipy
import scipy.io as scio
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from rich.console import Console
from rich.table import Column, Table
from module_crtbp import crtbp, crtbp_stm, crtbp_stt, crtbp_mdstt, crtbp_dstt
from module_stt import stt_mean_cov, dstt_mean_cov
from module_default_settings import setup_seed

"""Load the default parameters"""
from module_default_settings import MIU_EM, UNIT_L, UNIT_V, SEED, STD, INITIAL_STATE, INITIAL_COVARIANCE, \
    NAV_TIMES, DIM_R, RelTol, AbsTol, EPS_DSTT, EPS_CDSTT

def compute_measurement(
        xr: np.array,
        xe: np.array,
        std: float = STD,
):
    """The function of generating measurement"""
    """Simulate the true measurement"""
    y = xr[1] + np.random.randn(1) * std
    """Compute the predicted measurement"""
    h = xe[1]
    dy = y - h
    H = np.array([
        [0, 1, 0, 0, 0, 0],
    ])
    R = np.ones((1, 1)) * (std ** 2)
    """Return results"""
    return dy, H, R

def ekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Module of the extended Kalman filter (EKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    RHS = lambda t, y: crtbp_stm(t=t, y=y, mu=mu)
    """Prepare to record simulation data"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    """Implement EKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...EKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM"""
        start = time.perf_counter()
        X = np.concatenate((xe, np.eye(DIM).reshape(DIM ** 2)))
        sol = solve_ivp(RHS, [t0, tf], X, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = sol.y.T[-1][:DIM]
        STM = sol.y.T[-1][DIM:].reshape(DIM, DIM)
        """Propagate the prior covariance"""
        P = STM @ P @ STM.T + Q
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data

def sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Module of the second-order extended Kalman filter (SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    RHS = lambda t, y: crtbp_stt(t=t, y=y, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    """Implement SEKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr.T
        """Calculate the STM and STT"""
        start = time.perf_counter()
        X = np.concatenate((
            xe,  # orbital state
            np.eye(DIM).reshape(DIM ** 2),  # STM
            np.zeros([DIM, DIM, DIM]).reshape(DIM ** 3),  # STT
        ))
        sol = solve_ivp(RHS, [t0, tf], X, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = sol.y.T[-1][:DIM]
        STM = sol.y.T[-1][DIM:(DIM + DIM ** 2)].reshape(DIM, DIM)
        STT = sol.y.T[-1][(DIM + DIM ** 2):].reshape(DIM, DIM, DIM)
        """Propagate the prior covariance"""
        mf, Pf = stt_mean_cov(P0=P, STM=STM, STT=STT)
        P = Pf + Q
        xe = xe + mf
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data

def compute_dstt(
        x0: np.array,
        t0: float,
        tf: float,
        DIM: int = 6,
        dim_R: int = 1,
        mu: float = MIU_EM,
        if_adaptive: bool = False,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, int]:
    """Compute the second-order DSTT"""
    order = 2
    t_eval = [t0, tf]
    """Compute the sensitive directions"""
    STM0 = np.eye(DIM)
    X0 = np.concatenate((
        x0,  # orbital state
        STM0.reshape(DIM ** 2),  # STM
    ))
    sol = solve_ivp(
        fun=lambda t, y: crtbp_stm(t=t, y=y, mu=mu),
        t_span=[t0, tf],
        y0=X0,
        args=(),
        method="DOP853",
        t_eval=t_eval,
        max_step=np.inf,
        rtol=RelTol,
        atol=AbsTol,
    )
    xf = sol.y.T[-1][:DIM]
    STM = sol.y.T[-1][DIM:].reshape(DIM, DIM)
    CGT = STM.T @ STM
    try:
        eigenvalue, eigenvector = np.linalg.eig(CGT)
    except:
        eigenvalue, eigenvector = np.linalg.eig(CGT + np.eye(DIM) * 1.0e-14)
    # Now implement the adaptive strategy
    if if_adaptive is True:
        if np.max(eigenvalue) <= EPS_DSTT:
            order = 1
            DSTM, DSTT, R = np.zeros((DIM, dim_R)), np.zeros((DIM, dim_R, dim_R)), np.zeros((dim_R, DIM))
            return xf, STM, DSTM, DSTT, R, CGT, eigenvalue, eigenvector, order
    eigenvector = eigenvector.T[np.argsort(-eigenvalue)]
    R = eigenvector[:dim_R]
    """Propagate the DSTT"""
    DSTM0 = np.zeros((DIM, dim_R))
    for i in range(DIM):
        for k1 in range(dim_R):
            for l1 in range(DIM):
                DSTM0[i, k1] = DSTM0[i, k1] + STM0[i, l1] * R[k1, l1]
    X0 = np.concatenate((
        x0,  # orbital state
        DSTM0.reshape(DIM * dim_R),  # DSTM
        np.zeros([DIM, dim_R, dim_R]).reshape(DIM * (dim_R ** 2)),  # (measurement) DSTT
    ))
    sol = solve_ivp(
        fun=lambda t, y: crtbp_dstt(t=t, y=y, mu=mu, dim_r=dim_R),
        t_span=[t0, tf],
        y0=X0,
        args=(),
        method="DOP853",
        t_eval=t_eval,
        max_step=np.inf,
        rtol=RelTol,
        atol=AbsTol,
    )
    DSTM = sol.y.T[-1][DIM:(DIM + DIM * dim_R)].reshape(DIM, dim_R)
    DSTT = sol.y.T[-1][(DIM + DIM * dim_R):].reshape(DIM, dim_R, dim_R)
    """Return results"""
    return xf, STM, DSTM, DSTT, R, CGT, eigenvalue, eigenvector, order

def dstt_sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        dim_R: int = DIM_R,
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Module of the directional STT-based second-order extended Kalman filter (DSTT-SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    eigenvalues = np.zeros((num - 1, DIM))  # used to record the eigenvalues
    eigenvectors = np.zeros((num - 1, DIM, DIM))  # used to record the eigenvectors
    """Implement EKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...DSTT-SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM and DSTT"""
        start = time.perf_counter()
        xe, STM, _, DSTT, Rmatrix, _, eigenvalue, eigenvector, _ = compute_dstt(
            x0=xe,
            t0=t0,
            tf=tf,
            DIM=DIM,
            dim_R=dim_R,
            mu=mu,
            if_adaptive=False,
        )
        eigenvalues[k] = eigenvalue
        eigenvectors[k] = eigenvector.T
        """Propagate the prior covariance"""
        mf, Pf = dstt_mean_cov(P0=P, STM=STM, DSTT=DSTT, R=Rmatrix, dim=dim_R)
        P = Pf + Q
        xe = xe + mf
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data, eigenvalues, eigenvectors

def adaptive_dstt_sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        dim_R: int = DIM_R,
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """Module of the adaptive directional STT-based second-order extended Kalman filter (Adaptive-DSTT-SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    eigenvalues = np.zeros((num - 1, DIM))  # used to record the eigenvalues
    eigenvectors = np.zeros((num - 1, DIM, DIM))  # used to record the eigenvectors
    orders = np.zeros(num - 1)  # used to record the DSTT's order
    """Implement EKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...DSTT-SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM and DSTT"""
        start = time.perf_counter()
        xe, STM, _, DSTT, Rmatrix, _, eigenvalue, eigenvector, order = compute_dstt(
            x0=xe,
            t0=t0,
            tf=tf,
            DIM=DIM,
            dim_R=dim_R,
            mu=mu,
            if_adaptive=True,
        )
        eigenvalues[k] = eigenvalue
        eigenvectors[k] = eigenvector.T
        orders[k] = order
        """Propagate the prior covariance"""
        if order == 2:  # use second-order DSTT
            mf, Pf = dstt_mean_cov(P0=P, STM=STM, DSTT=DSTT, R=Rmatrix, dim=dim_R)
            P = Pf + Q
            xe = xe + mf
        else:  # use only the STM
            P = STM @ P @ STM.T + Q
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data, eigenvalues, eigenvectors, orders

def compute_cdstt(
        x0: np.array,
        t0: float,
        tf: float,
        P0: np.array,
        DIM: int = 6,
        dim_R: int = 1,
        mu: float = MIU_EM,
        if_adaptive: bool = False,
        adaptive_strategy: int = 1,  # 1 (DSTT) or 2 (CDSTT)
        eps_dstt: float = EPS_DSTT,
        eps_cdstt: float = EPS_CDSTT,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, int]:
    """Compute the second-order CDSTT"""
    order: int = 2  # record the maximal order of the STT used
    t_eval = [t0, tf]
    P0 = P0.copy()
    """Compute the sensitive directions"""
    STM0 = np.eye(DIM)
    X0 = np.concatenate((
        x0,  # orbital state
        STM0.reshape(DIM ** 2),  # STM
    ))
    sol = solve_ivp(
        fun=lambda t, y: crtbp_stm(t=t, y=y, mu=mu),
        t_span=[t0, tf],
        y0=X0,
        args=(),
        method="DOP853",
        t_eval=t_eval,
        max_step=np.inf,
        rtol=RelTol,
        atol=AbsTol,
    )
    xf = sol.y.T[-1][:DIM]
    STM = sol.y.T[-1][DIM:].reshape(DIM, DIM)
    A = STM.T @ STM
    # Now implement the adaptive strategy (strategy #1)
    if (if_adaptive is True) and adaptive_strategy == 1:  # use the first adaptive strategy
        CGT = A.copy()
        eigenvalue, eigenvector = np.linalg.eig(CGT)
        if np.max(eigenvalue) <= eps_dstt:
            order = 1
            DSTM, DSTT = np.zeros((DIM, dim_R)), np.zeros((DIM, dim_R, dim_R))
            R, B = np.zeros((dim_R, DIM)), np.zeros((DIM, DIM))
            return xf, STM, DSTM, DSTT, R, A, B, eigenvalue, eigenvector, order
    # Now compute the eigenvalues and eigenvectors
    P0 = (P0 + P0.T) * 0.5
    B = np.linalg.inv(P0)
    try:
        eigenvalue, eigenvector = scipy.linalg.eigh(
            a=A,
            b=B,
            check_finite=False,
        )  # solve a generalized eigenvalue problem
    except:
        B = np.linalg.inv(P0 + np.eye(DIM) * 1.0e-14)
        eigenvalue, eigenvector = scipy.linalg.eigh(
            a=A,
            b=B,
            check_finite=False,
        )  # solve a generalized eigenvalue problem
    # Now implement the adaptive strategy (strategy #1)
    if (if_adaptive is True) and adaptive_strategy == 2:  # use the second adaptive strategy
        if np.max(eigenvalue) <= eps_cdstt:
            order = 1
            DSTM, DSTT = np.zeros((DIM, dim_R)), np.zeros((DIM, dim_R, dim_R))
            R, B = np.zeros((dim_R, DIM)), np.zeros((DIM, DIM))
            return xf, STM, DSTM, DSTT, R, A, B, eigenvalue, eigenvector, order
    # Second-order STT is required
    eigenvector = eigenvector.T[np.argsort(-eigenvalue)]
    R = eigenvector[:dim_R]
    for k in range(dim_R):
        R[k] /= np.linalg.norm(R[k])
    """Propagate the DSTT"""
    DSTM0 = np.zeros((DIM, dim_R))
    for i in range(DIM):
        for k1 in range(dim_R):
            for l1 in range(DIM):
                DSTM0[i, k1] = DSTM0[i, k1] + STM0[i, l1] * R[k1, l1]
    X0 = np.concatenate((
        x0,  # orbital state
        DSTM0.reshape(DIM * dim_R),  # DSTM
        np.zeros([DIM, dim_R, dim_R]).reshape(DIM * (dim_R ** 2)),  # (measurement) DSTT
    ))
    sol = solve_ivp(
        fun=lambda t, y: crtbp_dstt(t=t, y=y, mu=mu, dim_r=dim_R),
        t_span=[t0, tf],
        y0=X0,
        args=(),
        method="DOP853",
        t_eval=t_eval,
        max_step=np.inf,
        rtol=RelTol,
        atol=AbsTol,
    )
    DSTM = sol.y.T[-1][DIM:(DIM + DIM * dim_R)].reshape(DIM, dim_R)
    DSTT = sol.y.T[-1][(DIM + DIM * dim_R):].reshape(DIM, dim_R, dim_R)
    """Return results"""
    return xf, STM, DSTM, DSTT, R, A, B, eigenvalue, eigenvector, order

def cdstt_sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        dim_R: int = DIM_R,
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Module of the covariance-directional STT-based second-order extended Kalman filter (CDSTT-SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    eigenvalues = np.zeros((num - 1, DIM))  # used to record the eigenvalues
    eigenvectors = np.zeros((num - 1, DIM, DIM))  # used to record the eigenvectors
    """Implement EKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...CDSTT-SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM and DSTT"""
        start = time.perf_counter()
        xe, STM, _, DSTT, Rmatrix, _, _, eigenvalue, eigenvector, _ = compute_cdstt(
            x0=xe,
            t0=t0,
            tf=tf,
            P0=P,
            DIM=DIM,
            dim_R=dim_R,
            mu=mu,
            if_adaptive=False,  # adaptive strategy is forbidden here
        )
        eigenvalues[k] = eigenvalue
        eigenvectors[k] = eigenvector.T
        """Propagate the prior covariance"""
        mf, Pf = dstt_mean_cov(P0=P, STM=STM, DSTT=DSTT, R=Rmatrix, dim=dim_R)
        P = Pf + Q
        xe = xe + mf
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data, eigenvalues, eigenvectors

def adaptive_cdstt_sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        dim_R: int = DIM_R,
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        adaptive_strategy: int = 1,  # 1 (DSTT) or 2 (CDSTT)
        eps_dstt: float = EPS_DSTT,
        eps_cdstt: float = EPS_CDSTT,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """Module of the adaptive covariance-directional STT-based second-order extended Kalman filter (Adaptive-CDSTT-SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    eigenvalues = np.zeros((num - 1, DIM))  # used to record the eigenvalues
    eigenvectors = np.zeros((num - 1, DIM, DIM))  # used to record the eigenvectors
    orders = np.zeros(num - 1)  # used to record the DSTT's order
    """Implement EKF process"""
    for k in range(num - 1):
        if if_print is True:
            print("...CDSTT-SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM and DSTT"""
        start = time.perf_counter()
        xe, STM, _, DSTT, Rmatrix, _, _, eigenvalue, eigenvector, order = compute_cdstt(
            x0=xe,
            t0=t0,
            tf=tf,
            P0=P,
            DIM=DIM,
            dim_R=dim_R,
            mu=mu,
            if_adaptive=True,  # adaptive strategy is used here
            adaptive_strategy=adaptive_strategy,
            eps_dstt=eps_dstt,
            eps_cdstt=eps_cdstt,
        )
        eigenvalues[k] = eigenvalue
        eigenvectors[k] = eigenvector.T
        orders[k] = order
        """Propagate the prior covariance"""
        if order == 2:  # use second-order DSTT
            mf, Pf = dstt_mean_cov(P0=P, STM=STM, DSTT=DSTT, R=Rmatrix, dim=dim_R)
            P = Pf + Q
            xe = xe + mf
        else:  # use only the STM
            P = STM @ P @ STM.T + Q
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data, eigenvalues, eigenvectors, orders

def mdstt_sekf(
        x0r: np.array,  # true orbit state
        x0e: np.array,  # estimated orbit state
        t_series: np.array = NAV_TIMES,  # time series
        P0: np.array = INITIAL_COVARIANCE,  # initial covariance
        std: float = STD,  # measurement standard
        mu: float = MIU_EM,
        if_set_seed: bool = False,
        seed: int = SEED,
        if_print: bool = False,
):
    """Module of the measurement-directional STT-based second-order extended Kalman filter (MDSTT-SEKF)"""
    DIM = 6
    if if_set_seed is True:
        setup_seed(seed=seed)
    """Parameters for orbit determination"""
    P = P0.copy()
    Q = np.zeros((DIM, DIM))  # without process noises
    I6 = np.eye(DIM)
    """Define dynamics model"""
    f = lambda t, x: crtbp(t=t, x=x, mu=mu)
    """Begin navigation"""
    num = len(t_series)  # length the of time series
    true_orbit = np.zeros((num, DIM))  # used to record true orbit states
    estimated_orbit = np.zeros((num, DIM))  # used to record estimated orbit state
    true_orbit[0] = x0r
    estimated_orbit[0] = x0e
    time_costs = np.zeros((num - 1, 2))
    cov_data = np.zeros((num, DIM, DIM))  # used to record the estimated covariance
    cov_data[0] = P
    md_data = np.zeros(num)  # used to record the Mahalanobis distance
    md_data[0] = mahalanobis(
        u=estimated_orbit[0],
        v=true_orbit[0],
        VI=np.linalg.inv(P),
    )
    """Implement EKF process"""
    RMatrix = np.eye(DIM)
    dim_R = DIM
    for k in range(num - 1):
        if if_print is True:
            print("...MDSTT-SEKF process, epoch %d, remaining %d epochs..." % (k + 1, num - 2 - k))
        """state update"""
        xr = true_orbit[k]  # true state of the k-th segment
        xe = estimated_orbit[k]  # estimated state of the k-th segment
        t0 = t_series[k]  # starting epoch of the k-th segment
        tf = t_series[k + 1]  # ending epoch of the k-th segment
        """Propagate the true orbit"""
        t_eval = [t0, tf]
        sol = solve_ivp(f, [t0, tf], xr, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xr = sol.y.T[-1, :]
        true_orbit[k + 1] = xr
        """Calculate the STM and MDSTT"""
        start = time.perf_counter()
        X = np.concatenate((
            xe,  # orbital state
            np.eye(DIM).reshape(DIM ** 2),  # STM
            np.zeros([DIM, dim_R, dim_R]).reshape(DIM * (dim_R ** 2)),  # (measurement) DSTT
        ))
        RHS = lambda t, y: crtbp_mdstt(t=t, y=y, mu=mu, R=RMatrix, dim_r=dim_R)
        sol = solve_ivp(RHS, [t0, tf], X, args=(), method="DOP853", t_eval=t_eval,
                        max_step=np.inf, rtol=RelTol, atol=AbsTol)
        xe = sol.y.T[-1][:DIM]
        STM = sol.y.T[-1][DIM:(DIM + DIM ** 2)].reshape(DIM, DIM)
        DSTT = sol.y.T[-1][(DIM + DIM ** 2):].reshape(DIM, dim_R, dim_R)
        """Propagate the prior covariance"""
        mf, Pf = dstt_mean_cov(P0=P, STM=STM, DSTT=DSTT, R=RMatrix, dim=dim_R)
        P = Pf + Q
        xe = xe + mf
        time_costs[k, 0] = time.perf_counter() - start  # record the cpu time of state propagation
        """Measurement update"""
        start = time.perf_counter()
        dy, H, R = compute_measurement(xr=xr, xe=xe, std=std)
        K_k = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        """Update estimations"""
        estimated_orbit[k + 1] = xe + K_k @ dy
        P = (I6 - K_k @ H) @ P @ (I6 - K_k @ H).T + K_k @ R @ K_k.T
        time_costs[k, 1] = time.perf_counter() - start  # record the cpu time of measurement update
        cov_data[k + 1] = P  # record the estimated covariance
        md_data[k + 1] = mahalanobis(
            u=estimated_orbit[k + 1],
            v=true_orbit[k + 1],
            VI=np.linalg.inv(P),
        )  # record the Mahalanobis distance
        """Update R matrix"""
        RMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        dim_R = 5
    """Return results"""
    error = estimated_orbit - true_orbit
    error[:, :3] *= UNIT_L  # scale the position variable: km
    error[:, 3:] *= (UNIT_V * 1e3)  # scale the velocity variable: m/s
    return error, time_costs, cov_data, md_data

if __name__ == '__main__':
    """Main implementation"""
    errR = 2.5e-5
    errV = 1.0e-6
    x0r = INITIAL_STATE
    x0e = x0r + np.concatenate((
        np.ones(3) * errR, np.ones(3) * errV,
    ))
    if_print = True
    """Extended Kalman filter"""
    error, time_cost, _, _ = ekf(
        x0r=x0r,  # true orbit state
        x0e=x0e,  # estimated orbit state
        t_series=NAV_TIMES,  # time series
        P0=INITIAL_COVARIANCE,  # initial covariance
        std=STD,  # measurement standard
        mu=MIU_EM,
        if_set_seed=True,
        if_print=if_print,
    )
    error_ekf = error
    t_ekf = time_cost[:, 0].mean()
    """Second-order extended Kalman filter"""
    error, time_cost, _, _ = sekf(
        x0r=x0r,  # true orbit state
        x0e=x0e,  # estimated orbit state
        t_series=NAV_TIMES,  # time series
        P0=INITIAL_COVARIANCE,  # initial covariance
        std=STD,  # measurement standard
        mu=MIU_EM,
        if_set_seed=True,
        if_print=if_print,
    )
    error_sekf = error
    t_sekf = time_cost[:, 0].mean()
    """Second-order directional extended Kalman filter"""
    error, time_cost, _, _, _, _ = dstt_sekf(
        x0r=x0r,  # true orbit state
        x0e=x0e,  # estimated orbit state
        t_series=NAV_TIMES,  # time series
        P0=INITIAL_COVARIANCE,  # initial covariance
        dim_R=DIM_R,
        std=STD,  # measurement standard
        mu=MIU_EM,
        if_set_seed=True,
        if_print=if_print,
    )
    error_dstt_sekf = error
    t_dstt_sekf = time_cost[:, 0].mean()
    """Second-order covariance-directional extended Kalman filter"""
    error, time_cost, _, _, _, _ = cdstt_sekf(
        x0r=x0r,  # true orbit state
        x0e=x0e,  # estimated orbit state
        t_series=NAV_TIMES,  # time series
        P0=INITIAL_COVARIANCE,  # initial covariance
        dim_R=DIM_R,
        std=STD,  # measurement standard
        mu=MIU_EM,
        if_set_seed=True,
        if_print=if_print,
    )
    error_cdstt_sekf = error
    t_cdstt_sekf = time_cost[:, 0].mean()
    """MDSTT-based second-order extended Kalman filter"""
    error, time_cost, _, _ = mdstt_sekf(
        x0r=x0r,  # true orbit state
        x0e=x0e,  # estimated orbit state
        t_series=NAV_TIMES,  # time series
        P0=INITIAL_COVARIANCE,  # initial covariance
        std=STD,  # measurement standard
        mu=MIU_EM,
        if_set_seed=True,
        if_print=if_print,
    )
    error_mdstt_sekf = error
    t_mdstt_sekf = time_cost[:, 0].mean()
    """Save the one-run error data"""
    file_name = "./data/example.mat"
    scio.savemat(
        file_name, {
            "std": STD,
            "error_ekf": error_ekf,
            "error_sekf": error_sekf,
            "error_dstt_sekf": error_dstt_sekf,
            "error_cdstt_sekf": error_cdstt_sekf,
            "error_mdstt_sekf": error_mdstt_sekf,
        },
    )
    """Table data"""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", justify="left")
    table.add_column("One-step CPU time (s)", justify="left")
    table.add_row("[red bold]EKF[/red bold]", "[bold]%.4f s[/bold]" % t_ekf)
    table.add_row("[blue bold]SEKF[/blue bold]", "[bold]%.4f s[/bold]" % t_sekf)
    table.add_row("[green bold]DSTT-SEKF[/green bold]", "[bold]%.4f s[/bold]" % t_dstt_sekf)
    table.add_row("[green bold]CDSTT-SEKF[/green bold]", "[bold]%.4f s[/bold]" % t_cdstt_sekf)
    table.add_row("[green bold]MDSTT-SEKF[/green bold]", "[bold]%.4f s[/bold]" % t_mdstt_sekf)
    console.print(table)
