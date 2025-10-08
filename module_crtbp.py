import numpy as np
import math
import scipy
from scipy.integrate import solve_ivp

from module_default_settings import MIU_EM

def crtbp(
        t: float,
        x: np.array,
        mu: float = MIU_EM,
):
    """The dynamics model of the CRTBP"""
    r1 = math.sqrt((mu + x[0]) ** 2 + (x[1]) ** 2 + (x[2]) ** 2)
    r2 = math.sqrt((1 - mu - x[0]) ** 2 + (x[1]) ** 2 + (x[2]) ** 2)
    m1 = 1 - mu
    m2 = mu
    dydt = np.array([
        x[3],
        x[4],
        x[5],
        x[0] + 2 * x[4] + m1 * (-mu - x[0]) / (r1 ** 3) + m2 * (1 - mu - x[0]) / (r2 ** 3),
        x[1] - 2 * x[3] - m1 * (x[1]) / (r1 ** 3) - m2 * x[1] / (r2 ** 3),
        -m1 * x[2] / (r1 ** 3) - m2 * x[2] / (r2 ** 3),
    ])
    return dydt

def crtbp_stm(
        t: float,
        y: np.array,
        mu: float = MIU_EM,
):
    """the dynamics of the CRTBP model (with STM)"""
    dim: int = 6
    x = y[:dim]
    STM = y[dim:].reshape(dim, dim)
    """Orbit state"""
    dxdt = crtbp(t=t, x=x, mu=mu)
    """State transition matrix"""
    A = compute_tensor1(x=x, mu=mu)
    dSTM = (A @ STM).reshape(dim ** 2)
    """Return results"""
    dy = np.concatenate((dxdt, dSTM))
    return dy

def crtbp_stt(
        t: float,
        y: np.array,
        mu: float = MIU_EM,
):
    """the dynamics of the CRTBP model (with STM and STT)"""
    dim: int = 6
    x = y[:dim]
    STM = y[dim: (dim + dim ** 2)].reshape(dim, dim)
    STT = y[(dim + dim ** 2):].reshape(dim, dim, dim)
    """Orbit state"""
    dxdt = crtbp(t=t, x=x, mu=mu)
    """State transition matrix"""
    N1 = compute_tensor1(x=x, mu=mu)
    dSTM = (N1 @ STM).reshape(dim ** 2)
    """Full state transition tensor"""
    N2 = compute_tensor2(x=x, mu=mu)
    dSTT = np.zeros((dim, dim, dim))
    for i in range(dim):
        for a in range(dim):
            for b in range(dim):
                for alpha in range(dim):
                    dSTT[i, a, b] = dSTT[i, a, b] + N1[i, alpha] * STT[alpha, a, b]
                    for beta in range(dim):
                        dSTT[i, a, b] = dSTT[i, a, b] + N2[i, alpha, beta] * STM[alpha, a] * STM[beta, b]
    dSTT = dSTT.reshape(dim ** 3)
    """Return results"""
    dy = np.concatenate((dxdt, dSTM, dSTT))
    return dy

def crtbp_dstt(
        t: float,
        y: np.array,
        mu: float = MIU_EM,
        dim_r: int = 1,  # dimension of the reduced state vector
):
    """the dynamics of the CRTBP model (with STM and DSTT)"""
    DIM = 6  # dimension of the full state vector
    x = y[:DIM]
    DSTM = y[DIM:(DIM + DIM * dim_r)].reshape(DIM, dim_r)  # here we only use the DSTM
    DSTT = y[(DIM + DIM * dim_r):].reshape(DIM, dim_r, dim_r)
    """Orbit state"""
    dxdt = crtbp(t=t, x=x, mu=mu)
    """Directional state transition matrix"""
    N1 = compute_tensor1(x=x, mu=mu)
    dDSTM = (N1 @ DSTM).reshape(DIM * dim_r)  # here we directly use the directional STM
    """Directional state transition tensor"""
    N2 = compute_tensor2(x=x, mu=mu)
    dDSTT = np.zeros((DIM, dim_r, dim_r))
    for i in range(DIM):
        for a in range(dim_r):
            for b in range(dim_r):
                for alpha in range(DIM):
                    dDSTT[i, a, b] = dDSTT[i, a, b] + N1[i, alpha] * DSTT[alpha, a, b]
                    for beta in range(DIM):
                        dDSTT[i, a, b] = dDSTT[i, a, b] + N2[i, alpha, beta] * DSTM[alpha, a] * DSTM[beta, b]
    dDSTT = dDSTT.reshape(DIM * (dim_r ** 2))
    """Return results"""
    dy = np.concatenate((dxdt, dDSTM, dDSTT))
    return dy

def crtbp_mdstt(
        t: float,
        y: np.array,
        mu: float = MIU_EM,
        R: np.array = np.array([[1, 0, 0, 0, 0, 0]]),  # here the MDSTT code still require the sensitive direction
        dim_r: int = 1,  # dimension of the reduced state vector
):
    """the dynamics of the CRTBP model (with STM and MDSTT)"""
    DIM = 6  # dimension of the full state vector
    x = y[:DIM]
    STM = y[DIM:(DIM + DIM ** 2)].reshape(DIM, DIM)
    DSTT = y[(DIM + DIM ** 2):].reshape(DIM, dim_r, dim_r)
    """Orbit state"""
    dxdt = crtbp(t=t, x=x, mu=mu)
    """State transition matrix"""
    N1 = compute_tensor1(x=x, mu=mu)
    dSTM = (N1 @ STM).reshape(DIM ** 2)  # here we still use the full STM
    """Directional state transition matrix"""
    DSTM = np.zeros((DIM, dim_r))
    for i in range(DIM):
        for k1 in range(dim_r):
            for l1 in range(dim_r):
                DSTM[i, k1] = DSTM[i, k1] + STM[i, l1] * R[k1, l1]
    """Directional state transition tensor"""
    N2 = compute_tensor2(x=x, mu=mu)
    dDSTT = np.zeros((DIM, dim_r, dim_r))
    for i in range(DIM):
        for a in range(dim_r):
            for b in range(dim_r):
                for alpha in range(DIM):
                    dDSTT[i, a, b] = dDSTT[i, a, b] + N1[i, alpha] * DSTT[alpha, a, b]
                    for beta in range(DIM):
                        dDSTT[i, a, b] = dDSTT[i, a, b] + N2[i, alpha, beta] * DSTM[alpha, a] * DSTM[beta, b]
    dDSTT = dDSTT.reshape(DIM * (dim_r ** 2))
    """Return results"""
    dy = np.concatenate((dxdt, dSTM, dDSTT))
    return dy

def compute_tensor1(
        x: np.array,
        mu: float,
):
    """the first-order tensor of the CRTBP dynamics"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    """Compute the elements"""
    daxdrx = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) + 1
    daxdry = (3 * mu * ry * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdrz = (3 * mu * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daxdvy = 2
    daydrx = (3 * mu * ry * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * ry * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    daydry = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * ry ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + 1
    daydrz = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    daydvx = -2
    dazdrx = (3 * mu * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))
    dazdry = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    dazdrz = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)
    """Jacobi matrix"""
    A = np.zeros([6, 6])
    A[:3, 3:] = np.eye(3)
    A[3:, :] = np.array([
        [daxdrx, daxdry, daxdrz, 0, daxdvy, 0],
        [daydrx, daydry, daydrz, daydvx, 0, 0],
        [dazdrx, dazdry, dazdrz, 0, 0, 0],
    ])
    """Return results"""
    return A

def compute_tensor2(
        x: np.array,
        mu: float,
):
    """the second-order tensor of the CRTBP dynamics"""
    rx = x[0]
    ry = x[1]
    rz = x[2]
    """Compute the elements"""
    daxdrxrx = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (3 * mu * (2 * mu + 2 * rx - 2)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrxrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdryrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdryry = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * ry ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdryrz = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daxdrzry = (15 * ry * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * ry * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daxdrzrz = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + (15 * rz ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) - (15 * mu * rz ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrxrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxry = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrxrz = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydryrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydryry = (9 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydryrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    daydrzry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    daydrzrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrxrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz * (2 * mu + 2 * rx - 2) ** 2) / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz * (mu - 1) * (2 * mu + 2 * rx) ** 2) / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxry = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrxrz = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdryrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdryry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdryrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzrx = (3 * mu * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (3 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))
    dazdrzry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    dazdrzrz = (9 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (9 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * rz ** 3) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) + (15 * rz ** 3 * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)
    """Compute the second-order Jacobi tensor"""
    A = np.zeros([6, 6, 6])
    A[3, :3, :3] = np.array([
        [daxdrxrx, daxdrxry, daxdrxrz],
        [daxdryrx, daxdryry, daxdryrz],
        [daxdrzrx, daxdrzry, daxdrzrz],
    ])
    A[4, :3, :3] = np.array([
        [daydrxrx, daydrxry, daydrxrz],
        [daydryrx, daydryry, daydryrz],
        [daydrzrx, daydrzry, daydrzrz],
    ])
    A[5, :3, :3] = np.array([
        [dazdrxrx, dazdrxry, dazdrxrz],
        [dazdryrx, dazdryry, dazdryrz],
        [dazdrzrx, dazdrzry, dazdrzrz],
    ])
    """Return results"""
    return A

if __name__ == '__main__':
    """Main test"""
    print("CRTBP dynamics module")
