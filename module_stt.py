from math import *
import numpy as np
from jplephem.spk import SPK
from scipy.integrate import solve_ivp
import time

def stm_predict(STM, x):
    """Calculate the first-order term using STM"""
    prediction = STM @ x.reshape(len(x), 1)
    return prediction.T[0]

def stt_predict(STT, x, y):
    """Calculate the second-order term using STT (or DSTT)"""
    prediction = np.zeros(len(STT))
    for i in range(len(STT)):
        for k1 in range(len(x)):
            for k2 in range(len(x)):
                prediction[i] += 1 / 2 * STT[i, k1, k2] * x[k1] * y[k2]
    return prediction

def stt_mean_cov(
        P0: np.array,
        STM: np.array,
        STT: np.array,
):
    """Calculate the mean and covariance using the STTs"""
    n, m = len(STM), np.size(STM[0])
    """Mean"""
    mf = np.zeros([n])
    for i in range(n):
        for i1 in range(m):
            for i2 in range(m):
                mf[i] += STT[i, i1, i2] * P0[i1, i2] / 2
    """Covariance"""
    Pf = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Pf[i, j] = -mf[i] * mf[j]
            """First-order"""
            for a in range(m):
                for b in range(m):
                    Pf[i, j] = Pf[i, j] + STM[i, a] * STM[j, b] * P0[a, b]
            """Second-order"""
            for a in range(m):
                for b in range(m):
                    for alpha in range(m):
                        for beta in range(m):
                            Pf[i, j] = Pf[i, j] + STT[i, a, b] * STT[j, alpha, beta] * (P0[a, b] * P0[alpha, beta] + P0[a, alpha] * P0[b, beta] + P0[a, beta] * P0[b, alpha]) / 4
    return mf, Pf

def dstt_mean_cov(
        P0: np.array,
        STM: np.array,
        DSTT: np.array,
        R: np.array,
        dim: int,
):
    """Calculate the mean and covariance using the DSTTs"""
    n, m = len(STM), np.size(STM[0])
    R = np.mat(R)
    P0R = np.matmul(np.matmul(R, P0), R.T)
    """Mean"""
    mf = np.zeros([n])
    for i in range(n):
        for i1 in range(dim):
            for i2 in range(dim):
                mf[i] += DSTT[i, i1, i2] * P0R[i1, i2] / 2
    """Covariance"""
    Pf = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Pf[i, j] = -mf[i] * mf[j]
            """First-order"""
            for a in range(m):
                for b in range(m):
                    Pf[i, j] = Pf[i, j] + STM[i, a] * STM[j, b] * P0[a, b]
            """Second-order"""
            for a in range(dim):
                for b in range(dim):
                    for alpha in range(dim):
                        for beta in range(dim):
                            Pf[i, j] = Pf[i, j] + DSTT[i, a, b] * DSTT[j, alpha, beta] * (P0R[a, b] * P0R[alpha, beta] + P0R[a, alpha] * P0R[b, beta] + P0R[a, beta] * P0R[b, alpha]) / 4
    return mf, Pf
