import pandas as pd
import numpy as np
import h5py as hd
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.stats as sts

from tqdm import tqdm_notebook as tqdm


def frobenius(A, B):
    return np.linalg.norm(A-B, 2)

def normalize_factor_matrices(W, H):
    norms = np.linalg.norm(W, 2, axis=0)
    norm_gt_0 = norms > 0
    W[:, norm_gt_0] /= norms[norm_gt_0]
    H[norm_gt_0, :] = ((H[norm_gt_0, :].T) * norms[norm_gt_0]).T

    return (W, H)

def solve_fast_unbinned(matrix, k=3, max_iter=1000, eps=1e-3):
    n = matrix.shape[-1]
    alpha = 0.
    V = matrix
    W, H = 1.+np.random.random((n, k)), 1.+np.random.random((k, n))
    losses = np.zeros((max_iter))
    for _ in range(max_iter):
        Whb2 = (W@H) ** -1
        Whb1 = (W@H) ** 0
        Wf = np.divide(np.multiply(Whb2, V)@H.T, Whb1@H.T + eps)
        W = np.multiply(W, Wf)
        W = W.clip(min=0)
        
        Whb2 = (W@H) ** -1
        Whb1 = (W@H) ** 0
        Hf = np.divide(W.T@np.multiply(Whb2, V), (W.T@Whb1+eps)**(1+alpha))
        H = np.multiply(H, Hf)
        H = H.clip(min=0)
        
        Ws = np.sum(W, axis=0)+eps
        W = W @ np.diag(1./Ws)
        H = np.diag(Ws)@H
        loss = frobenius(V, W@H)
        losses[_] = loss
    return W, H, losses

def solve_slow_unbinned(matrix, k=3, max_iter=5):
    n = matrix.shape[-1]
    V = matrix.reshape(n, n)
    for _ in range(max_iter):
        H = cp.Variable((k,n))
        obj = cp.Minimize(cp.norm(V - W@H))
        constraint = [H >= 0]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, verbose=False)
        H = H.value
        loss = prob.value
        losses.append(loss)
         
        W = cp.Variable((n,k))
        obj = cp.Minimize(cp.norm(V - W@H))
        constraint = [W >= 0]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, verbose=False)
        W = W.value
        loss = prob.value
        losses.append(loss)
    return W, H, losses

def solve_slow_binned(matrix, k=2, max_iter=5):
    q, n = matrix.shape[0], matrix.shape[2]
    s = q * q
    Vs = matrix.reshape(-1, n, n)
    W = np.random.uniform(size=(n,k))
    losses = []
    
    for _ in tqdm(range(max_iter)):
        Hs = [cp.Variable((k,n)) for i in range(s)]
        obj = cp.Minimize(cp.sum([cp.norm(Vs[i] - W@Hs[i]) for i in range(s)]))
        constraint = [H >= 0 for H in Hs]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, verbose=False)
        Hs = [H.value for H in Hs]
        loss = prob.value
        losses.append(loss)
         
        W = cp.Variable((n,k))
        obj = cp.Minimize(cp.sum([cp.norm(Vs[i] - W@Hs[i]) for i in range(s)]))
        constraint = [W >= 0]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, verbose=False)
        W = W.value
        loss = prob.value
        losses.append(loss)
    return W, Hs, losses