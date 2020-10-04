import pandas as pd
import numpy as np
import h5py as hd
import matplotlib.pyplot as plt
import os
import cvxpy as cp

from tqdm import tqdm_notebook as tqdm
import pyjet
from pyjet import cluster, DTYPE_PTEPM
from pyjet.testdata import get_event

def get_mass(E, px, py, pz):
    return (E**2 - px**2 - py**2 - pz**2)**(0.5)

def get_masses(j1, j2):
    m1 = get_mass(j1.e, j1.px, j1.py, j1.pz)
    m2 = get_mass(j2.e, j2.px, j2.py, j2.pz)
    m12 = get_mass((j1.e+j2.e), (j1.px+j2.px), (j1.py+j2.py), (j1.pz+j2.pz))
    return np.array([np.real(m1), np.real(m2), np.real(m12)])

def process_and_cache_data(fn, N, R=1.0, ptcut=0, force=False, training=False):
    if not force and os.path.exists("../data/LHCO/masses_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut)):
        # skip out because we have it already
        return
    # N = number of events; p = number of points per event
    p = 700
    chunksize = 50000
    masses_full, etas_full, labels_full = [], [], []
    starts = [chunksize*(i)   for i in range(0, int(N/chunksize))]
    stops  = [chunksize*(i+1) for i in range(0, int(N/chunksize))]
    for i in tqdm(range(len(starts))):
        df = pd.read_hdf(fn, start=starts[i], stop=stops[i])
        if training:
            X = df.iloc[:,:-1].values
            y = df.iloc[:,-1].values
            print(X.shape, y.shape)
            X = np.reshape(X, (chunksize, p, 3))

            # Remove extra zero padding from events
            arr = []
            ys = []
            for i,x in enumerate((X)):
                x = x[x[:, 0] != 0]
                eta_phi_avg = np.average(x[:, 2], weights=x[:, 0], axis=0)
                x[:, 2] -= eta_phi_avg
                arr.append(x)

            masses  = []
            mults   = []
            ys      = []
            etas    = []
            for i, event in enumerate((arr)):
                event = np.insert(event, 3, 0, axis=1)
                event = np.asarray([tuple(a) for a in event], dtype=DTYPE_PTEPM)
                cs = pyjet.cluster(event, algo="antikt", R=R, ep=False)
                jets = sorted(cs.inclusive_jets(), key=lambda x: x.pt, reverse=True)
                if len(jets) >= 3 and jets[0].pt > ptcut:
                    j1, j2  = jets[0], jets[1]
                    f = np.array([list(a) for a in j1.constituents_array()])
                    event_mass = get_masses(j1, j2)
                    masses.append(event_mass)
                    etas.append(np.array([j1.eta, j2.eta]))
                    ys.append(y[i])

            masses = np.array(masses)
            etas = np.array(etas)
            labels = np.array(ys)
            masses_full.append(masses)
            etas_full.append(etas)
            labels_full.append(labels)
        else:
            X = df.values
            X = np.reshape(X, (N, p, 3))

            # Remove extra zero padding from events
            arr = []
            for i,x in enumerate((X)):
                x = x[x[:, 0] != 0]
                eta_phi_avg = np.average(x[:, 2], weights=x[:, 0], axis=0)
                x[:, 2] -= eta_phi_avg
                arr.append(x)

            masses  = []
            mults   = []
            etas    = []
            for i, event in enumerate((arr)):
                event = np.insert(event, 3, 0, axis=1)
                event = np.asarray([tuple(a) for a in event], dtype=DTYPE_PTEPM)
                cs = pyjet.cluster(event, algo="antikt", R=R, ep=False)
                jets = sorted(cs.inclusive_jets(), key=lambda x: x.pt, reverse=True)
                if len(jets) >= 3 and jets[0].pt > ptcut:
                    j1, j2  = jets[0], jets[1]
                    f = np.array([list(a) for a in j1.constituents_array()])
                    event_mass = get_masses(j1, j2)
                    masses.append(event_mass)
                    etas.append(np.array([j1.eta, j2.eta]))

            masses = np.array(masses)
            etas = np.array(etas)
            labels = np.zeros(len(masses))
            masses_full.append(masses)
            etas_full.append(etas)
            labels_full.append(labels)

    masses_full = np.concatenate(masses_full)
    labels_full = np.concatenate(labels_full)
    etas_full   = np.concatenate(etas_full)
    np.save("../data/masses_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut), masses_full)
    np.save("../data/labels_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut), labels_full)
    np.save("../data/etas_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut), etas_full)




