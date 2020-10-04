import pandas as pd
import numpy as np
import h5py as hd
import matplotlib.pyplot as plt
import os
import cvxpy as cp
import scipy.stats as sts

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
from pyjet import cluster, DTYPE_PTEPM
from pyjet.testdata import get_event

from . import util

###############################################
#       LHC-Olympics loader and utils         #
# ##############################################


class LHCOLoader:

    def __init__(self, file_name="../data/anomaly.h5", N=1100000, R=1.0, ptcut=0, n_bins=60, anomaly_frac=1.0):
        self.N      = N
        self.R      = R
        self.ptcut  = ptcut
        self.frac   = anomaly_frac
        self.n_bins = n_bins
        # Set maxval by hand
        self.max_val = 800
        fn = file_name.split("/")[-1]
        training = fn == "anomaly"
        try:
            self.masses = np.load("../data/LHCO/{3}/masses_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
            self.labels = np.load("../data/LHCO/{3}/labels_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
            self.etas   = np.load("../data/LHCO/{3}/etas_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
        except:
            print("We didn't find the processed data files. Generating now...")
            util.process_and_cache_data(file_name, N, R, ptcut, training=training, force=True)
            self.masses = np.load("../data/LHCO/{3}/masses_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
            self.labels = np.load("../data/LHCO/{3}/labels_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
            self.etas   = np.load("../data/LHCO/{3}/etas_N{0}_R{1}_cut{2}.npy".format(N, R, ptcut, fn))
        self.bins   = np.linspace(0, self.max_val, self.n_bins+1)
        self.induce_signal_fraction(self.frac)

    def induce_signal_fraction(self, anomaly_frac):
        ixs   = np.arange(self.N)
        bg_ix = ixs[self.labels == 0]
        sg_ix = ixs[self.labels == 1]
        self.max_ix = max_ix = int(anomaly_frac*len(sg_ix))
        # shuffle to select a random fraction
        np.random.shuffle(sg_ix)
        np.random.shuffle(bg_ix)
        # cut to the desired signal fraction
        bg_ix_cut = bg_ix
        sg_ix_cut = sg_ix[:max_ix]
        all_ix    = np.concatenate([bg_ix_cut, sg_ix_cut])
        np.random.shuffle(all_ix)
        self.x      = x      = self.masses[all_ix]
        self.y_true = y_true = self.labels[all_ix]
        self.z      = z      = self.etas[all_ix]
        self.N = len(self.x)
        return x, y_true, z

    def get_histograms_binned(self, n_quantile=3, symmetric=True):
        # First, generate the masks for the eta bins
        quantile_split = np.linspace(0, 1, n_quantile+1)
        quantiles_0 = np.quantile(self.z[:, 0], quantile_split)
        quantiles_1 = np.quantile(self.z[:, 1], quantile_split)
        masks_0 = [(quantiles_0[i] < self.z[:, 0]) & (quantiles_0[i+1] >= self.z[:, 0]) for i in range(n_quantile)]
        masks_1 = [(quantiles_1[i] < self.z[:, 1]) & (quantiles_1[i+1] >= self.z[:, 1]) for i in range(n_quantile)]
        quantile_size = int(self.N / n_quantile) - 1

        #Second, iterate through cartesian product of masks and bin appropriately
        #Store the histogram as H, and the ground truth mixture fractions as F
        var_0, var_1 = self.x[:, 0], self.x[:, 1]

        histograms = []
        self.bins = bins = np.linspace(0, self.max_val, self.n_bins+1)

        for quantile_0, mask_0 in enumerate(tqdm(masks_0)):
            for quantile_1, mask_1 in enumerate(masks_1):
                masked_var_0 = var_0[mask_0]
                masked_var_1 = var_1[mask_1]

                # Sometimes rounding issues make these different lengths, so just cut off a datapoint if needed
                if masked_var_0.shape != (quantile_size,):
                    masked_var_0 = masked_var_0[:quantile_size]
                if masked_var_1.shape != (quantile_size,):
                    masked_var_1 = masked_var_1[:quantile_size]

                # If symmetrize, then make the histogram symmetric by concatenating data
                if symmetric:
                    masked_var_01 = np.concatenate([masked_var_0, masked_var_1])
                    masked_var_10 = np.concatenate([masked_var_1, masked_var_0])
                    masked_var_0 = masked_var_01
                    masked_var_1 = masked_var_10

                H, _, _ = np.histogram2d(masked_var_0, masked_var_1, bins=[bins, bins], density=False)
                histograms.append(H)

        self.histograms = np.array(histograms).reshape(n_quantile, n_quantile, self.n_bins, self.n_bins)
        return histograms

    def get_histograms_unbinned(self, symmetric=True):
        var_0, var_1 = self.x[:, 0], self.x[:, 1]
        self.bins = bins = np.linspace(0, self.max_val, self.n_bins+1)

        if symmetric:
            var_01 = np.concatenate([var_0, var_1])
            var_10 = np.concatenate([var_1, var_0])
            var_0 = var_01
            var_1 = var_10

        H,_,_ = np.histogram2d(var_0, var_1, bins=[bins,bins], density=False)
        self.histograms = H
        return H

    def plot_histogram_sample(self):
        if self.histograms is None:
            raise AssertionError("No histogram found -- please make it first!")

        if len(self.histograms.shape) == 2:
            matrix = self.histograms
        elif len(self.histograms.shape) == 4:
            matrix = self.histograms[0, 0]

        cmap = plt.cm.jet
        cmap.set_under(color='white')
        plt.matshow(
            matrix[::-1], cmap=cmap, extent=[0, max(self.bins), 0, max(self.bins)],
            aspect=max(self.bins)/max(self.bins), vmin=10
        )
        plt.gca().xaxis.tick_bottom()
        plt.gca().set_xlabel("Mass of jet 1")
        plt.gca().set_ylabel("Mass of jet 2")
        plt.title("Histogram as a matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def normalize_and_order_components(self, W, plot=False):
        centered_bins = (self.bins[1:] + self.bins[:-1])/2.
        N = self.N
        n, k = W.shape
        # First, normalize each component.
        w_normalized = []
        for i in range(k):
            # Cut anything less than zero -- you can get floating point error
            w = W[:, i]
            w[w < 0] = 0.
            w /= w.sum()
            w_normalized.append(W[:, i])

        w1, w2, w3 = w_normalized
        # This is a hacky solution that requires you to have some knowledge of the
        # true solution -- i.e., you need to know that there are 3 components,
        # and the resonances have peaks at 100 and 500.

        # Here, we sample from the distribution induced by the components to form a KDE,
        # from which we can extract the pdf value at a specific location.

        # TODO: Please improve this...
        resamples1 = np.random.choice(centered_bins, size=N*5, p=w1)
        component1_kde = sts.gaussian_kde(resamples1)
        hist1,_ = np.histogram(component1_kde.resample(N*5), bins=self.bins, density=True)

        resamples2 = np.random.choice(centered_bins, size=N*5, p=w2)
        component2_kde = sts.gaussian_kde(resamples2)
        hist2,_ = np.histogram(component2_kde.resample(N*5), bins=self.bins, density=True)

        resamples3 = np.random.choice(centered_bins, size=N*5, p=w3)
        component3_kde = sts.gaussian_kde(resamples3)
        hist3,_ = np.histogram(component3_kde.resample(N*5), bins=self.bins, density=True)

        # Again, this is fine for the LHCO dataset specifically.
        kdes  = [component1_kde, component2_kde, component3_kde]
        hists = [hist1, hist2, hist3]
        if plot:
            plt.title("Extracted components")
            plt.plot(centered_bins, w1)
            plt.plot(centered_bins, w2)
            plt.plot(centered_bins, w3)
            plt.show()
        # Here, the 1st component is the 100GeV resonance,
        #       the 2nd component is the 500GeV resonance,
        #.      the 3rd component is the background signal
        _,component1 = max(zip(kdes, hists), key=lambda x: x[0](100))
        _,component2 = max(zip(kdes, hists), key=lambda x: x[0](500))
        _,component3 = max(zip(kdes, hists), key=lambda x: x[0](30))
        return component1, component2, component3

    def orthogonalize_components(self, W, plot=False):
        component1, component2, component3 = self.normalize_and_order_components(W)

        # Now, we need to subtract out the background, for the purpose of mutual irreducibility
        # This is something akin to orthogonalization of the signals, but not quite.
        # TODO: this is also pretty hacky, there's definitely a more principled way of doing this
        component1 = component1 - (component3)
        component2 = component2 - (component3)
        component3 = component3 - (component1 + component2) / 2.
        component1[component1 < 0] = 0.
        component2[component2 < 0] = 0.
        component3[component3 < 0] = 0.
        component1 /= component1.sum()
        component2 /= component2.sum()
        component3 /= component3.sum()

        if plot:
            centered_bins = (self.bins[1:] + self.bins[:-1])/2.
            plt.title("Orthogonalized components")
            plt.plot(centered_bins, component1, drawstyle="steps")
            plt.plot(centered_bins, component2, drawstyle="steps")
            plt.plot(centered_bins, component3, drawstyle="steps")
            plt.show()
        return component1, component2, component3

    def retrieve_learned_mixing_matrices_unbinned(self, W):
        n, k = W.shape
        H = self.histograms
        F = cp.Variable((k,k),symmetric=True)
        obj = cp.Minimize(cp.norm(H - W@F@W.T))
        constraint = [F >= 0]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, verbose=False)
        F = F.value
        # Normalize
        F = F / np.sum(F)
        return F

    def plot_ROC_curve(self, W, verbose=True):
        centered_bins = (self.bins[1:] + self.bins[:-1])/2.
        component1, component2, component3 = self.normalize_and_order_components(W)
        # Now, we compute the probability that each datapoint was generated under each of 2 hypotheses:
        # Hypothesis 1: data is anomalous:
        #       P(m1, m2 | H) = P(m1 | c1) * P(m2 | c2) + P(m1 | c2) * P(m2 | c1)
        # Hypothesis 2: data is background:
        #       P(m1, m2 | H) = P(m1 | c3) * P(m2 | c3)
        # This is the Bayesian way with equal priors.
        # It is possible to include the implied mixing fractions here as well, but we won't do that for now
        # W_n = np.vstack([component1, component2, component3]).T
        # mixing_matrix = self.retrieve_learned_mixing_matrices_unbinned(W_n)

        probs = []
        # Get the corresponding bin for each datapoint
        x1_bins = np.digitize(self.x[:, 0], centered_bins)-1
        x2_bins = np.digitize(self.x[:, 1], centered_bins)-1
        # Compute probabilities
        prob_1 = np.array([component1[a] for a in x1_bins]) * np.array([component2[a] for a in x2_bins])
        prob_2 = np.array([component1[a] for a in x2_bins]) * np.array([component2[a] for a in x1_bins])
        prob_bg = np.array([component3[a] for a in x1_bins]) * np.array([component3[a] for a in x2_bins])
        prob_sg = np.maximum(prob_1, prob_2)
        prob_bg = prob_bg
        # Perform a likelihood ratio test
        y_proba = prob_sg / (prob_bg + prob_sg + 1e-6)
        ids = np.argsort(y_proba)
        # Pick out the indices which correspond to the largest {frac} of datapoints by LRT
        # and set them to be 1s in our prediction
        ids = ids[::-1]
        ids = ids[:self.max_ix]
        y_pred = np.zeros(len(self.x))
        y_pred[ids] = 1

        # print out some sklearn stuff
        if verbose:
            plt.hist(y_proba, bins=np.linspace(0, 1, 50))
            plt.title("Histogram of logits")

            plt.show()
            print("----- FRACTION = {0} -------".format(self.frac))
            print(confusion_matrix(self.y_true, y_pred))
            print(classification_report(self.y_true, y_pred))
        fpr, tpr, thresholds = roc_curve(self.y_true, y_proba)
        auc = roc_auc_score(self.y_true, y_proba)
        return tpr, fpr, auc

###############################################
#        Quark-gluon loader and utils         #
# ##############################################

class QGLoader:
    def __init__(self, file_name="../data/copy.npz", var_name=None, max_val=None, n_bins=60):

        file = np.load("../data/copy.npz")
        obs = file["obs"]
        self.var_name = var_name

        self.pt    = obs[:,:, 0].reshape(-1, 2)
        self.eta   = obs[:,:,-1].reshape(-1, 2)
        self.mult  = obs[:,:, 2].reshape(-1, 2)
        self.mass  = obs[:,:, 1].reshape(-1, 2)
        self.n95   = obs[:,:, 3].reshape(-1, 2)
        self.nsub  = obs[:,:, 12].reshape(-1, 2)

        self.labels = obs[:,:,-2].reshape(-1, 2)
        self.labels[self.labels <= 20] = 0
        self.labels[self.labels >= 20] = 1

        self.histograms = None
        self.mixing_matrices = None

        self.n_bins   = n_bins
        self.n_labels = 2
        self.N        = self.pt.shape[0]

        if var_name == "multiplicity":
            self.var    = self.mult
            self.max    = 120

        elif var_name == "mass":
            self.var    = self.mass
            self.max    = 140

        elif var_name == "nsubjettiness":
            self.var    = self.nsub
            self.max    = 0.2


        elif var_name == "n95":
            self.var    = self.n95
            self.max    = 50

        else:
            raise AssertionError("The variable you have selected \
                is not supported in the current dataset. Please  \
                choose from the set {0}.".format(str(VAR_NAMES)))

        if max_val is not None:
            self.max = max_val

    def get_histograms_unbinned(self, symmetric=True):
        var_0, var_1 = self.var[:, 0], self.var[:, 1]
        labels_0, labels_1 = self.labels[:, 0], self.labels[:, 1]
        bins = np.linspace(0, self.max, self.n_bins+1)

        if symmetrize:
            var_01 = np.concatenate([var_0, var_1])
            var_10 = np.concatenate([var_1, var_0])
            var_0 = var_01
            var_1 = var_10

        H,_,_ = np.histogram2d(var0, var1, bins=[bins,bins], density=False)

        # Now get ground truth for mixing matrix
        F = np.zeros(self.n_labels, self.n_labels)
        for (label_0, label_1) in zip(masked_labels_0, masked_labels_1):
            F[int(label_0), int(label_1)] += 1

        # Similarly make mixing matrix symmetric if necessary
        if symmetric:
            F = (F + F.T) / 2.

        self.histograms = H
        self.mixing_matrices = F
        return H, F

    def get_histograms_binned(self, n_quantile=3, symmetric=True):
        # First, generate the masks for the eta bins
        self.n_quantile = n_quantile
        quantile_split = np.linspace(0, 1, n_quantile+1)
        quantiles_0 = np.quantile(self.eta[:, 0], quantile_split)
        quantiles_1 = np.quantile(self.eta[:, 1], quantile_split)
        self.masks_0 = masks_0 = [(quantiles_0[i] < self.eta[:, 0]) & (quantiles_0[i+1] >= self.eta[:, 0]) for i in range(n_quantile)]
        self.masks_1 = masks_1 = [(quantiles_1[i] < self.eta[:, 1]) & (quantiles_1[i+1] >= self.eta[:, 1]) for i in range(n_quantile)]
        quantile_size = int(self.N / n_quantile) - 1

        #Second, iterate through cartesian product of masks and bin appropriately
        #Store the histogram as H, and the ground truth mixture fractions as F
        var_0, var_1 = self.var[:, 0], self.var[:, 1]
        labels_0, labels_1 = self.labels[:, 0], self.labels[:, 1]

        histograms = []
        mixing_matrices = []
        self.bins = bins = np.linspace(0, self.max, self.n_bins+1)

        for quantile_0, mask_0 in enumerate(tqdm(masks_0)):
            for quantile_1, mask_1 in enumerate(masks_1):
                masked_var_0 = var_0[mask_0]
                masked_var_1 = var_1[mask_1]

                # Sometimes rounding issues make these different lengths, so just cut off a datapoint if needed
                if masked_var_0.shape != (quantile_size,):
                    masked_var_0 = masked_var_0[:quantile_size]
                if masked_var_1.shape != (quantile_size,):
                    masked_var_1 = masked_var_1[:quantile_size]

                # If symmetrize, then make the histogram symmetric by concatenating data
                if symmetric:
                    masked_var_01 = np.concatenate([masked_var_0, masked_var_1])
                    masked_var_10 = np.concatenate([masked_var_1, masked_var_0])
                    masked_var_0 = masked_var_01
                    masked_var_1 = masked_var_10

                H, _, _ = np.histogram2d(masked_var_0, masked_var_1, bins=[bins, bins], density=False)
                histograms.append(H)

                # Now get ground truth for mixing matrix
                F = np.zeros((self.n_labels, self.n_labels))
                masked_labels_0 = labels_0[mask_0]
                masked_labels_1 = labels_1[mask_1]
                for (label_0, label_1) in zip(masked_labels_0, masked_labels_1):
                    F[int(label_0), int(label_1)] += 1

                # Similarly make mixing matrix symmetric if necessary
                if symmetric:
                    F = (F + F.T) / 2.

                F = F/np.sum(F)
                mixing_matrices.append(F)

        self.histograms = np.array(histograms).reshape(n_quantile, n_quantile, self.n_bins, self.n_bins)
        self.mixing_matrices = np.array(mixing_matrices).reshape(n_quantile, n_quantile, self.n_labels, self.n_labels)
        return histograms, mixing_matrices

    def plot_histogram_sample(self):
        if self.histograms is None:
            raise AssertionError("No histogram found -- please make it first!")

        if len(self.histograms.shape) == 2:
            matrix = self.histograms
        elif len(self.histograms.shape) == 4:
            # pick a bin arbitrarily
            matrix = self.histograms[0, 0]

        cmap = plt.cm.jet
        cmap.set_under(color='white')
        plt.matshow(
            matrix[::-1], cmap=cmap, extent=[0, max(self.bins), 0, max(self.bins)],
            aspect=max(self.bins)/max(self.bins), vmin=10
        )
        plt.gca().xaxis.tick_bottom()
        plt.gca().set_xlabel("{0} of jet 1".format(self.var_name))
        plt.gca().set_ylabel("{0} of jet 2".format(self.var_name))
        plt.title("Histogram as a matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def retrieve_learned_mixing_matrices(self, W):
        n, k = W.shape
        learned_mixing_matrices = []
        for quantile_0, mask_0 in enumerate(tqdm(self.masks_0)):
            for quantile_1, mask_1 in enumerate(self.masks_1):
                H = self.histograms[quantile_0, quantile_1]
                F = cp.Variable((k,k),symmetric=True)
                obj = cp.Minimize(cp.norm(H - W@F@W.T))
                constraint = [F >= 0]
                prob = cp.Problem(obj, constraint)
                prob.solve(solver=cp.SCS, verbose=False)
                F = F.value
                # Normalize
                F = F / np.sum(F)
                learned_mixing_matrices.append(F)
        learned_mixing_matrices = np.array(learned_mixing_matrices)
        learned_mixing_matrices = learned_mixing_matrices.reshape(
            self.n_quantile, self.n_quantile, self.n_labels, self.n_labels
        )
        return learned_mixing_matrices
