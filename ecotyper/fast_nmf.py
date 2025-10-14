from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from tqdm import tqdm
from collections import defaultdict
from typing import Literal
from dataclasses import dataclass
from typing import Optional
from matplotlib.ticker import MaxNLocator

import scipy.sparse as sp
import numpy as np
import os
import matplotlib.pyplot as plt

from .utils import sparse_rmse


def connectivity_matrix(X):

    return sum(
        # x(i) is True if belong to cluster i
        # Matrix multiplication computes to 1 if (i, j) is both True
        np.matmul(x.T, x)

        for cluster in range(X.max() + 1)
        # For each cluster membership
        # Skip if there are no elements of a certain cluster
        if (x := (X == cluster).reshape(1, -1)).any()
    )


class FastNMF:

    @dataclass
    class NMFInfo:

        rank: int
        W: np.ndarray
        H: np.ndarray
        connectivity_mat: float
        reconstruction_err: float
        cophenet: float


    def __init__(
        self,
        X,
        random_state: int = 0
    ):

        self.X = X
        self.random_state = random_state


    def select_rank(
        self,
        nmf_runs: dict,
        cutoff: float = 0.95
    ):

        # Rank is selected by the following two conditions
        # 1. First rank with cophenetic correlation below cutoff after two
        #    previous ranks above threshold
        # 2. Closest correlation to cutoff adjacent to crossover point
        # For example, with cutoff 0.95
        # [ 1, 0.98, 0.93, 0.97, 0.97, 0.94, 0.93, ... ]
        #                                ^ selects this rank

        # Compute the difference of cophenetic correlation from cutoff
        # Positive values above cutoff, negatives below cutoff
        diff = np.array([nmf_runs[rank].cophenet for rank in nmf_runs]) - cutoff

        # Let's dissect this...
        arg = np.argwhere(
            np.convolve(
                # Converts the difference into a binary array such that
                # positive values (and 0) are 1, negative values are -1
                (diff >= 0) * 2 - 1,

                # Through convolution, seeks a pattern such that there is a
                # positive, positive, negative difference (above, above, below)
                # (Note that convolution will flip this before mapping)
                [ -1, 1, 1 ]
            # Convolution value will equal 3 if pattern is matched
            ) == 3
        )

        # No pattern found, just go with max rank
        if not np.any(arg):
            return max(nmf_runs)

        # Choose the first crossing point, there can be multiple
        cross = arg.flatten()[0]
        idx = cross if abs(diff[cross]) < abs(diff[cross - 1]) else cross - 1

        return list(nmf_runs.keys())[idx]


    def estimate_rank(
        self,
        rank_range: list = range(2, 20 + 1),
        nmf_restarts: list = range(1, 5 + 1),
        cutoff: float = 0.95,
        solver: Literal['mu', 'cd'] = 'cd',
        figures: Optional[str] = None
    ):

        nmf_runs = defaultdict(FastNMF.NMFInfo)

        for rank in rank_range:

            # Row by row <-- cummulative connectivity matrix
            conns = np.zeros((self.X.shape[1], self.X.shape[1]))
            best_fit = -1

            W = np.zeros((self.X.shape[0], rank))
            H = np.zeros((rank, self.X.shape[1]))

            for restart in nmf_restarts:

                model = NMF(
                    n_components = rank,
                    # init = 'nndsvdar',
                    shuffle = True,
                    # solver = "mu",
                    # beta_loss = 'kullback-leibler',
                    random_state = self.random_state + restart
                )

                W_ = model.fit_transform(self.X)
                H_ = model.components_

                # Use RMSE, reconstruction_err_ from sklearn uses Frobenius
                rmse = sparse_rmse(
                    self.X,
                    sp.csr_matrix(np.matmul(W_, H_))
                )

                cell_states = np.argmax(H_, axis = 0)
                conns += connectivity_matrix(cell_states)

                # Although it is ugly, minimizes number of loops
                # Update best fitting NMF
                if (best_fit == -1) or (rmse < best_fit):
                    best_fit = rmse
                    W = W_
                    H = H_

            # Consensus connectivity matrix
            C = conns / len(nmf_restarts)

            # Convert distance (1 - C) to condensed matrix form
            # squareform performs inverse if square matrix --> upper-triangle
            d = squareform(1 - C)

            coph_corr, _ = cophenet(linkage(d, method = 'average'), d)

            nmf_runs[rank] = FastNMF.NMFInfo(
                rank = rank,
                W = W,
                # Normalize by column --> each cell has probability of state
                H = H / H.sum(axis = 0),
                connectivity_mat = C,
                reconstruction_err = best_fit,
                cophenet = coph_corr
            )

        selected_rank = self.select_rank(nmf_runs, cutoff = cutoff)

        if figures:

            os.makedirs(figures, exist_ok = True)
            plt.cla()

            # Cophenetic coefficient
            plt.plot(
                nmf_runs.keys(),
                [
                    nmf_runs[rank].cophenet
                    for rank in nmf_runs
                ],
                marker = 'o',
                color = 'black',
                linestyle = '-',
                label = "Cophenetic Coefficient"
            )

            # Cutoff point
            plt.axhline(
                y = cutoff,
                color = 'black',
                linestyle = '--',
                label = "Cophenetic Cutoff"
            )

            # Selected rank
            plt.axvline(
                x = selected_rank,
                color = 'red',
                linestyle = '--',
                label = "Selected Rank"
            )

            plt.xlabel("Rank")
            plt.ylabel("Cophenetic Coefficient")

            # This is strictly to avoid floating point values being labeled for
            # x-ticks --> force them to be integers
            plt.xticks(list(map(int, nmf_runs)))
            plt.legend()
            plt.savefig(figures / "cophenet.png")

        return nmf_runs[selected_rank]
