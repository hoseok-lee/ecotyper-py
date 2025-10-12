from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.metrics import root_mean_squared_error
from time import time
from tqdm import tqdm
from collections import defaultdict
from typing import Literal
from dataclasses import dataclass

import scipy.sparse as sp
import numpy as np

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


def sparse_rmse(A, B) -> float:

    return np.sqrt(
        (A - B) \
            .power(2) \
            .mean()
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

        nmf_infos = list(nmf_runs.values())

        # Compute the difference of cophenetic correlation from cutoff
        # Positive values above cutoff, negatives below cutoff
        diff = np.array([nmf_info.cophenet for nmf_info in nmf_infos]) - cutoff

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
        solver: Literal['mu', 'cd'] = 'cd'
    ):

        nmf_runs = defaultdict(FastNMF.NMFInfo)

        for rank in tqdm(
            rank_range,
            desc = f"Ranks",
            leave = False
        ):

            # Row by row <-- cummulative connectivity matrix
            conns = np.zeros((self.X.shape[1], self.X.shape[1]))
            best_fit = -1

            W = np.zeros((self.X.shape[0], rank))
            H = np.zeros((rank, self.X.shape[1]))

            for restart in tqdm(
                nmf_restarts,
                desc = "Restarts",
                leave = False
            ):

                model = NMF(
                    n_components = rank,
                    solver = solver,
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
            d = 1 - C

            # Upper triangle off-diagonal
            coph_corr, _ = cophenet(
                linkage(d, method='average'),
                d[np.triu_indices(d.shape[0], k = 1)]
            )

            nmf_runs[rank] = FastNMF.NMFInfo(
                rank = rank,
                W = W,
                # Normalize by column --> each cell has probability of state
                H = H / H.sum(axis = 0),
                connectivity_mat = C,
                reconstruction_err = best_fit,
                cophenet = coph_corr
            )

        return nmf_runs[
            self.select_rank(
                nmf_runs,
                cutoff = cutoff
            )
        ]
