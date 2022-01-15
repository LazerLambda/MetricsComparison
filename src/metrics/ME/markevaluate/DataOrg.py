"""Class to pre-organize data to improve speed."""

import math
import numpy as np
import sys

from progress.bar import ShadyBar
from typing import Tuple
from sklearn.neighbors import BallTree


class DataOrg:
    """Class to organize data.

    Pre-ordering and computing essential
    data to improve computational speed.
    """

    def __init__(
        self,
        cand: np.ndarray,
        ref: np.ndarray,
        k: int = 1,
        verbose: bool = False,
        orig: bool = False
    ) -> None:
        """Initialize data structures and compute knn."""
        if len(cand) == 0 or len(ref) == 0:
            exc_str: str = (
                "Set cannot be empty!\n\t\'->"
                "len(cand) == {}, len(cand) == {}").format(
                    len(ref),
                    len(cand))
            raise Exception(exc_str)

        if len(cand) < k or len(ref) < k:
            exc_str: str = (
                "Set cannot be smaller than k!\n\t\'->"
                " len(ref) == {}, len(cand) == {}, k = {}").format(
                    len(ref),
                    len(cand),
                    k)
            raise Exception(exc_str)

        self.verbose: bool = verbose
        self.orig: bool = orig

        # cand and ref as sets
        self.cand_embds: np.ndarray = np.unique(cand, axis=0)
        self.ref_embds: np.ndarray = np.unique(ref, axis=0)

        # cand
        # array to store k-nearest-neighbors
        # self.cand_knnghrhd: np.ndarray =\
        #     np.zeros((
        #         len(self.cand_embds),
        #         k + 1,
        #         self.cand_embds.shape[1]))
        self.cand_knnghrhd: list = []
        self.cand_knns_indx: np.ndarray =\
            np.zeros((len(self.cand_embds), k + 1))
        self.cand_kmaxs: np.ndarray =\
            np.zeros((len(self.cand_embds), 1))

        # ref
        # array to store k-nearest-neighbors
        # self.ref_knnghrhd: np.ndarray =\
        #     np.zeros((
        #         len(self.ref_embds),
        #         k + 1,
        #         self.ref_embds.shape[1]))
        self.ref_knnghrhd: list = []
        self.ref_knns_indx: np.ndarray =\
            np.zeros((len(self.ref_embds), k + 1))
        self.ref_kmaxs: np.ndarray =\
            np.zeros((len(self.ref_embds), 1))

        if self.verbose:
            bar = ShadyBar(
                'Constructing data structure',
                max=(len(self.cand_embds) + len(self.ref_embds) + 2))

        self.cand_kdt: BallTree = BallTree(self.cand_embds, metric='euclidean')
        if self.verbose:
            bar.next()

        self.ref_kdt: BallTree = BallTree(self.ref_embds, metric='euclidean')
        if self.verbose:
            bar.next()

        # Adapt size k to the actual length of cand and ref
        if len(self.cand_embds) < k + 1:
            k = len(self.cand_embds) - 1

        if len(self.ref_embds) < k + 1:
            k = len(self.ref_embds) - 1

        # BallTree for candidate embeddings
        for i in range(len(self.cand_embds)):

            knns_dist, knns_indx = self.cand_kdt.query(
                [self.cand_embds[i]],
                k=(k + 1))

            self.cand_knns_indx[i] = knns_indx[0]
            self.cand_kmaxs[i] = max(knns_dist[0])
            # self.cand_knnghrhd[i] =\
            #     self.cand_embds[knns_indx]
            self.cand_knnghrhd.append({
                tuple(e) for e in self.cand_embds[knns_indx][0]
            })
            if self.verbose:
                bar.next()

        # BallTree for reference embeddings
        for i in range(len(self.ref_embds)):

            knns_dist, knns_indx = self.ref_kdt.query(
                [self.ref_embds[i]],
                k=(k + 1))

            self.ref_knns_indx[i] = knns_indx[0]
            self.ref_kmaxs[i] = max(knns_dist[0])
            # self.ref_knnghrhd[i] =\
            #     self.ref_embds[knns_indx]
            self.ref_knnghrhd.append({
                tuple(e) for e in self.ref_embds[knns_indx][0]
            })
            if self.verbose:
                bar.next()

        if self.verbose:
            bar.finish()

        # check point for future parallelism
        if False:
            raise NotImplementedError("This section is not implemented!")
        else:
            # In-hypersphere-binary-matrix
            self.bin_vec_cand: np.ndarray =\
                self.create_bin_vec(
                    self.cand_embds,
                    self.ref_embds,
                    self.ref_kmaxs)
            self.bin_vec_ref: np.ndarray =\
                self.create_bin_vec(
                    self.ref_embds,
                    self.cand_embds,
                    self.cand_kmaxs)

            if orig:
                self.bin_mat_cand: np.ndarray =\
                    self.create_bin_matrix(
                        self.cand_embds,
                        self.ref_embds,
                        self.ref_knnghrhd,
                        self.ref_kmaxs)
                self.bin_mat_ref: np.ndarray =\
                    self.create_bin_matrix(
                        self.ref_embds,
                        self.cand_embds,
                        self.cand_knnghrhd,
                        self.cand_kmaxs)
        
        self.k: int = k

    def create_bin_vec(
            self,
            samples: np.ndarray,
            set_to_check: np.ndarray,
            kmaxs: np.ndarray) -> None:
        """Compute hypersphere.

        Create binary array which entails
        0/1 values about whether a sample of
        a set lies in the k-nearest-neighborhoood
        of a sample from the set to be checked.
        """
        bin_vec: np.ndarray = np.zeros(len(samples))

        if self.verbose:
            bar = ShadyBar(
                "Determining capture neighborhood (Vektor)",
                max=len(samples))

        for i, sample_outer in enumerate(samples):
            for j, sample_inner in enumerate(set_to_check):
                dist: float = np.linalg.norm(
                    sample_inner - sample_outer)
                # largest distance in the knn-set can
                # only be the distance to the k-distant-
                # neighbor
                if dist <= kmaxs[j]:
                    bin_vec[i] = 1
                    break
            if self.verbose:
                bar.next()
        if self.verbose:
            bar.finish()

        return bin_vec

    def create_bin_matrix(
            self,
            samples: np.ndarray,
            set_to_check: np.ndarray,
            knn_of_set: np.ndarray,
            kmaxs: np.ndarray) -> None:
        """Determine binary function for KNN.

        Given two sets and the respective k-nn set
        of one set, this functions computes a 0/1
        matrix which sample from the first set lies
        closer to a sample from each knn-set than
        the longest distance in the knn-set.
        m_{i,j} = f(s_i, KNN(s_j, S))
        """
        bin_matrix: np.ndarray = np.zeros((
            len(samples),
            len(set_to_check)))

        if self.verbose:
            bar = ShadyBar(
                "Determining capture neighborhood (Matrix)",
                max=len(samples) * len(set_to_check))

        for i, sample in enumerate(samples):
            for j, s_ in enumerate(set_to_check):
                kmax: float = kmaxs[j]
                for kn in knn_of_set[j]:
                    if np.linalg.norm(sample - kn) <= kmax:
                        bin_matrix[i][j] = 1
                        continue
                if self.verbose:
                    bar.next()
        if self.verbose:
            bar.finish()

        return bin_matrix

    # TODO rename cand_in_hypsphr
    def in_hypsphr_cand(self, i: int) -> int:
        """Cand. smpl. in hypersphere.

        Determine whether sample at index i
        in the cand set lies in the euclidean
        k-nearest-neighborhood of a sample of
        the reference set.
        """
        return self.bin_vec_cand[i]

    # TODO rename ref_in_hypsphr
    def in_hypsphr_ref(self, i: int):
        """Ref. smpl. in hypersphere.

        Determine whether sample at index i
        in the ref set lies in the euclidean
        k-nearest-neighborhood of a sample
        of the candidate set.
        """
        return self.bin_vec_ref[i]

    def cand_in_hypsphr_knn(self):
        """Sum binary function for KNN (cand).

        Ref set := S
        sum_{s in S} sum_{s' in S'} f(s', KNN(s, S))
        """
        return self.bin_mat_cand.sum(axis=0).sum()

    def ref_in_hypsphr_knn(self):
        """Sum binary function for KNN (ref).

        Cand set := S'
        sum_{s in S} sum_{s' in S'} f(s, KNN(s', S'))
        """
        return self.bin_mat_ref.sum(axis=0).sum()

    def in_knghbd_cand_ref(
            self,
            ind_cand: int,
            ind_ref: int) -> int:
        """Compute indicator function cand samples.

        Determine whether sample from candidate set
        lies in the k-nearest-neighborhood of a
        sample in the ref set at position ind_ref.
        """
        dist: float = np.linalg.norm(
            self.cand_embds[ind_cand] - self.ref_embds[ind_ref])
        kmax: float = self.ref_kmaxs[ind_ref][0]
        # due to precision issues
        return 1\
            if dist < kmax or math.isclose(dist, kmax)\
            else 0

    def in_knghbd_ref_cand(
            self,
            ind_ref: int,
            ind_cand: int) -> int:
        """Compute indicator function ref samples.

        Determine whether sample from reference set
        lies in thec k-nearest-neighborhood of a
        sample in the cand set at position ind_cand.
        """
        dist: float = np.linalg.norm(
            self.ref_embds[ind_ref] - self.cand_embds[ind_cand])
        kmax: float = self.cand_kmaxs[ind_cand][0]
        # due to precision issues
        return 1\
            if dist < kmax or math.isclose(dist, kmax)\
            else 0

    def get_knn_set_cand(self, i: int) -> set:
        """Get KNN from cand samples.

        Return set of k-nearest-neighbors for
        sample in cand set at position i.
        """
        # return {
        #     tuple(elem)
        #     for elem in self.cand_knnghrhd[i]}
        return self.cand_knnghrhd[i]

    def get_knn_set_ref(self, i: int) -> set:
        """Get KNN from ref samples.

        Return set of k-nearest-neighbors for
        sample in ref set at position i.
        """
        # return {
        #     tuple(elem)
        #     for elem in self.ref_knnghrhd[i]}
        return self.ref_knnghrhd[i]

    def switch_input(self) -> None:
        """Switch candidate and reference set."""
        self.ref_embds, self.cand_embds =\
            self.cand_embds, self.ref_embds

        self.bin_vec_ref, self.bin_vec_cand =\
            self.bin_vec_cand, self.bin_vec_ref

        self.ref_knnghrhd, self.cand_knnghrhd =\
            self.cand_knnghrhd, self.ref_knnghrhd

        self.ref_knns_indx, self.cand_knns_indx =\
            self.cand_knns_indx, self.ref_knns_indx

        self.ref_kdt, self.cand_kdt =\
            self.cand_kdt, self.ref_kdt

        self.ref_kmaxs, self.cand_kmaxs =\
            self.cand_kmaxs, self.ref_kmaxs

        if self.orig:
            self.bin_mat_ref, self.bin_mat_cand =\
                self.bin_mat_cand, self.bin_mat_ref