#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
    Distributional Inclusive Hypothesis related models
"""

import numpy as np
import logging
from tqdm import tqdm  # tqdm(iterator)
from scipy.stats import entropy

import sys
sys.path.append("..")
from patternBased.preprocess import read_sparse_matrix
from patternBased.base import HypernymySuiteModel,Precomputed

logging.basicConfig()


def invCL(x_row, y_row):
    """
    Computes invCL(x, y)
    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.
    Returns:
        float. Estimation of distributional inclusion.
    """
    return np.sqrt(clarkeDE(x_row, y_row) * (1 - clarkeDE(y_row, x_row)))


def clarkeDE(x_row, y_row):
    """
    clarkeDE similarity
    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.
    Returns:
        float. Estimation of distributional inclusion.
    """
    # Get the sum of the minimum for each context. Only the mutual contexts
    # will yield values > 0
    numerator = np.min([x_row, y_row], axis=0).sum(axis=1)
    # The sum of x's contexts (for ppmi) is the sum of x_row.
    denominator = x_row.sum(axis=1)
    return numerator / (denominator + 1e-12)


def weeds_prec(x_row, y_row):
    """
    WeedsPrec similarity
    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.
    Returns:
        float. Estimation of distributional inclusion.
    """
    # Get the mutual contexts: use y as a binary vector and apply dot product
    # with x: If c is a mutual context, it is 1 in y_non_zero and the value
    # ppmi(x, c) is added to the sum Otherwise, if it is 0 in either x or y, it
    # adds 0 to the sum.
    numerator = np.sum(x_row * (y_row > 0), axis=1)
    # The sum of x's contexts (for ppmi) is the sum of x_row.
    denominator = x_row.sum(axis=1)
    return numerator / (denominator + 1e-12)


def mdot(x_row, y_row):
    """
    Inner product of x and y
    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.
    Returns:
        float. Estimation of distributional inclusion.
    """
    return (x_row * y_row).sum(axis=1)


def cosine(x_row, y_row):
    """
    Cosine similarity
    Args:
        x_row, y_row: ndarray[float]. Vectors for x and y.
    Returns:
        float.
    """
    return mdot(x_row, y_row) / np.sqrt(mdot(x_row, x_row) * mdot(y_row, y_row) + 1e-12)


class baseDIH(object):
    """
    Basic distributional inclusive hypothesis related models, including WeedsPrec, clarkeDE, invCL, cosine.
    """
    def __init__(self, spacefile_path):
        logging.info("Prepping sparse matrix.")
        
        self.spacefile_path = spacefile_path
        self.matrix, self.objects, self.vocab, self.col_vocab = read_sparse_matrix(spacefile_path)
        self.metrics2fns = {
            "weeds_prec": self.predict_weeds_prec,
            "clarkeDE": self.predict_clarkeDE,
            "invCL": self.predict_invCL,
            "cosine": self.predict_cosine
        }
    
    def predict_weeds_prec(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return weeds_prec(lhs, rhs)[0]

    def predict_clarkeDE(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return clarkeDE(lhs, rhs)[0]
    
    def predict_invCL(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return invCL(lhs, rhs)[0]

    def predict_cosine(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        return cosine(lhs, rhs)[0]
    
    def predict(self, hypo, hyper, metric_name="weeds_prec"):
        """
        Core modeling procedure, estimating the degree to which hypo is_a hyper.
        Args:
            hypo: str. A hypothesized hyponym.
            hyper: str. A hypothesized hypernym.
            metric_name: str. A DIH metric name
        Returns:
            float. The score estimating the degree to which hypo is_a hyper.
                Higher values indicate a stronger degree.
        """
        if metric_name == "weeds_prec":
            return self.predict_weeds_prec(hypo, hyper)
        elif metric_name == "clarkeDE":
            return self.predict_clarkeDE(hypo, hyper)
        elif metric_name == "invCL":
            return self.predict_invCL(hypo, hyper)
        elif metric_name == "cosine":
            return self.predict_cosine(hypo, hyper)


class POSSearchDict(object):
    """
    Utility, hack-ish "dictionary" which automatically tries to find the
    most appropriate pos-tagged vector for a given non-tagged word.
    Simply prefers nouns over verbs over adjectives.
    """

    def __init__(self, lookup):
        self.lookup = lookup
        self.cache = {"<OOV>": self.lookup["<OOV>"]}

    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]

        for pos in ["-n", "-v", "-j"]:
            if (item + pos) in self.lookup:
                self.cache[item] = self.lookup[(item + pos)]
                return self.cache[item]

        raise KeyError(item)


class SparseMatrixBaseline(HypernymySuiteModel):
    """
    Abstract class based on sparse, distributional models.
    """

    def __init__(self, space_filename, pos_tagged_space=None):
        logging.info("Prepping sparse matrix")

        self.space_filename = space_filename
        self.matrix, self.objects, full_vocab, col_vocab = read_sparse_matrix(
            space_filename
        )

        if pos_tagged_space is None:
            logging.warning("Having to guess whether this is a POS tagged space.")
            pos_tagged_space = "animal-n" in full_vocab

        # hard hack to mark it as POS tagged or not
        if pos_tagged_space:
            logging.info("This is a POS tagged space.")
            self.vocab = POSSearchDict(full_vocab)
        else:
            logging.info("This is a non-tagged space.")
            self.vocab = full_vocab

    def forward(self, inputs):
        raise NotImplementedError("SparseMatrixBaseline is an abstract class")


class UnsupervisedBaseline(SparseMatrixBaseline):
    """
    Unsupervised distributional similarity model. Based on some similarity
    measure.
    Args:
        space_filename: filename of the 3-column sparse distributional space.
        measurefn: fn[v,v] -> s: a measure taking two (dense) vectors and returns
            their similarity. Can be an asymmetric measure.
    """

    def __init__(self, space_filename, measurefn):
        super(UnsupervisedBaseline, self).__init__(space_filename)
        self.measure = measurefn

    def predict(self, hypo, hyper):
        lhs = self.matrix[self.vocab.get(hypo, 0)].todense().A
        rhs = self.matrix[self.vocab.get(hyper, 0)].todense().A
        result = self.measure(lhs, rhs)
        assert result.shape == (1,)
        return result[0]


class SLQS(SparseMatrixBaseline):
    """
    Core implementation of SLQS model. See Santus 2014.
    Args:
        space_filename: filename of sparse distributional space
        topk: The number of entropy items for each row.
    """

    _row_entropy_cache = {}

    def __init__(self, space_filename, topk):
        super(SLQS, self).__init__(space_filename)
        self.topk = topk
        logging.info("Computing column entropies")
        tr = self.matrix.transpose().tocsr()

        entropies = []
        # Minibatches for computation efficiency
        bs = 1024
        logging.warning("Calculating the entropy maybe slow!")
        for idx_start in tqdm(range(0, tr.shape[0], bs)):
            idx_end = min(idx_start + bs, tr.shape[0])
            v = tr[idx_start:idx_end].todense().A
            entropies += list(entropy(v.T))
        self.colent = np.array(entropies)
        assert len(self.colent) == tr.shape[0]
        logging.info("Done computing entropies")
        # cleanup
        del tr
        logging.info("Done computing row entropies")

    def compute_row_entropy(self, i):
        if i in self._row_entropy_cache:
            return self._row_entropy_cache[i]
        row = self.matrix[i]
        data = row.data
        indx = row.indices
        if len(data) == 0:
            return 0
        k = min(self.topk, len(data))
        ranked = np.argpartition(data, -k)
        sigdims = indx[ranked[-k:]]
        rowent = np.median(self.colent[sigdims])
        self._row_entropy_cache[i] = rowent
        return rowent

    def predict(self, hypo, hyper):
        x = self.vocab.get(hypo, 0)
        y = self.vocab.get(hyper, 0)
        # Compute entropies for each individual word
        ent_x = self.compute_row_entropy(x)
        ent_y = self.compute_row_entropy(y)
        # Prevent divide by zero, which shouldn't happen but does when testing things
        # sometimes
        return 1 - ent_x / (ent_y + 1e-12)


class SLQS_Cos(SLQS):
    """
    Variant of SLQS which also includes the cosine similarity.
    """

    def __init__(self, space_filename, topk):
        super(SLQS_Cos, self).__init__(space_filename, topk)
        self.measure = cosine

    def predict(self, hypo, hyper):
        entropy_measure = super(SLQS_Cos, self).predict(hypo, hyper)
        # Gross hack, inheritence is messed up but python DGAF
        cosines = UnsupervisedBaseline.predict(self, hypo, hyper)
        return entropy_measure * cosines

