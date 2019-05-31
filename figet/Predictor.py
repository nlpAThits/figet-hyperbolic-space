
from pyflann import *
from pyflann.exceptions import FLANNException
import numpy as np
from figet.utils import get_logging
from figet.Constants import COARSE_FLAG, FINE_FLAG, UF_FLAG
from figet.hyperbolic import poincare_distance
import torch
from operator import itemgetter

log = get_logging()
cos_sim_func = torch.nn.CosineSimilarity(dim=0)


def cosine_distance(a, b):
        return 1 - cos_sim_func(a, b)


class kNN(object):
    def __init__(self, type2vec, type_vocab, metric):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type2vec = type2vec.to(self.device).type(torch.float)
        self.type_vocab = type_vocab
        self.hyperbolic = metric == "hyperbolic"
        self.query_factor = 25 if self.hyperbolic else 1

        self.neighs_per_granularity = {COARSE_FLAG: 1, FINE_FLAG: 1, UF_FLAG: 3}

        self.granularity_ids = {COARSE_FLAG: list(type_vocab.get_coarse_ids()),
                                FINE_FLAG: list(type_vocab.get_fine_ids()),
                                UF_FLAG: list(type_vocab.get_ultrafine_ids())}

        self.granularity_sets = {gran_flag: set(ids) for gran_flag, ids in self.granularity_ids.items()}
        self.knn_searchers = {}
        self.checks = {}

        self.build_indexes()

    def build_indexes(self):
        for granularity in [COARSE_FLAG, FINE_FLAG, UF_FLAG]:
            ids = self.granularity_ids[granularity]
            type_vectors = self.type2vec[ids]
            gran_flann = FLANN(log_level="none")
            numpy_vecs = type_vectors.cpu().numpy()
            log.info(f"NUMPY type of vectors in index: {numpy_vecs.dtype}")

            params = gran_flann.build_index(numpy_vecs, algorithm='autotuned', target_precision=0.99,
                                            build_weight=0.01, memory_weight=0, sample_fraction=0.25, log_level="none")
            self.knn_searchers[granularity] = gran_flann
            self.checks[granularity] = params["checks"]
            log.info(f"Finish building index fro gran {granularity} with {len(type_vectors)} vectors")

    def _query_index(self, predictions, gran_flag, k=-1):
        """
        :param gran_flag: flag that indicates the granularity of the search
        :param k: amount of neighbors to retrieve
        """
        max_neighbors = len(self.granularity_ids[gran_flag])
        if k == -1:
            k = self.neighs_per_granularity[gran_flag]
        if k > max_neighbors:
            k = max_neighbors

        predictions_numpy = predictions.detach().cpu().numpy()

        requested_neighbors = self.query_factor * k if self.query_factor * k <= max_neighbors else max_neighbors
        for i in range(3):
            try:
                knn_searcher = self.knn_searchers[gran_flag]
                checks = self.checks[gran_flag]
                indexes, _ = knn_searcher.nn_index(predictions_numpy, requested_neighbors, checks=checks)
                break
            except FLANNException:
                log.debug(f"Index failed: rebuilding index with gran {gran_flag}")
                self.build_indexes()
        mapped_indexes = self.map_indices_to_type2vec(indexes, gran_flag)
        result = []

        if self.hyperbolic:
            for x in range(len(mapped_indexes)):
                idx = mapped_indexes[x]
                predicted = predictions[x]

                idx_and_tensors = list(zip(idx, [tensor for tensor in self.type2vec[idx]]))
                idx_and_distance = [(idx, poincare_distance(predicted, tensor)) for idx, tensor in idx_and_tensors]
                sorted_idx_and_tensors = sorted(idx_and_distance, key=itemgetter(1))
                result.append([sorted_idx_and_tensors[i][0] for i in range(k)])
        else:
            for x in range(len(mapped_indexes)):
                result.append(mapped_indexes[x][:k])

        return torch.LongTensor(result).to(self.device)

    def map_indices_to_type2vec(self, indexes, gran_flag):
        """
        :param indexes: batch x n
        :param gran_flag:
        :return:
        """
        original_ids = self.granularity_ids[gran_flag]
        result = []
        for index in indexes:
            if type(index) == np.int32: index = [index]
            row = [original_ids[value] for value in index]
            result.append(row)
        return result

    def neighbors(self, predictions, k, gran_flag):
        try:
            return self._query_index(predictions, gran_flag, k)
        except ValueError:
            log.debug("Query index failed!!")
            log.debug(predictions)


def assign_types(gran_predictions, type_indexes, predictor):
    """
    From one projection tries to assign types of all granularities
    :param gran_predictions: batch x k
    :param neighbor_indexes: batch x k
    :param type_indexes: batch x type_len
    :return: list of pairs of predicted type indexes, and true type indexes
    """
    coarses = predictor.neighbors(gran_predictions, -1, gran_flag=COARSE_FLAG)
    fines = predictor.neighbors(gran_predictions, -1, gran_flag=FINE_FLAG)
    ufines = predictor.neighbors(gran_predictions, -1, gran_flag=UF_FLAG)
    neighs = [coarses, fines, ufines]

    return get_gold_and_pred(neighs, type_indexes)


def assign_co_plus_uf(predictions, type_indexes, predictor):
    """Assigns coarse types from coarse and ultra-fine predictions"""
    co_co = predictor.neighbors(predictions[COARSE_FLAG], -1, gran_flag=COARSE_FLAG)
    fi_fi = predictor.neighbors(predictions[FINE_FLAG], -1, gran_flag=FINE_FLAG)
    uf_co = predictor.neighbors(predictions[UF_FLAG], -1, gran_flag=COARSE_FLAG)
    uf_uf = predictor.neighbors(predictions[UF_FLAG], -1, gran_flag=UF_FLAG)
    all_neighs = [co_co, fi_fi, uf_co, uf_uf]

    return get_gold_and_pred(all_neighs, type_indexes)


def assign_total_types(predictions, type_indexes, predictor):
    """Assigns coarse types from coarse prediction, fine from fine and uf from uf"""
    coarses = predictor.neighbors(predictions[COARSE_FLAG], -1, gran_flag=COARSE_FLAG)
    fines = predictor.neighbors(predictions[FINE_FLAG], -1, gran_flag=FINE_FLAG)
    ufines = predictor.neighbors(predictions[UF_FLAG], -1, gran_flag=UF_FLAG)
    neighs = [coarses, fines, ufines]

    return get_gold_and_pred(neighs, type_indexes)


def get_gold_and_pred(neighs, type_indexes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = []
    for i in range(len(neighs[0])):
        assigned = sum([items[i].tolist() for items in neighs], [])
        result.append([type_indexes[i], torch.LongTensor(list(set(assigned))).to(device)])

    return result

