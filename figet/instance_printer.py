import torch
from operator import itemgetter
from figet.utils import get_logging
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB, COARSE_FLAG, FINE_FLAG, UF_FLAG
from figet.Predictor import assign_total_types
from figet.evaluate import COARSE

log = get_logging()

ASSIGN = 0
TRUE = 1
CORRECT = 2


def stratify(types, co_fi_ids):
    co_fi, uf = set(), set()
    for t in types.tolist():
        if t in co_fi_ids:
            co_fi.add(t)
        else:
            uf.add(t)
    return co_fi, uf


def get_score(true, predicted):
    """Returns F1 per instance"""
    numerator = len(set(predicted.tolist()).intersection(set(true.tolist())))
    p = numerator / float(len(predicted))
    r = numerator / float(len(true))
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


class InstancePrinter(object):

    def __init__(self, vocabs, model, knn):
        self.token_vocab = vocabs[TOKEN_VOCAB]
        self.type_vocab = vocabs[TYPE_VOCAB]
        self.model = model
        self.knn = knn
        self.coarse_matrixes = [{self.type_vocab.label2idx[label]: [0, 0, 0] for label in COARSE
                                 if label in self.type_vocab.label2idx} for _ in [COARSE_FLAG, FINE_FLAG, UF_FLAG]]
        self.co_fi_ids = self.type_vocab.get_coarse_ids().union(self.type_vocab.get_fine_ids())

    def show(self, data):
        to_show = []
        with torch.no_grad():
            for batch_index in range(len(data)):
                batch = data[batch_index]
                types = batch[5]

                model_loss, predicted_embeds, attn, _, _, _ = self.model(batch)
                partial_result = assign_total_types(predicted_embeds, types, self.knn)

                for i in range(len(partial_result)):
                    true, predicted = partial_result[i]
                    score = get_score(true, predicted)
                    # score, mention_idx, ctx, attn, true, predicted
                    to_show.append([score, batch[3][i], batch[0][i], attn[i].tolist(), true, predicted])

                # self.update_coarse_matrixes(partial_result)

        self.print_results(to_show)

        # self.print_coarse_matrix()

    def print_results(self, to_show):
        unk = "@"
        to_show = sorted(to_show, key=itemgetter(0), reverse=True)
        for score, mention, ctx, attn, true, predicted in to_show:
            mention_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in mention if i != 0])

            ctx_words = [self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in ctx]
            ctx_and_attn = map(lambda t: t[0] + f"({t[1][0]:0.2f})", zip(ctx_words, attn))
            ctx_words = " ".join(ctx_and_attn)

            true_co_fi, true_uf = stratify(true, self.co_fi_ids)
            pred_co_fi, pred_uf = stratify(predicted, self.co_fi_ids)

            true_co_fi_types = " ".join([self.type_vocab.get_label(i) for i in true_co_fi])
            true_uf_types = " ".join([self.type_vocab.get_label(i) for i in true_uf])
            pred_co_fi_types = " ".join([self.type_vocab.get_label(i) for i in pred_co_fi])
            pred_uf_types = " ".join([self.type_vocab.get_label(i) for i in pred_uf])
            # neighbor_types = " ".join([self.type_vocab.get_label(i.item()) for i in neighbors])

            log.debug(f"Mention: '{mention_words}'\nCtx:'{ctx_words}'\n"
                      f"Score: {score * 100:0.2f}: True: co: '{true_co_fi_types}', uf: '{true_uf_types}' - "
                      f"Pred: co:'{pred_co_fi_types}', uf: '{pred_uf_types}'\n\n")

    def update_coarse_matrixes(self, results):
        for idx in range(len(results)):
            result = results[idx]
            matrix = self.coarse_matrixes[idx]

            for true_types, predictions in result:
                true_set = set([x.item() for x in true_types])
                for true_type in true_set:
                    if true_type in matrix:
                        matrix[true_type][TRUE] += 1

                for predicted in [y.item() for y in predictions]:
                    if predicted in matrix:
                        matrix[predicted][ASSIGN] += 1

                        if predicted in true_set:
                            matrix[predicted][CORRECT] += 1

    def print_coarse_matrix(self):
        grans = ["COARSE", "FINE", "ULTRAFINE"]
        for i in range(len(self.coarse_matrixes)):
            matrix = self.coarse_matrixes[i]
            results = []
            for coarse, values in matrix.items():
                label = self.type_vocab.get_label(coarse)
                assign, true, correct = values[ASSIGN], values[TRUE], values[CORRECT]
                p = correct / assign * 100 if assign != 0 else 0
                r = correct / true * 100 if true != 0 else 0
                f1 = 2 * p * r / (p + r) if p + r != 0 else 0
                extra_tab = '    ' if label != 'organization' and label != 'location' else ''
                results.append(f"{label}\t{extra_tab}{assign}/{correct}/{true}\t"
                               f"{p:0.2f}\t{r:0.2f}\t{f1:0.2f}")

            log.info(f"{grans[i]} labels matrix results (assign/correct/true) (P,R,F1):\n" + "\n".join(results))


def is_strictly_right(true, predicted):
    if true.size() != predicted.size():
        return False
    return torch.all(true == predicted).item() == 1


def is_partially_right(true, predicted):
    rights = len(set([i.item() for i in predicted]).intersection(set([j.item() for j in true])))
    return 0 < rights < len(true)


def is_totally_wrong(true, predicted):
    rights = len(set([i.item() for i in predicted]).intersection(set([j.item() for j in true])))
    return rights == 0
