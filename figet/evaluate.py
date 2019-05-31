import sys
import torch
import json

from figet.Constants import COARSE, FINE


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def strict(true_and_prediction):
    """
    Correct: all types must be predicted exactly equal to the label
    """
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if true_labels.size() != predicted_labels.size():
            continue
        correct_num += torch.all(true_labels == predicted_labels).item()
    precision = recall = correct_num / num_entities
    return precision, recall, f1(precision, recall)


def loose_macro(true_and_prediction):
    """Metrics at mention level (dividing for the amount of instances/examples/mentions).
    Takes an average of the metrics on the amount of mentions
    Code taken from OpenType Repository (using the exact same metrics than them)
    """
    p = 0.
    r = 0.
    pred_count = 0.
    gold_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        numerator = len(set([i.item() for i in predicted_labels]).intersection(set([j.item() for j in true_labels])))
        if len(predicted_labels):
            pred_count += 1
            p += numerator / float(len(predicted_labels))
        if len(true_labels):
            gold_count += 1
            r += numerator / float(len(true_labels))

    precision, recall = 0., 0.
    if pred_count:
        precision = p / pred_count
    if gold_count:
        recall = r / gold_count
    return precision, recall, f1(precision, recall)


def loose_micro(true_and_prediction):
    """Metrics at type/class level.
    Correct types of all types on all mentions"""
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set([i.item() for i in predicted_labels]).intersection(set([j.item() for j in true_labels])))

    if num_predicted_labels == 0 or num_true_labels == 0:
        return 0., 0., 0.

    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)


def evaluate(true_and_prediction, verbose=False):
    ret = ""
    p, r, f = strict(true_and_prediction)
    if verbose:
        ret += "| strict (%.2f, %.2f, %.2f) " %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    p, r, f = loose_macro(true_and_prediction)
    if verbose:
        ret += "| macro (%.2f, %.2f, %.2f) " %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    p, r, f = loose_micro(true_and_prediction)
    if verbose:
        ret += "| micro (%.2f, %.2f, %.2f) |" %(p*100, r*100, f*100)
    else:
        ret += "%.2f\t%.2f\t%.2f\t" % (p * 100, r * 100, f * 100)
    return ret


def raw_evaluate(true_and_prediction):
    metrics = [strict, loose_macro, loose_micro]
    res = []
    for metric in metrics:
        p, r, f = metric(true_and_prediction)
        res.append((p * 100, r * 100, f * 100))
    return res


def stratified_evaluate(true_and_prediction, type_dict):
    coarse_true_and_preds = []
    fine_true_and_preds = []
    ultrafine_true_and_preds = []

    for true_labels, predicted_labels in true_and_prediction:
        coarse_gold, fine_gold, ultrafine_gold = stratify(true_labels, type_dict)
        coarse_pred, fine_pred, ultrafine_pred = stratify(predicted_labels, type_dict)
        coarse_true_and_preds.append((coarse_gold, coarse_pred))
        fine_true_and_preds.append((fine_gold, fine_pred))
        ultrafine_true_and_preds.append((ultrafine_gold, ultrafine_pred))

    return [evaluate(true_and_preds) for true_and_preds in
            [coarse_true_and_preds, fine_true_and_preds, ultrafine_true_and_preds]]


def stratify(labels, type_dict):
    """
    Divide label into three categories.
    """
    coarse_ids = type_dict.get_coarse_ids()
    fine_ids = type_dict.get_fine_ids()
    labels = [i.item() for i in labels]
    strats = ([l for l in labels if l in coarse_ids],
              [l for l in labels if l in fine_ids and l not in coarse_ids],
              [l for l in labels if l not in coarse_ids and l not in fine_ids])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return list(map(lambda strat: torch.LongTensor(strat).to(device), strats))
