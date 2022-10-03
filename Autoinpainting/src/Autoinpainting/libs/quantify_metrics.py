import numpy as np


def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score


def confusion_matrix(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    return tp, fp, tn, fn


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tn)


def precision(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return tp / (tp + fp)


def recall(P, G):
    return tpr(P, G)


def specificty(P, G):
    return 1 - fpr(P, G)


def evaluate_results_volumetric(pred, truth, thr):
    pred[pred > thr] = 1
    pred[pred <= thr] = 0
    truth[truth > thr] = 1
    truth[truth <= thr] = 0

    pred = pred.astype('bool')
    truth = truth.astype('bool')

    score_dice = dice(pred, truth)
    score_precision = precision(pred, truth)
    score_recall = recall(pred, truth)
    score_fpr = fpr(pred, truth)
    score_spc = specificty(pred, truth)

    return score_dice, score_recall, score_spc, score_precision, score_fpr
