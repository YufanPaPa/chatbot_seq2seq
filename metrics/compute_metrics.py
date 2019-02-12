from .bleu.bleu import Bleu
from .rouge.rouge import Rouge
from .cider.cider import Cider
#from .meteor.meteor import Meteor

def compute_bleu(gts, res, weights=None, bleu_order=4):
    assert set(gts.keys()) == set(res.keys()), f"res missing keys: {set(gts.keys() - set(res.keys()))}"
    if weights:
        assert set(gts.keys()) == set(weights.keys()), f"weights missing keys: {gts.keys() - weights.keys()}"

    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(gts, res, weights)
    for i, bleu_score in enumerate(bleu_scores):
        m_name = f'W-Bleu-{i+1}' if weights else f'Bleu-{i+1}'
        scores[m_name] = bleu_score
    return scores


def compute_rouge(gts, res, weights=None):
    assert set(gts.keys()) == set(res.keys())
    if weights:
        assert set(gts.keys()) == set(res.keys())

    rouge_score, _ = Rouge().compute_score(gts, res, weights)
    return rouge_score

def compute_cider(gts, res, weights=None):
    assert set(gts.keys()) == set(res.keys())
    if weights:
        assert set(gts.keys()) == set(weights.keys())

    cider, _ = Cider().compute_score(gts, res, weights)
    return cider

def compute_meteor(gts, res, weights=None):
    assert set(gts.keys()) == set(res.keys())
    if weights:
        assert set(weights.keys()) == set(gts.keys())
    meteor, _  = Meteor().compute_score(gts, res, weights)
    return meteor


def compute_metrics(gts, res, weights=None):
    bleu_scores = compute_bleu(gts, res, weights)
    rouge = compute_bleu(gts, res, weights)
    cider = compute_cider(gts, res, weights)
    meteor = compute_meteor(gts, res, weights)
    return {
        'BLEU': bleu_scores,
        'Rouge_L': rouge,
        'CIDEr': cider,
        'METEOR': meteor
    }


