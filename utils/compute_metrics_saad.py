# -*- coding: utf-8 -*-

import collections
import json
import argparse

from convert_searchqa_to_squad import SearchQA
from transformers.data.metrics.squad_metrics import compute_f1, compute_exact


def compute_metrics_from_nbest(searchqa_dir, split, fname_nbest_preds):
    qid2preds = collections.defaultdict(list)
    
    with open(fname_nbest_preds) as rf:
        preds = json.load(rf)
        for uid in preds:
            qid, idx = uid.split("_")
            # Take only non-empty predictions
            # pred = [p for p in preds[uid] if p["text"]][0]
            pred = preds[uid][0] # sorted by probs (we are taking best one here)
            ans, score = pred["text"], pred["probability"]
            qid2preds[int(qid)].append((ans, score))
    
    preds_qid2ans = dict()
    
    for qid, ans in qid2preds.items():
        # select best answer from all paragraphs
        ans, _ = sorted(ans, key=lambda x: x[1], reverse=True)[0]
        preds_qid2ans[qid] = ans
    
    val_gold = list(SearchQA(searchqa_dir, split))
    gold_qid2ans = dict()
    
    for sqa_gold_inst in val_gold:
        gold_qid2ans[sqa_gold_inst.id] = sqa_gold_inst.a
    
    qid2f1 = dict()
    qid2em = dict()
    
    for qid in preds_qid2ans.keys():
        a_pred = preds_qid2ans[qid]
        a_gold = gold_qid2ans[qid]
        qid2f1[qid] = compute_f1(a_gold, a_pred)
        qid2em[qid] = compute_exact(a_gold, a_pred)
    
    f1 = sum(list(qid2f1.values())) / len(qid2f1)
    f1 *= 100
    em = sum(list(qid2em.values())) / len(qid2em)
    em *= 100
    
    metrics = {"f1": f1, "em": em}
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument(
        "--searchqa_dir",
        default=None, type=str, required=True,
        help="SearchQA data directory containing `train.txt`, `val.txt` and `test.txt`."
    )
    parser.add_argument(
        "--split", 
        default=None, type=str, required=True,
        help="Data split."
    )
    parser.add_argument(
        "--nbest_predictions", 
        default=None, type=str, required=True,
        help="nbest_predictions_.json path"
    )
    args = parser.parse_args()
    metrics = compute_metrics_from_nbest(args.searchqa_dir, args.split, args.nbest_predictions)
    print(metrics)


"""

python compute_metrics.py \
    --searchqa_dir /raid/saam01/data/SearchQA \
    --split train \
    --nbest_predictions outputs/squadv1.1_pretraining/predictions/train/nbest_predictions_.json  

Train: {'f1': 94.71830315531223, 'em': 93.02218392416985}
Dev: {'f1': 81.73607571593188, 'em': 77.56424098466854}
Test: {'f1': 80.09105171242942, 'em': 75.78449003560024}

Test ACL'18 Joint CE+RL: {'f1': 44.7, 'em': 51.2}

SQuADv2.0

Dev: {'f1': 29.061644732266956, 'em': 26.293817030159072}
Test: {'f1': 29.047236416138944, 'em': 26.30381326384556}

Dev: {'f1': 62.841700834575406, 'em': 57.87806809184482} (ignoring where no preds)

Dev: {'f1': 58.785239569088134, 'em': 54.82617145324984} (when predicting as if SQuAD v1.1)
Dev: {'f1': 58.785239569088134, 'em': 54.82617145324984} (same even when we take only non-empty strings)

"""