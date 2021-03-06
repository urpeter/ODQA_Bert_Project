# -*- coding: utf-8 -*-

import collections
import json
import argparse
import os

# from convert_searchqa_to_squad import SearchQA
from transformers.data.metrics.squad_metrics import compute_f1, compute_exact


def compute_metrics_from_nbest(quasar_dir, split, fname_nbest_preds):
    qid2preds = collections.defaultdict(list)

    with open(fname_nbest_preds) as rf:
        preds = json.load(rf)
        for uid in preds:
            # print(uid)
            qid, idx = uid.split("_")
            # print(qid)
            # Take only non-empty predictions
            pred = [p for p in preds[uid] if p["text"]][0]
            # pred = preds[uid][0]  # sorted by probs (we are taking best one here)
            ans, score = pred["text"], pred["probability"]
            qid2preds[qid].append((ans, score, uid))

    preds_qid2ans = dict()

    for qid, ans in qid2preds.items():
        # select best answer from all paragraphs
        # print(sorted(ans, key=lambda x: x[1], reverse=True))
        # print(sorted(ans, key=lambda x: x[1], reverse=True)[0])
        sorted_list = sorted(ans, key=lambda x: x[1], reverse=True)
        # print(sorted_list[0])
        # print(sorted_list[1])
        # ans, _, idx = sorted(ans, key=lambda x: x[1], reverse=True)[0]
        ans, _, idx = sorted_list[0]
        # ans2, _2, idx2 = sorted(ans, key=lambda x: x[1], reverse=True)[1]
        ans2, _2, idx2 = sorted_list[1]
        preds_qid2ans[qid] = (idx, idx2)

    # print(preds_qid2ans)

    # gold_qid2ans = dict()

    # quasar_data = os.path.join(quasar_dir, split + "_questions.json")
    quasar_data = os.path.join(quasar_dir, split + ".json")

    # with open(quasar_data, 'r') as qa_data:
    #     for line in qa_data:
    #         data = json.loads(line)
    #         gold_qid2ans[data['uid']] = data['answer']

    with open(quasar_data) as qa_data:
        as_squad = squad_template()
        data = json.load(qa_data)
        data_list = data['data'][0]['paragraphs']
        print(len(data_list))
        new_data_list = list()
        pred_vals = list(sum(preds_qid2ans.values(), ()))
        print(pred_vals)
        for p0 in data_list:
            if p0['qas'][0]['id'] in pred_vals:
                new_data_list.append(p0)
        print(len(new_data_list))
        as_squad["data"][0]["paragraphs"].extend(new_data_list)

    return as_squad


def create_new_train_file(squad_like_split, output_dir, split):
    output_file_path = os.path.join(output_dir, "{}.json".format(split))
    with open(output_file_path, "w", encoding="utf-8", errors="ignore") as wf:
        json.dump(squad_like_split, wf)


def squad_template(version="2.0", title="quasar-t"):
    template = {
        "version": version,
        "data": [
            {
                "title": title,
                "paragraphs": []
            }
        ]
    }
    return template

    #         try:
    #             # if p0['qas'][0]['id'] in pred_keys:
    #             gold_qid2ans[p0['qas'][0]['id']] = p0['qas'][0]['answers'][0]['text']
    #         except IndexError:
    #             continue

    # print((len(preds_qid2ans.keys())), len(gold_qid2ans.keys()))

    # qid2f1 = dict()
    # qid2em = dict()
    #
    # counter = 0
    # for qid in preds_qid2ans.keys():
    #     try:
    #         a_pred = preds_qid2ans[qid]
    #         a_gold = gold_qid2ans[qid]
    #         # print('pred: ', a_pred, 'gold: ', a_gold)
    #         qid2f1[qid] = compute_f1(a_gold, a_pred)
    #         qid2em[qid] = compute_exact(a_gold, a_pred)
    #     except KeyError:
    #         counter += 1
    #         # print()
    #         continue
    #
    # print(counter)
    #
    # f1 = sum(list(qid2f1.values())) / len(qid2f1)
    # f1 *= 100
    # em = sum(list(qid2em.values())) / len(qid2em)
    # em *= 100
    #
    # metrics = {"f1": f1, "em": em}
    #
    # return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--quasar_dir",
        default=None, type=str, required=True,
        help="Quasar data directory containing converted squad-format data: \n"
             "`train.json`, `val.json` and `test.json`.\n"
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
    parser.add_argument(
        "--out_dir",
        default=None, type=str, required=True,
        help="output path for new dataset"
    )
    args = parser.parse_args()
    # metrics = compute_metrics_from_nbest(args.quasar_dir, args.split, args.nbest_predictions)
    new_squad = compute_metrics_from_nbest(args.quasar_dir, args.split, args.nbest_predictions)
    create_new_train_file(new_squad, args.out_dir, args.split)
    # print(metrics)


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