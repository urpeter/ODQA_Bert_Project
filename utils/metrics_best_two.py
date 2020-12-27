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
            # Take only non-empty predictions
            pred = [p for p in preds[uid] if p["text"]][0]
            pred2 = [p for p in preds[uid] if p["text"]][1]
            # pred = preds[uid][0]  # sorted by probs (we are taking best one here)
            ans, start_logit = pred["text"], pred["start_logit"]
            ans2, start_logit2 = pred2["text"], pred2["start_logit"]
            qid2preds[uid].append((ans, start_logit, uid + "_1"))
            qid2preds[uid].append((ans2, start_logit2, uid + "_2"))

    quasar_data = os.path.join(quasar_dir, split + ".json")

    with open(quasar_data) as qa_data:
        counter = 0
        as_squad = squad_template()
        data = json.load(qa_data)
        data_list = data['data'][0]['paragraphs']
        print(len(data_list))
        new_data_list = list()

        for p0 in data_list:
            p0_id = p0['qas'][0]['id']
            pred_vals = qid2preds[p0_id]

            for elem in pred_vals:
                new_data_list.append({'qas': [{'question': p0['qas'][0]['question'], 'id': elem[2],
                                       'answers': [{'text': elem[0], 'answer_start': elem[1]}],
                                       'is_impossible': False}], 'context': p0['context']})

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
        help="nbest_predictions_.json path + file"
    )
    parser.add_argument(
        "--out_dir",
        default=None, type=str, required=True,
        help="output path for new dataset"
    )
    args = parser.parse_args()
    new_squad = compute_metrics_from_nbest(args.quasar_dir, args.split, args.nbest_predictions)
    create_new_train_file(new_squad, args.out_dir, args.split)


"""

python compute_metrics.py \
    --quasar_dir /raid/saam01/data/SearchQA \
    --split dev \
    --nbest_predictions outputs/squadv1.1_pretraining/predictions/train/nbest_predictions_.json  \
    --out_dir /SOMEWHERE/

"""