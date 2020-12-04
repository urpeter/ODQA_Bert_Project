# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import collections


Context = collections.namedtuple("Context", ["text", "matches"])


class SearchQAInstance:
    
    def __init__(self, uid, question, answer=None, contexts=None, split=None):
        self.id = uid
        self.q = question
        self.a = answer
        self.c = contexts
        self.type = split
    
    @classmethod
    def init(cls, processed_line, split, uid=None):
        items = processed_line.split("|||")
        question = items[1].strip()
        answer = items[2].strip()
        contexts = list()
        for match in re.finditer("<s>(.*?)</s>", items[0].strip()):
            context = match.groups()[0].strip()
            if not context:
                continue
            answers_matches_in_context = search_string(context, answer)
            contexts.append(Context(context, answers_matches_in_context))
        return cls(uid, question, answer, contexts, split)
    
    def __repr__(self):
        return "(Q = `{}` ; A = `{}` ; id = `{}` ; type = `{}`)".format(
            self.q, self.a, self.id, self.type
        )
    
    def to_json(self):
        return {
            "id": self.id,
            "question": self.q,
            "answer": self.a,
            "contexts": self.c,
            "type": self.type
        }


class SearchQA:
    
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
    
    def __iter__(self):
        split_file_path = os.path.join(self.data_dir, self.split + ".txt")
        with open(split_file_path, encoding="utf-8", errors="ignore") as rf:
            uid = 0
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                else:
                    inst = SearchQAInstance.init(line, self.split, uid)
                    # Skip over empty question/contexts
                    if not inst.c or not inst.q:
                        continue
                    else:
                        yield inst
                        uid += 1
    
    def check_squad_example(self, example):
        q = example["qas"][0]["question"]
        assert (isinstance(q, str) and len(q) > 0)
        for a in example["qas"][0]["answers"]:
            assert (isinstance(a["text"], str) and len(a["text"]) > 0)
        c = example["context"]
        assert (isinstance(c, str) and len(c) > 0)
    
    def to_squad(self, version="1.1"):
        # The if version 2.0 is passed, an additional `is_impossible` field will be added
        as_squad = squad_template(version, "SearchQA")
        no_ans = 0
        total = 0
        for sqa_instance in self:
            if self.split == "train":
                # FIXME only here for small dataset
                if sqa_instance.id == 2500:
                    break
                # Original SearchQA paper considers at most 50 contexts and ignore
                # data points which contain less than 41 contexts for training.
                contexts = sqa_instance.c[:50]
                if len(contexts) < 41:
                    continue
            else:
                # FIXME only here for small dataset
                if sqa_instance.id == 500:
                    break
                contexts = sqa_instance.c
            
            for idx, context in enumerate(contexts):
                # Search for answer string in context
                matches = context.matches
                if not matches and version != "2.0":
                    continue
                squad_example = {
                    "qas": [
                            {
                              "question": sqa_instance.q,
                              # Note how id is assigned, SQAID_CONTEXTID, so one
                              # can aggregate SQuAD like predictions afterwards.
                              "id": "{}_{}".format(sqa_instance.id, idx), 
                              "answers": matches
                            }
                        ],
                    "context": context.text
                }
                self.check_squad_example(squad_example)
                if version == "2.0":
                    is_impossible = matches == list()
                    squad_example["qas"][0]["is_impossible"] = is_impossible
                    if is_impossible:
                        no_ans += 1
                
                as_squad["data"][0]["paragraphs"].append(squad_example)
                total += 1
        
        print("Total SQuAD-like instances: {}".format(total))
        if version == "2.0":
            print("No. of <q, c> pairs without answers: {}".format(no_ans))
        
        return as_squad


def squad_template(version="1.1", title="SearchQA"):
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


def search_string(text, string):
    matches = list()
    string_re = re.compile(r"%s" % re.escape(string))
    for match in string_re.finditer(text):
        text = match.group(0)
        # Make sure non-empty text field
        if not text.strip():
            continue
        answer_start = match.span()[0]
        matches.append({"text": match.group(0), "answer_start": match.span()[0]})
    return matches


def convert_searchqa_to_squad(searchqa_dir, output_dir, version):
    for split in ("train", "val", "test"):
        sqa = SearchQA(searchqa_dir, split)
        squad_like_split = sqa.to_squad(version)
        output_file_path = os.path.join(output_dir, "{}.json".format(split))
        with open(output_file_path, "w", encoding="utf-8", errors="ignore") as wf:
            json.dump(squad_like_split, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument(
        "--searchqa_dir",
        default=None, type=str, required=True,
        help="SearchQA data directory containing `train.txt`, `val.txt` and `test.txt` obtained "
        "from decompressing the `processed_json2txt.json` folder; It can be downloaded from "
        "https://drive.google.com/drive/u/0/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649"
    )
    parser.add_argument(
        "--output_dir", 
        default=None, type=str, required=True,
        help="Output directory where SQuAD-like data will be created."
    )
    parser.add_argument(
        "--squad_version", 
        default="1.1", type=str,
        help="Whether to convert as SQuAD version 1.1 or 2.0"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    convert_searchqa_to_squad(args.searchqa_dir, args.output_dir, args.squad_version)

"""
Small stats

Total SQuAD-like instances: 124574
No. of <q, c> pairs without answers: 79523
Total SQuAD-like instances: 25406
No. of <q, c> pairs without answers: 16474
Total SQuAD-like instances: 25450
No. of <q, c> pairs without answers: 16725

"""