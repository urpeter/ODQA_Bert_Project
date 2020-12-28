# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import collections

Context = collections.namedtuple("Context", ["text", "matches"])


class SearchQAInstance:
    def __init__(self, counter, cont_path, uid, question, answer=None, contexts=None, split=None):
        self.counter = counter
        self.cont_path = cont_path
        self.id = uid
        self.q = question
        self.a = answer
        self.c = contexts
        self.type = split

    @classmethod
    def init(cls, line_counter, cont_path, processed_line, split, uid=None):

        question = processed_line["question"]
        answer = processed_line["answer"]
        contexts = list()

        with open(cont_path, "r") as cf:
            # second part
            for line2 in cf:
                parsed_context = json.loads(line2)
                # Answer ID should have a corresponding question ID
                context_id = parsed_context["uid"]
                if context_id == uid:
                    # List of contexts with retrieval scores, contexts are sorted from highest to lowest score
                    p_contexts = parsed_context["contexts"]
                    # remove scores of contexts
                    cleaned_answer_contexts = [ls_elem[1] for ls_elem in p_contexts]

                    for one_context in cleaned_answer_contexts:
                        answers_matches_in_context = search_string(one_context, answer)
                        contexts.append(Context(one_context, answers_matches_in_context))

        return cls(line_counter, cont_path, uid, question, answer, contexts, split)

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

    def __init__(self, data_dir_quest, data_dir_cont, split):
        self.data_dir_quest = data_dir_quest
        self.data_dir_cont = data_dir_cont
        self.split = split

    def __iter__(self):
        """split_file_path = os.path.join(self.data_dir, self.split + ".txt")
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
                        uid += 1"""

        # my old code
        counter = 0
        quest_split_file_path = os.path.join(self.data_dir_quest, self.split + "_questions.json")
        cont_split_file_path = os.path.join(self.data_dir_cont, self.split + "_contexts.json")
        with open(quest_split_file_path, "r") as qf:

            # Parse each line separate to avoid memory issues
            for line in qf:
                counter += 1
                parsed_question = json.loads(line)
                question_id = parsed_question["uid"]
                if not line:
                    continue
                else:
                    inst = SearchQAInstance.init(counter, cont_split_file_path, parsed_question, self.split,
                                                 question_id)
                    # print(inst)
                yield inst

    def check_squad_example(self, example):
        q = example["qas"][0]["question"]
        assert (isinstance(q, str) and len(q) > 0)
        for a in example["qas"][0]["answers"]:
            assert (isinstance(a["text"], str) and len(a["text"]) > 0)
        c = example["context"]
        assert (isinstance(c, str) and len(c) > 0)

    def to_squad(self, version="1.1"):
        # The if version 2.0 is passed, an additional `is_impossible` field will be added
        as_squad = squad_template(version, "quasar")
        no_ans = 0
        total = 0
        for sqa_instance in self:
            if self.split == "train":
                # print(sqa_instance.counter)
                if sqa_instance.counter == 2500:
                    break
                # Original SearchQA paper considers at most 50 contexts and ignore
                # data points which contain less than 41 contexts for training.
                # contexts = sqa_instance.c[:50]
                # if len(contexts) < 41:
                contexts = sqa_instance.c[:41]
                print(len(contexts))
                if len(contexts) < 41:
                    continue
            else:
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
                            # Note how id is assigned, SQAUD_CONTEXTID, so one
                            # can aggregate SQuAD like predictions afterwards.
                            "id": "{}_{}".format(sqa_instance.id, idx),
                            "answers": matches
                        }
                    ],
                    "context": context.text
                }
                # self.check_squad_example(squad_example)
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


# TODO adapt to question- and context-dir !!!
def convert_searchqa_to_squad(quasar_dir_quest, quasar_dir_cont, output_dir, version):
    cond = True

    for split in ("dev", "test"):
    # for split in ("train", "dev", "test"):
    # for split in ("train", "dev"):
        sqa = SearchQA(quasar_dir_quest, quasar_dir_cont, split)
        squad_like_split = sqa.to_squad(version)

        output_file_path = os.path.join(output_dir, "{}.json".format(split))
        with open(output_file_path, "w", encoding="utf-8", errors="ignore") as wf:
            json.dump(squad_like_split, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--quasar_dir_quest",
        default=None, type=str, required=False,
        help="TODO"
    )
    parser.add_argument(
        "--quasar_dir_cont",
        default=None, type=str, required=False,
        help="TODO"
    )
    parser.add_argument(
        "--output_dir",
        default=None, type=str, required=True,
        help="Output directory where SQuAD-like data will be created."
    )
    parser.add_argument(
        "--squad_version",
        default="2.0", type=str,
        help="Whether to convert as SQuAD version 1.1 or 2.0"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    convert_searchqa_to_squad(args.quasar_dir_quest, args.quasar_dir_cont, args.output_dir, args.squad_version)
