from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import os
# Command line interaction
from argparse import ArgumentParser
# Json to parse files
import json
# Pickle to save dictionary
import pickle
# Handle the file paths
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                      return_dict=True)


def process_searchqa(folder, set_type):
    # question_dic = {}
    file_path = Path("/".join([folder, 'train_val_test_json_split', 'data_json', set_type]))
    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), "r") as f:
            json_data = json.loads(f.read().replace(r" \n", " "))
            # question_dic[json_data["id"]] = {"question":json_data["question"], "answer":json_data["answer"],
            # "contexts":[c["snippet"] for c in json_data["search_results"] if c["snippet"] is not None]}
            tokens_list = [tokenizer.encode_plus(json_data["question"], c["snippet"], padding=True, truncation=True,
                                                 return_tensors="pt")
                           for c in json_data["search_results"] if c["snippet"] is not None]
            # TODO: FOR REINFORCEMENT LEARNING ANSWERS NEEDED?!
            # question_dic_answer =
    return tokens_list


def process_quasar(folder, set_type, doc_size):
    """
    Processes the quasar t set by mapping question ids to corresponding context snippets.
    :param folder: topmost folder for the dataset
    :param set_type: train, dev or test
    :param doc_size: short or large
    :return:
    """
    # Question File and Path
    question_dic = list()
    question_file = set_type + "_questions.json"
    question_file_path = Path("/".join([folder, "questions", question_file]))

    # Contexts File and Path
    context_file = set_type + "_contexts.json"
    context_file_path = Path("/".join([folder, "contexts", doc_size, context_file]))

    with open(question_file_path, "r") as qf, open(context_file_path, "r") as cf:
        # Parse each line separate to avoid memory issues

        question_dic = dict()

        for line in qf:
            parsed_question = json.loads(line)
            question = parsed_question["question"]
            # print(question)
            question_id = parsed_question["uid"]
            # question_answer = parsed_question["answer"]
            question_dic[question_id] = [question]

        for line in cf:
            parsed_answer = json.loads(line)
            # Answer ID should have a corresponding question ID
            answer_id = parsed_answer["uid"]
            # List of contexts with retrieval scores, contexts are sorted from highest to lowest score
            answer_contexts = parsed_answer["contexts"]
            # remove scores of contexts
            cleaned_answer_contexts = [ls_elem[1] for ls_elem in answer_contexts]
            # join all contexts to one single string => creates too long tokens for model
            # one_string_contexts = ' '.join(cleaned_answer_contexts)
            question_dic[answer_id].append(cleaned_answer_contexts)

        # for key, value in question_dic.items():
        #    print("key: " + key)
        #    print("value: " + value)

        tokens_list = [tokenizer.encode(value[0], con, max_length=512) for key, value in question_dic.items()
                       for con in value[1]]

        # todo: determine whether it is computationally more efficient to save a list of tuples instead of a
        # nested list
        # todo: throw an exception if there is no question for a found answer, i.e. the uid is not matching
        # todo: check if the encoding of the contexts is correct, think I saw a "u355" wrongly encoded piece

        # Check if every question has contexts
        # for qid, q in question_dic.items():
        #     assert len(q["contexts"])>0, "Question {} missing context".format(qid)

        # print("Question dic of type <quasar> and set type <{}> has {} entries.".format(set_type, len(question_dic)))
        # return question_dic
        return tokens_list


def save_to_file(path, question_dic, type, set_type, doc_size=None):
    """
    Save question dictionary to a file.
    :param path: filepath
    :param type: quasar or searchqa
    :param set_type: train, dev or set
    :param doc_size: only for quasar short or long
    :param question_dic: mapping of question ids to contexts
    :return:
    """
    # Check whether question dic contains values
    assert len(question_dic)>0, "question dic is empty"

    # Create filename
    if type == "quasar":
        filename = "_".join([type, set_type, doc_size]) + ".pkl"
    else:
        filename = "_".join([type, set_type]) + ".pkl"
    full_path_to_file = Path("/".join([str(path), str(filename)]))

    # Create the output directory if doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Write to the file
    with open(full_path_to_file, "wb") as of:
        pickle.dump(question_dic, of)
    print("pickled file {} and saved it to {}".format(filename, full_path_to_file))


def main(type, folder, set_type, doc_size):
    """
    :param type of qa-set: either searchqa or quasar
    :param folder: folderpath
    :param set_type: either train, dev, or test
    :return: dictionary of type {question_id : {question:"", category:"", snippets:[]}}
    """
    print(type, folder, set_type, doc_size)

    if type == "quasar":
        return process_quasar(folder, set_type, doc_size)
    elif type == "searchqa":
        return process_searchqa(folder, set_type)
    # else:
    # A wrong type should be identified by argparse already but this is another safeguard
    return ValueError("type must be either 'quasar' or 'searchqa'")


if __name__ == '__main__':
    parser = ArgumentParser()

    # Specify the arguments the script will take from the command line
    # Type
    # Dest specifies how the attribute is referred to by arparse
    parser.add_argument("-t", "--type", required=True, dest="TYPE",
                        help="Specify type of question answer set. either searchqa or quasar",
                        choices=['searchqa', 'quasar'])

    # Folderpath
    parser.add_argument("-f", "--folder", required=True, dest="FOLDERPATH",
                        help="Specify source folder")
    # Set
    parser.add_argument("-s", "--set", required=True, dest="SETTYPE", help="specify set: either train, dev or test",
                        choices=['train', 'dev', 'test', 'val'])

    # Optional Argument: Size of pseudo documents (only relevant for quasar)
    parser.add_argument("-ds", "--docsize", dest="DOCSIZE", help="specify size of pseudo documents",
                        choices=['long', 'short'], default="short")

    # Return an argparse object by taking the commands from the command line (using sys.argv)
    args = parser.parse_args() # Argparse returns a namespace object

    # Call the main function with with the argparse arguments
    test_dic = main(type=args.TYPE, folder=args.FOLDERPATH, set_type=args.SETTYPE, doc_size=args.DOCSIZE)
    output_path = Path(os.getcwd() + '/outputs')
    save_to_file(output_path, test_dic, args.TYPE, args.SETTYPE, args.DOCSIZE)

    # Sample call
    # python3 preprocessing.py -t "searchqa" -f /Users/vanessahahn/Documents/QA/searchQA  -s "test"
