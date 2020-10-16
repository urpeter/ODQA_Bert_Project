from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
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
import re

# tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
# return_dict=True)
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")


def add_end_idx(answ_cont_dict):
    # print("def add_end_idx(answ_cont_dict) ...")
    idx_answ_cont_dict = dict()

    answers_list = list()

    print("answ_cont_dict ", len(answ_cont_dict))
    for key, value in answ_cont_dict.items():
        answer = value["answer"]
        context = value["contexts"]

        for c in context:
            index = [(m.start(0), m.end(0)) for m in re.finditer(re.escape(answer), re.escape(c.lower()))]

            if index == []:
                start_idx = None
                end_idx = None
            else:
                start_idx = index[0][0]
                end_idx = index[0][1]

            idx_answ_cont_dict[key] = (answer, start_idx, end_idx)

        for q_id, answer in idx_answ_cont_dict.items():
            answers_list.append({'text': answer[0], 'answer_start': answer[1], 'answer_end': answer[2]})

    return answers_list


def add_token_positions(encodings, answers):
    # print("def add_token_positions(encodings, answers) ...")
    start_positions = []
    end_positions = []
    print("encondings: ", len(encodings))
    print("answers: ", len(answers))
    for i in range(len(answers)):
        if answers[i]['answer_start'] is None:
            start_positions.append(encodings.char_to_token(i, 0))
            end_positions.append(encodings.char_to_token(i, 0))
            # if None, the answer passage has been truncated
        else:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    # return encodings


def create_context_and_qustions_lists(data_to_lists_dict):
    # print("def create_context_and_qustions_lists(data_to_lists_dict) ...")
    context_list = list()
    question_list = list()

    for q_id, data in data_to_lists_dict.items():
        context_list.extend(data["contexts"])
        question_list.extend([data["question"]] * len(data["contexts"]))

    return context_list, question_list


def create_encodings(data_dict, answer_list):
    # print("def create_encodings(data_dict, answer_list) ...")
    context_list, question_list = create_context_and_qustions_lists(data_dict)
    # print("context: " + str(context_list[:10]))
    # print("question: " + str(question_list[:10]))

    encodings = tokenizer(context_list, question_list, padding=True, truncation=True)
    add_token_positions(encodings, answer_list)
    return encodings


def process_searchqa(folder, set_type): # TODO: check if data is properly processed !!
    print("def process_searchqa(folder, set_type) ...")

    counter = 0

    question_id_list = list()
    data_dict = dict()
    batches_data = list()

    file_path = Path("/".join([folder, 'searchqa', set_type]))

    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), "r") as f:

            for line in f:
                json_data = json.loads(line.replace(r" \n", " "))
                question_id = json_data["id"]
                question_id_list.append(question_id)
                data_dict[question_id] = {"answer": json_data["answer"]}
                data_dict[question_id].update({"question": json_data["question"]})

                # List of contexts with retrieval scores, contexts are sorted from highest to lowest score
                answer_contexts = [c["snippet"] for c in json_data["search_results"] if c["snippet"] is not None]
                # remove scores of contexts
                data_dict[question_id].update({"contexts": answer_contexts})

                if len(data_dict) == 30:
                    # add information where answer in context is
                    sqa_answer_list = add_end_idx(data_dict)

                    # create the batch-encodings
                    batches_data.append(create_encodings(data_dict, sqa_answer_list))
                    data_dict.clear()
                    question_id_list.clear()
                    # if len(batches_data) % 1000 == 0:
                    print("\n length batches_data " + str(len(batches_data)) + " " + str(counter))

                if len(batches_data) == 2000:
                    counter += 1
                    # def save_to_file(path, question_dic, type, set_type, doc_size=None):
                    save_batch_files("/local/anasbori/bert_odqa/ODQA_Bert_Project/batch_output", batches_data,
                                     counter)
                    batches_data.clear()

        if len(batches_data) == 2000:
            counter += 1
            save_batch_files(Path("/local/anasbori/bert_odqa/ODQA_Bert_Project/batch_output"), batches_data,
                             counter)
            batches_data.clear()

    counter += 1
    save_batch_files(Path("/local/anasbori/bert_odqa/ODQA_Bert_Project/batch_output"), batches_data, counter)

    # json_data = json.loads(f.read().replace(r" \n", " "))
    # answer_context_dic[json_data["id"]] = {json_data["answer"]: [c["snippet"]
                                                                # for c in json_data["search_results"]
                                                                # if c["snippet"] is not None]}

    # encodings = create_encodings(question_dict, answer_list)
    # New_encodings = add_token_positions(encodings, answer_list)
    # return New_encodings
    # TODO: adjust according to quasar !!!


def process_quasar(folder, set_type, doc_size):
    """
    Processes the quasar t set by mapping question ids to corresponding context snippets.
    :param folder: topmost folder for the dataset
    :param set_type: train, dev or test
    :param doc_size: short or large
    :return:
    """
    print("def process_quasar(folder, set_type, doc_size) ...")

    # create counter for enumeration of batch-files
    counter = 0

    # Question File and Path
    question_file = set_type + "_questions.json"
    question_file_path = Path("/".join([folder, "questions", question_file]))

    # Contexts File and Path
    context_file = set_type + "_contexts.json"
    context_file_path = Path("/".join([folder, "contexts", doc_size, context_file]))

    with open(question_file_path, "r") as qf, open(context_file_path, "r") as cf:
        question_id_list = list()
        data_dict = dict()
        batches_data = list()

        # Parse each line separate to avoid memory issues
        for line in qf:
            parsed_question = json.loads(line)
            question_id = parsed_question["uid"]
            question_id_list.append(question_id)
            data_dict[question_id] = {"answer": parsed_question["answer"]}
            # answer_context_dict[question_id] = [parsed_question["answer"]]
            data_dict[question_id].update({"question": parsed_question["question"]})
            # question_dic[question_id] = question

            # in order to create batches with the size of 30 and to avoid Memory Errors
            if len(data_dict) == 30:
                contexts_counter = 0
                for line2 in cf:
                    parsed_answer = json.loads(line2)
                    # Answer ID should have a corresponding question ID
                    answer_id = parsed_answer["uid"]
                    if answer_id in question_id_list:
                        contexts_counter += 1
                        # List of contexts with retrieval scores, contexts are sorted from highest to lowest score
                        answer_contexts = parsed_answer["contexts"]
                        # remove scores of contexts
                        cleaned_answer_contexts = [ls_elem[1] for ls_elem in answer_contexts]
                        data_dict[answer_id].update({"contexts": cleaned_answer_contexts})
                    if contexts_counter == 30:
                        contexts_counter = 0
                        break

                # add information where answer in context is
                answer_list = add_end_idx(data_dict)

                # create the batch-encodings
                batches_data.append(create_encodings(data_dict, answer_list))
                data_dict.clear()
                question_id_list.clear()
                # if len(batches_data) % 1000 == 0:
                print("\n length batches_data " + str(len(batches_data)) + " " + str(counter))

                if len(batches_data) == 2000:
                    counter += 1
                    # def save_to_file(path, question_dic, type, set_type, doc_size=None):
                    save_batch_files("/local/anasbori/bert_odqa/ODQA_Bert_Project/batch_output", batches_data,
                                     counter)

                    batches_data.clear()

        counter += 1
        save_batch_files(Path("/local/anasbori/bert_odqa/ODQA_Bert_Project/batch_output"), batches_data, counter)


        # todo: determine whether it is computationally more efficient to save a list of tuples instead of a
        # nested list
        # todo: throw an exception if there is no question for a found answer, i.e. the uid is not matching
        # todo: check if the encoding of the contexts is correct, think I saw a "u355" wrongly encoded piece

        # Check if every question has contexts
        # for qid, q in question_dic.items():
        #     assert len(q["contexts"])>0, "Question {} missing context".format(qid)

        # print("Question dic of type <quasar> and set type <{}> has {} entries.".format(set_type, len(question_dic)))
        # return question_dic
        # return encodings2


def save_batch_files(batch_path, batch, counter):
    print("def save_batch_files(batch_path, batch, counter) ...")
    file_name = str("_".join(["batch", args.TYPE, args.SETTYPE, str(counter)]) + ".pkl")
    # Create the output directory if doesn't exist
    path_type = "/".join([str(batch_path), args.SETTYPE])

    if not os.path.exists(path_type):
        os.makedirs(path_type)

    # Write to the file
    full_path = Path("/".join([str(path_type), file_name]))
    with open(str(full_path), "wb") as of:
        pickle.dump(batch, of)
    print("pickled file {} and saved it to {}".format(str(file_name), str(full_path)))


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
    print("def save_to_file(path, question_dic, type, set_type, doc_size=None) ...")

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
    # save_to_file(output_path, test_dic, args.TYPE, args.SETTYPE, args.DOCSIZE)

    # Sample call
    # python3 preprocessing.py -t "searchqa" -f /Users/vanessahahn/Documents/QA/searchQA  -s "test"
