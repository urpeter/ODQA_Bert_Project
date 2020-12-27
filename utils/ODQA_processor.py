from transformers.data.processors import squad
from tqdm import tqdm

class OdqaProcessor(squad.SquadV2Processor):
    # Creates a dict of example lists per question.
    # Each key is a question with a list of the examples corresponding to the question as the value
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples_dict = {}
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                if paragraph["qas"]["question"] not in examples_dict.keys():
                    examples_dict[paragraph["qas"]["question"]] = []

                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = squad.SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples_dict[paragraph["qas"]["question"]].append(example)
        return examples_dict