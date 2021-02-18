from transformers.configuration_bert import BertConfig
class ODQA_config(BertConfig):
    def __init__(self):
        super(ODQA_config, self).__init__(return_dict=True)