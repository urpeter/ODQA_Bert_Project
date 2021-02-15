from transformers.tokenization_bert_fast import BertTokenizerFast

class ODQATokenizerFast(BertTokenizerFast):
    def __call__(self, *args, **kwargs):
        return super().__call__(return_offsets_mapping = True)
