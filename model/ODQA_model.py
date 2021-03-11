import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_callable)
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput)
from transformers.modeling_bert import BertForQuestionAnswering, BERT_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, \
    _TOKENIZER_FOR_DOC, BertModel
from utils.Candid_rep_UA import Candid_rep
from utils.qa_utils import postprocess_qa_predictions
from transformers.data.metrics import squad_metrics

class ODQAModel(BertForQuestionAnswering):

    def __init__(self, config):
        super().__init__(config)

        self.candidate_representation = Candid_rep(k=41)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.examples = None
        self.features = None
        self.init_weights()
        self.wz = nn.Linear(256, 1, bias=False)
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        return_dict=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        print("\n len logits", len(logits), "\n")
        # <- Answer Selection Part

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        start_indexes = squad_metrics._get_best_indexes(start_logits.tolist(), n_best_size=41)
        end_indexes = squad_metrics._get_best_indexes(end_logits.tolist(), n_best_size=41)
        candidate_spans = (start_indexes,end_indexes)
        feat = self.features

        # spans in the original is structured like [passages, number of candidates, span of the answer]
        self.candidate_representation.calculate_candidate_representations(spans=candidate_spans,
                                                                          features=feat,
                                                                          seq_outpu=sequence_output)
        r_Cs = self.candidate_representation.r_Cs  # [200, 100]
        r_Ctilde = self.candidate_representation.tilda_r_Cs  # [200, 100]
        p_C = self.score_answers(r_Ctilde)
        #print("p_C",p_C,"\n")
        value, index = torch.max(p_C, 0)
        print("Value",value,"Index",index.view())
        #encoded_candidates = self.candidate_representation.encoded_candidates
        # take maximum candidate whatever is highest
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_examples_and_features(self, examples, features):
        self.features = features
        self.examples = examples

    def score_answers(self, z_C, pretraining = False):
        s = []
        for c in z_C:
            s.append(self.wz(c)) # wz:(200,100) for us this should be [256]
        s = torch.stack(s, dim=0)
        print("ping")
        if pretraining == True:
            return s.squeeze().unsqueeze(dim=0)
        else:
            return torch.softmax(s, dim=0) #p_C


