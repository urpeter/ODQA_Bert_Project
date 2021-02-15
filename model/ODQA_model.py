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

        self.candidate_representation = Candid_rep(k=82)
        self.examples = None
        self.features = None

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

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        predictions_dict, ODQA_predictions_list = postprocess_qa_predictions(examples=self.examples,
                                                      features=self.features,
                                                      predictions=(start_logits, end_logits),
                                                      version_2_with_negative=True,
                                                      n_best_size=1
                                                      )

        #candidate_spans = predictions_dict[self.examples.qas_id]["start_index"] + \
         #                 predictions_dict[self.examples.qas_id]["end_index"]
        start_indexes = squad_metrics._get_best_indexes(start_logits, n_best_size=1)
        end_indexes = squad_metrics._get_best_indexes(end_logits, n_best_size=1)

        candidate_spans_list = [x['candidate_span'] for x in ODQA_predictions_list]
        start_indices = [x['start_index'] for x in ODQA_predictions_list]
        end_indices = [x['end_index'] for x in ODQA_predictions_list]
        texts = [x['text'] for x in ODQA_predictions_list]

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # spans in the original is structured like [passages, number of candidates, span of the answer]
        candidate_spans = torch.stack(candidate_spans_list, dim = 0)
        self.candidate_representation.calculate_candidate_representations(S_p=S_p, spans=candidate_spans)
        S_Cs = self.candidate_representation.S_Cs  # [200, 100, 200]
        r_Cs = self.candidate_representation.r_Cs  # [200, 100]
        r_Ctilde = self.candidate_representation.tilda_r_Cs  # [200, 100]
        encoded_candidates = self.candidate_representation.encoded_candidates


    def get_examples_and_features(self, examples, features):
        self.features = features
        self.examples = examples


