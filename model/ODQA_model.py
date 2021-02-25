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
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.examples = None
        self.features = None

        self.init_weights()
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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        print("Length of SeqOutputs",len(sequence_output), "\n")
        print("Hidden states Lenge:", type(outputs[2]), "\n")
        print("Hidden states:", outputs[2].size(), "\n")

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
        #print("first line", start_logits[1])
        #print("start_logits",list(enumerate(start_logits)), len(list(enumerate(start_logits))))

        start_indexes = squad_metrics._get_best_indexes(start_logits.tolist(), n_best_size=41)
        end_indexes = squad_metrics._get_best_indexes(end_logits.tolist(), n_best_size=41)
        print("start_indexes", start_indexes)
        print("end_indexes", end_indexes)
        candidate_spans=(start_indexes,end_indexes)
        feat = self.features

        # spans in the original is structured like [passages, number of candidates, span of the answer]
        self.candidate_representation.calculate_candidate_representations(spans=candidate_spans, features=feat) # TODO take care of Sp remains
        S_Cs = self.candidate_representation.S_Cs  # [200, 100, 200]
        r_Cs = self.candidate_representation.r_Cs  # [200, 100]
        r_Ctilde = self.candidate_representation.tilda_r_Cs  # [200, 100]
        encoded_candidates = self.candidate_representation.encoded_candidates

        #predictions_dict, ODQA_predictions_list = postprocess_qa_predictions(examples=self.examples,
         #                                             features=self.features,
          #                                            predictions=(start_logits, end_logits),
           #                                           version_2_with_negative=True,
            #                                          n_best_size=1
             #                                         )

        #candidate_spans = predictions_dict[self.examples.qas_id]["start_index"] + \
         #                 predictions_dict[self.examples.qas_id]["end_index"]


        #candidate_spans_list = [x['candidate_span'] for x in ODQA_predictions_list]
        #start_indices = [x['start_index'] for x in ODQA_predictions_list]
        #end_indices = [x['end_index'] for x in ODQA_predictions_list]
        #texts = [x['text'] for x in ODQA_predictions_list]
        #print("candidate_spans_list",candidate_spans_list,"\n")
        #candidate_spans = torch.stack(candidate_spans_list, dim=0)
        #print("candidate_spans",candidate_spans,"\n")

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


