
import torch
from transformers import AutoTokenizer, BartForSequenceClassification
import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (

    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel, BartModel, shift_tokens_right, BartClassificationHead
)


class BartForStoryEntailment(BartPretrainedModel):


    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, num_classes, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            num_classes=num_classes,
            pooler_dropout=0.1,
        )
        self.num_classes = num_classes
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        prefix_input_ids: torch.LongTensor = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        suffix_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if prefix_input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )
        model_fn = self.model.encoder

        prefix_inputs_embeds = model_fn.embed_tokens(prefix_input_ids) * model_fn.embed_scale
        if suffix_input_ids is not None:
            assert suffix_inputs_embeds is None
            suffix_inputs_embeds = model_fn.embed_tokens(suffix_input_ids) * model_fn.embed_scale
        else:
            suffix_inputs_embeds = suffix_inputs_embeds * model_fn.embed_scale
        inputs_embeds = torch.cat((prefix_inputs_embeds, suffix_inputs_embeds), dim=1)
        attention_mask = torch.cat((prefix_attention_mask, suffix_attention_mask), dim=1)

        outputs = self.model.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # [bs, s, dim]
        hidden_states = outputs[0]  # last hidden state
        weight = attention_mask / attention_mask.sum(1, keepdim=True)
        sentence_representation = torch.sum(hidden_states * weight.unsqueeze(-1), dim=1)
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output



