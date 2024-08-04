
import torch
import torch.nn as nn
from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from torch.utils.data import Dataset


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassifier(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            label=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]

        if label is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            outputs = (loss,) + outputs

        return outputs


class EntScoreDataset(Dataset):
    def __init__(self, tokenizer, data):
        init_premise_list, coun_premise_list, hypos_list = [], [], []
        maxlen = 0
        self.input_ids, self.attention_mask = [], []
        for js in data:
            # init_premise = js.original_context
            count_premise = js.cf_context
            hypothesis = js.predicted_ending
            # init_premise_list.append(init_premise)
            coun_premise_list.append(count_premise)
            hypos_list.append(hypothesis)
            encodings = tokenizer(count_premise, hypothesis, return_tensors='pt', padding=True)
            input_ids = encodings['input_ids'].numpy().tolist()[0]
            attention_mask = encodings['attention_mask'].numpy().tolist()[0]
            maxlen = max(maxlen, len(input_ids))
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
        maxlen += 2
        self.maxlen = maxlen
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        input_ids = self.input_ids[i]
        attention_mask = self.attention_mask[i]
        while len(input_ids) < self.maxlen:
            input_ids.append(self.pad_id)
            attention_mask.append(0)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, attention_mask

