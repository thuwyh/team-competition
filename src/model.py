from pytorch_pretrained_bert import BertForSequenceClassification
from torch import nn
from collections import OrderedDict
import torch


class PairModel(nn.Module):

    def __init__(self, pretrain_path, dropout=0.1):
        super(PairModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrain_path, cache_dir=None, num_labels=1)
        self.head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('clf', nn.Linear(self.bert.config.hidden_size*2, 1)),
            ])
        )

    def forward(self, input_a, mask_a, input_b, mask_b, token_type_ids=None):
        _, pooled_output_a = self.bert.bert(input_a, token_type_ids, mask_a, output_all_encoded_layers=False)
        _, pooled_output_b = self.bert.bert(input_b, token_type_ids, mask_b, output_all_encoded_layers=False)

        x = torch.cat([pooled_output_a, pooled_output_b], dim=-1)
        out = self.head(x)
        return out
