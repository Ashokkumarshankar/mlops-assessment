from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from config import pretrained_model
class GoEmotionClassifier(nn.Module):
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model):
        super(GoEmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(768, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output

def ret_model(n_train_steps, do_prob):
    model = GoEmotionClassifier(n_train_steps, n_labels, do_prob, bert_model=bert_model)
    return model

tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

bert_model = transformers.SqueezeBertModel.from_pretrained(pretrained_model)


