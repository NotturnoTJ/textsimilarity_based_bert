''' 将分别获取两句话的bert输出句向量，再对两个句向量交互 '''

import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertModel, BertTokenizer,BertForSequenceClassification,BertAdam
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd

EPOCH = 3
BATCH_SIZE = 64
DEVICE = torch.device("cuda")
train_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\train_balance.csv')
vld_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\dev_balance.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

qlen = 35  # 单句话包含cls和sep的总长度
class ATEC_Dataset(Dataset):
    def __init__(self, loadin_data):
        self.df = loadin_data
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        '''根据IDX返回数据 '''
        text1 = self.df.iloc[idx, 1]
        text2 = self.df.iloc[idx, 2]
        label = torch.tensor(self.df.iloc[idx, 3])

        tokens_1 = tokenizer.tokenize(text1)
        tokens_2 = tokenizer.tokenize(text2)

        if len(tokens_1) > qlen - 2:
            tokens_1 = tokens_1[:qlen - 2]
        seq_word_1 = ["[CLS]"] + tokens_1 + ["[SEP]"]
        real_len_1 = len(seq_word_1)
        ids_1 = tokenizer.convert_tokens_to_ids(seq_word_1)
        ids_tensor_1 = torch.tensor(ids_1)
        pad0 = torch.zeros(qlen - real_len_1).long()
        ids_tensor_1 = torch.cat([ids_tensor_1, pad0])
        token_type_ids_1 = torch.tensor([0] * qlen).long()
        attention_mask_1 = torch.tensor([1] * (2 + len(tokens_1)) + [0] * (qlen - 2 - len(tokens_1))).long()

        if len(tokens_2) > qlen - 2:
            tokens_2 = tokens_2[:qlen - 2]
        seq_word_2 = ["[CLS]"] + tokens_2 + ["[SEP]"]
        real_len_2 = len(seq_word_2)
        ids_2 = tokenizer.convert_tokens_to_ids(seq_word_2)
        ids_tensor_2 = torch.tensor(ids_2)
        pad0 = torch.zeros(qlen - real_len_2).long()
        ids_tensor_2 = torch.cat([ids_tensor_2, pad0])
        token_type_ids_2 = torch.tensor([0] * qlen).long()
        attention_mask_2 = torch.tensor([1] * (2 + len(tokens_2)) + [0] * (qlen - 2 - len(tokens_2))).long()

        res1 = [ids_tensor_1, token_type_ids_1, attention_mask_1]
        res2 = [ids_tensor_2, token_type_ids_2, attention_mask_2]
        res = [res1, res2, label]

        return res

train_dataset = ATEC_Dataset(train_df)
vld_dataset = ATEC_Dataset(vld_df)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
vld_iter = torch.utils.data.DataLoader(vld_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dp1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(768 * 4, 2)

    def bert_encoder(self, ids_tensor, token_type_ids, attention_mask):
        _, pooled = self.bert(ids_tensor, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              output_all_encoded_layers=False)
        return pooled

    def forward(self, ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2, attention_mask2):
        vec1 = self.bert_encoder(ids_tensor1, token_type_ids1, attention_mask1)
        vec2 = self.bert_encoder(ids_tensor2, token_type_ids2, attention_mask2)
        vec = torch.cat([vec1, vec2, vec1 - vec2, vec1 * vec2], dim=1)

        out = self.dp1(vec)
        out = self.dense2(out)
        out = F.log_softmax(out, dim=1)
        return out

model4 = Model4().to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.00005)
loss_funtion = F.nll_loss

for epoch in range(EPOCH):
    model4.train()
    for i, batchgroup in enumerate(train_iter):
        torch.cuda.empty_cache()
        text1, text2, label = batchgroup
        label = label.to(DEVICE)

        ids_tensor1, token_type_ids1, attention_mask1 = text1[0].to(DEVICE), text1[1].to(DEVICE), text1[2].to(DEVICE)
        ids_tensor2, token_type_ids2, attention_mask2 = text2[0].to(DEVICE), text2[1].to(DEVICE), text2[2].to(DEVICE)

        predicted = model4(ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2,
                                       attention_mask2)

        optimizer.zero_grad()
        loss = loss_funtion(predicted, label)
        loss.backward()
        optimizer.step()

    model4.eval()
    train_correct = 0
    with torch.no_grad():
        for i, batchgroup in enumerate(train_iter):
            torch.cuda.empty_cache()
            text1, text2, label = batchgroup
            label = label.to(DEVICE)
            ids_tensor1, token_type_ids1, attention_mask1 = text1[0].to(DEVICE), text1[1].to(DEVICE), text1[2].to(
                DEVICE)
            ids_tensor2, token_type_ids2, attention_mask2 = text2[0].to(DEVICE), text2[1].to(DEVICE), text2[2].to(
                DEVICE)
            output = model4(ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2,
                                        attention_mask2)
            pred = output.max(1, keepdim=True)[1]
            train_correct += pred.eq(label.view_as(pred)).sum().item()
        print(epoch, 'train_acc:', train_correct / len(train_dataset),)

    vld_correct = 0
    with torch.no_grad():
        for i, batchgroup in enumerate(vld_iter):
            torch.cuda.empty_cache()
            text1, text2, label = batchgroup
            label = label.to(DEVICE)
            ids_tensor1, token_type_ids1, attention_mask1 = text1[0].to(DEVICE), text1[1].to(DEVICE), text1[2].to(
                DEVICE)
            ids_tensor2, token_type_ids2, attention_mask2 = text2[0].to(DEVICE), text2[1].to(DEVICE), text2[2].to(
                DEVICE)
            output = model4(ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2,
                                        attention_mask2)
            pred = output.max(1, keepdim=True)[1]
            vld_correct += pred.eq(label.view_as(pred)).sum().item()
        print('vld_acc:', vld_correct / len(vld_dataset))
        print('\n')