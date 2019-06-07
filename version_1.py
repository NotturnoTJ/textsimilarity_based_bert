''' 将两句组合成cls+句1+sep+句2+sep输入bert，取cls位置输出分类 '''

import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertModel, BertTokenizer,BertForSequenceClassification,BertAdam
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd

DEVICE = torch.device("cuda")
EPOCH = 3
BATCH_SIZE = 16
# tokenizer = BertTokenizer.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\train_balance.csv')
vld_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\dev_balance.csv')

qlen = 100  # 两句话包含cls和sep的总长度
class ATEC_Dataset(Dataset):  # 自定义数据处理类
    def __init__(self, loadin_data):
        self.df = loadin_data
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text1 = self.df.iloc[idx, 1]
        text2 = self.df.iloc[idx, 2]
        label = torch.tensor(self.df.iloc[idx, 3])

        tokens_1 = tokenizer.tokenize(text1)
        tokens_2 = tokenizer.tokenize(text2)
        if len(tokens_1) + len(tokens_2) > qlen-3:
            tokens_2 = tokens_2[:(qlen-3 - len(tokens_1))]
        seq_word = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
        real_len = len(seq_word)
        ids = tokenizer.convert_tokens_to_ids(seq_word)

        # 构成输入进bert的3个参数ids_tensor、token_type_ids、attention_mask
        ids_tensor = torch.tensor(ids)
        if real_len < qlen:
            pad0 = torch.zeros(qlen - real_len, ).long()
            ids_tensor = torch.cat([ids_tensor, pad0])

        token_type_ids = torch.tensor([0] * (len(tokens_1) + 2) + [1] * (len(tokens_2) + 1) + [0] * (
                    qlen - 3 - len(tokens_2) - len(tokens_1))).long()
        attention_mask = torch.tensor(
            [1] * (3 + len(tokens_1) + len(tokens_2)) + [0] * (qlen - 3 - len(tokens_1) - len(tokens_2))).long()

        return [ids_tensor, token_type_ids, attention_mask, label]  # __getitem__的返回值要是tensor或者list或者数值类型


train_dataset = ATEC_Dataset(train_df)
vld_dataset = ATEC_Dataset(vld_df)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 欠采样形成train_balance.csv的时候已经对数据shuffle
vld_iter = torch.utils.data.DataLoader(vld_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        # 加载bert模型
        # self.bert = BertModel.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12\bert-base-chinese.tar')
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dp2 = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 2)

    def forward(self, ids_tensor, token_type_ids, attention_mask):
        _, pooled = self.bert(ids_tensor, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              output_all_encoded_layers=False)
        out = self.dense(pooled)
        out = F.softmax(out, dim=1)
        return out

class FocalLoss(nn.Module): #   可以不用看这个类  定义focal loss
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs
        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

model4 = Model4().to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.00005)
loss_funtion = FocalLoss(2)

for epoch in range(EPOCH):
    model4.train()
    for i, batchgroup in enumerate(train_iter):
        torch.cuda.empty_cache()
        ids_tensor, token_type_ids, attention_mask, label = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), \
                                                            batchgroup[2].to(DEVICE), batchgroup[3].to(DEVICE)
        predicted = model4(ids_tensor, token_type_ids, attention_mask)
        optimizer.zero_grad()
        loss = loss_funtion(predicted, label)
        loss.backward()
        optimizer.step()

    model4.eval()
    train_correct = 0
    # train_loss = 0
    with torch.no_grad():
        for i, batchgroup in enumerate(train_iter):
            torch.cuda.empty_cache()
            ids_tensor, token_type_ids, attention_mask, label = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), \
                                                                batchgroup[2].to(DEVICE), batchgroup[3].to(DEVICE)
            output = model4(ids_tensor, token_type_ids, attention_mask)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            # train_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
            train_correct += pred.eq(label.view_as(pred)).sum().item()
        print('train_acc:', train_correct / len(train_dataset))  # ,'\t','tarin_loss:',train_loss/len(train_dataset))
    vld_correct = 0
    # val_loss = 0
    with torch.no_grad():
        for i, batchgroup in enumerate(vld_iter):
            torch.cuda.empty_cache()
            ids_tensor, token_type_ids, attention_mask, label = batchgroup[0].to(DEVICE), batchgroup[1].to(DEVICE), \
                                                                batchgroup[2].to(DEVICE), batchgroup[3].to(DEVICE)
            output = model4(ids_tensor, token_type_ids, attention_mask)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            #             val_loss += F.nll_loss(output, batchgroup.label, reduction='sum').item() # 将一批的损失相加
            vld_correct += pred.eq(label.view_as(pred)).sum().item()
        print('vld_acc:', vld_correct / len(vld_dataset))  # ,'\t','val_loss:',val_loss/len(vld_dataset))
        print('\n')