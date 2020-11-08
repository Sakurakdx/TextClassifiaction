# coding:utf-8
import numpy as np
import support
import jieba
from models.CNN import TextCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from support import read_data, make_data, predict, accuracy
from performance import performance

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_file = "vocab.txt"
word2idx = support.read_vocab(vocab_file)
vocab = list(set(word2idx.keys()))  # 词表 [词1， 词2， ...... ， 词n] 词之间互斥

inputs, label = read_data("pre_data/train.txt")
sen_list = support.participle(inputs)  # [["词11", "词12", ...], ["词21", "词22", ...], ..., ["词n1", "词n2", ...]]
sen_length = 41  # 句子平均长度 max为211
print("-" * 40)
print("词表读取完成")

batch_size = 128

# CNN的参数
num_class = 3
vocab_size = len(vocab)  # 词表长度
embedding_size = 100
print("-" * 40)
print("参数设置完成")


def train(model, criterion, optimizer):
    # 数据处理
    input_batch, target_batch = make_data(sen_length, sen_list, word2idx, label)
    input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)
    print("-" * 40)
    print("数据处理完成")

    for epoch in range(500):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model.forward(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        if (epoch + 1) % 10 == 0:
            model_name = "mdl/" + str(epoch + 1) + "_model.mdl"
            torch.save(model.state_dict(), model_name)
            print(model_name + "保存成功")


def main():
    model = TextCNN(vocab_size, embedding_size, num_class).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, criterion, optimizer)

    input_batch, target_batch = make_data(sen_length, sen_list, word2idx, label)
    model_name = "mdl/10_model.mdl"
    pre = predict(model, model_name, input_batch)
    pre_list = pre[:, 0].cpu().numpy()
    pre_list = pre_list.tolist()
    label_list = []
    for i in label:
        label_list.append(int(i))
    micro_F1, macro_F1, ave_acc = performance(label_list, pre_list)
    print("micro_F1:", micro_F1)
    print("macro_F1:", macro_F1)
    print(f"ave_acc:{ave_acc}")

    # train_acc = accuracy(input_batch, target_batch, model, model_name)
    # print("train_acc:", train_acc)


if __name__ == '__main__':
    main()