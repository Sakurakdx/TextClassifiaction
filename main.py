# coding:utf-8
from data_pre import Data_pre
import numpy as np
from support import support
import jieba
from create_list import create_list
from models.CNN import TextCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_txt = "data/hotel_train.txt"
data = Data_pre()
data.read(train_txt)  # 读取数据
data.data_clean()  # 清理数据
data_list = create_list(data.x)  # 构建词表
data_list.cre_list()
word2idx = data_list.word2idx  # 词->id的映射
vocab = data_list.vocab  # 词表 [词1， 词2， ...... ， 词n] 词之间互斥
word_list = data_list.word_list  # 分词后的结果 [词1， 词2， ...... ， 词n] 词之间不互斥
sen_list = data_list.participle()
length_sum = sum(data_list.length)
sen_length = int(length_sum / len(data_list.length)) # 句子平均长度
# sen_length = 41
print("-" * 20)
print("词表构建完成")

batch_size = 128

# CNN的参数
num_class = 3
vocab_size = len(vocab)  # 词表长度
# print(vocab_size)
embedding_size = 40
print("-" * 20)
print("参数设置完成")


# 数据预处理
def make_data(sentence_length, sentence_list, word_idx, label=[]):
    """

    :param sentence_length:
    :param sentence_list:
    :param word_idx:
    :param label:
    :return:
    """
    inputs = []
    for sen in sentence_list:
        input = [word_idx[n] for n in sen]  # 将单词转换成词表里的映射

        len_input = len(input)
        sentence_length = int(sentence_length)

        # input>平均长度，截断；input<平均长度，补0
        if len_input > sentence_length:
            input = input[:sentence_length]
        else:
            input.extend(0 for x in range(0, sen_length - len_input))
        # print(input)
        inputs.append(input)

    target = []
    for out in label:
        target.append(float(out))  # To using Torch Softmax Loss function
    return inputs, target


def train(model, criterion, optimizer):
    # 数据处理
    input_batch, target_batch = make_data(sen_length, sen_list, word2idx, data.y)
    input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)
    print("-" * 20)
    print("数据处理完成")

    for epoch in range(100):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model.forward(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "last_model.mdl")
            print("保存成功")


def test(model, model_file, test_text):
    model.load_state_dict(torch.load(model_file))
    test_vec, target = make_data(sen_length, sen_list, word2idx)
    test_batch = torch.LongTensor(test_vec).to(device)
    model = model.eval()
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "0")
    else:
        print(test_text, "1")


def main():
    model = TextCNN(vocab_size, embedding_size, num_class).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train(model, criterion, optimizer)
    test_txt = ["暑假出行价格难免会水涨船高最好的是酒店打扫房间的大姐天天都来打扫态度特别热情给这个短头发的服务员大姐点赞周边交通挺方便的价格跟同类型相比全是好的了下次来还会选择这里"]
    test(model, model_file="last_model.mdl", test_text=test_txt)


if __name__ == '__main__':
    main()
