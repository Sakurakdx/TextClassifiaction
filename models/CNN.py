# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
sentences = ["我 爱 你 你", "他 爱 我 我", "她 爱 球 球", "我 恨 你 你", "对 不 起 啊", "这 不 好 好"]
labels = [1, 1, 1, 0, 0, 0]

# TextCNN一些参数
embedding_size = 2
sentence_length = len(sentences[0])  # 句子长度，统一为3
num_class = len(set(labels))  # 标签数量，去重之后标签数，为3
batch_size = 3

word_list = " ".join(sentences).split()  # 分词
print(type(word_list))
vocab = list(set(word_list))  # 构建词表
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)


# 数据预处理
def make_data(x, y, length):
    input = [0 for i in range(0, length)]
    inputs = []
    for sen in x:
        inputs.append([word2idx[n] for n in sen.split()])  # 将单词转换成词表里的映射
        inputs.append(input)


    target = []
    for out in y:
        target.append(out)  # To using Torch Softmax Loss function
    return inputs, target


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()  # 执行父类的构造函数，使得我们能够调用父类的属性。
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (2, embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        # fc
        self.fc = nn.Linear(output_channel, num_class)

    def forward(self, X):
        """
        X:[batch_size, sequence_length]
        :param X:
        :return:
        """
        batch_size = X.shape[0]  # 动态更新，防止验证的时候报错
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel*1*1]
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return output


# train
model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(500):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Test
test_text = '我 恨 你'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
# Predict
model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text, "0")
else:
    print(test_text, "1")
