# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_class):
        super(TextCNN, self).__init__()  # 执行父类的构造函数，使得我们能够调用父类的属性。
        self.W = nn.Embedding(vocab_size + 1, embedding_size)

        output_channel = 3
        kenel_size = 3

        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (kenel_size, kenel_size)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        # fc
        self.fc = nn.Linear(output_channel * 20 * 38, num_class)

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
        # print(conved.shape)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return output

# net = TextCNN(126, 40, 3)
# print(net)
