import jieba
from support import read_data


class create_list(object):
    def __init__(self, x):
        self.x = x  # 句子集合
        self.word_list = []  # 分词后的结果
        self.vocab = [0]  # 词表[词1， 词2， ...... ， 词n]  互斥
        self.word2idx = dict()  # 词->id的映射
        self.length = []  # 句子长度

    def cre_list(self):
        """
        创建词表dict
        :param a:
        :return:
        """
        self.participle()
        self.vocab = list(set(self.word_list))
        self.word2idx = {w: (i + 1) for i, w in enumerate(self.vocab)}
        self.count_length()
        print("构建词表成功")

    def participle(self):
        """
        对x内的句子进行分词,改变word_list,返回分词后的句子集合
        :return:a：[["词11", "词12", ...], ["词21", "词22", ...], ..., ["词n1", "词n2", ...]]
        """
        a = []
        for sen in self.x:
            sen_list = jieba.lcut(sen, cut_all=False)
            self.word_list.extend(sen_list)
            a.append(sen_list)

        return a

    def count_length(self):
        """

        :return:
        """
        a = self.participle()  # 分词，a = [["词11", "词12", ...], ["词21", "词22", ...], ..., ["词n1", "词n2", ...]]
        for sen in a:
            length = len(sen)
            self.length.append(length)


read_file = "pre_data/data.txt"
vocab_file = "vocab.txt"
x, y = read_data(read_file)
data_list = create_list(x)
data_list.cre_list()
# sum_sen_length = sum(data_list.length)
# avg = sum_sen_length / len(data_list.x)
# print(avg)
# print(sum_sen_length)
# print(max(data_list.length))
with open(vocab_file, "a", encoding="utf-8") as f:
    for key, value in data_list.word2idx.items():
        f.write(key)
        f.write(" ")
        f.write(str(value))
        f.write("\n")
