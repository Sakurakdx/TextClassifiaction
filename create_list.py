import jieba


class create_list(object):
    def __init__(self, x):
        self.x = x  # 句子集合
        self.word_list = []  # 分词后的结果
        self.vocab = []  # 词表[词1， 词2， ...... ， 词n]
        self.word2idx = dict()  # 词->id的映射
        self.vocab_size = 0

    def participle(self):
        """
        对x内的句子进行分词
        :return:
        """
        a = []
        for sen in self.x:
            sen_list = jieba.lcut(sen, cut_all=False)
            a.extend(sen_list)

        return a

    def cre_list(self):
        """
        创建词表dict
        :param a:
        :return:
        """
        self.word_list = self.participle()
        self.vocab = list(set(self.word_list))
        self.word2idx = {w: (i+1) for i, w in enumerate(self.vocab)}
        print("构建词表成功")
