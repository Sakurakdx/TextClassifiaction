import jieba


class word_list(object):
    def __init__(self):
        self.word_dict = dict()

    def clean_sym(self, sen):
        """
        清除<>及其中间的字符
        :param sen:
        :param strat:
        :param end:
        :return:
        """
        while 1:
            start = sen.find("<")
            end = sen.find(">")
            if start == -1:
                break
            else:
                new_sen = sen[:start] + sen[end:]

        return new_sen

    def participle(sens):
        """
        对句子进行分词
        :return:
        """
        sen_list = jieba.lcut(sens, cut_all=False)
        return sen_list

    def cre_list(self, a):
        """
        创建词表dict
        :param a:
        :return:
        """
        for sen in a:
            index = a.index(sen)
            new_sen = self.clean_sym(sen)  # 清理<>字符
            sen_list = self.participle(new_sen)  # 分词
            for word in sen_list:
                if word not in self.word_dict:
                    self.word_dict[word] = 1
                else:
                    self.word_dict[word] = self.word_dict[word] + 1
        return self.word_dict

    def jiekou(a):
        dic1 = word_list.cre_list(a)
        return dic1


