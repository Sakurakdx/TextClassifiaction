# coding:utf-8
import numpy
import pandas
import jieba
import re
import emoji
import string
from zhon.hanzi import punctuation  # 中文标点库


class Data_pre(object):
    def __init__(self):
        self.x = []  # 样本x
        self.y = []  # 样本x的标签

    def read(self, txt):
        """
        读取txt文档，将每一个样例和标签组成一个样本以数组形式存入data
        :return:
        """
        with open(txt, "r", encoding="utf-8") as file:
            a = []
            for line in file:
                a.append(line.strip())
                if line.isspace():
                    self.x.append(a[-3])
                    self.y.append(a[-2])
                    a = []

    def clean_sym(self, sen):
        """
        清除一句话里面的<>及其中间的字符及中英文标点
        :param sen:
        :param strat:
        :param end:
        :return:
        """
        # 去除<>
        while 1:
            start = sen.find("<")
            end = sen.find(">")
            if start == -1:
                break
            else:
                sen = sen[:start] + sen[end + 1:]

        # 去除中英文标点
        remove = str.maketrans('', '', string.punctuation)
        sen = sen.translate(remove)
        sen = re.sub("[{}]+".format(punctuation), "", sen)

        return sen

    def clean_emoji(self, sen):
        sen = emoji.demojize(sen)
        while 1:
            start = sen.find(":")
            end = sen.find(":", start + 1)
            if start == end:
                end == sen.find(":", start + 1)
            if start == -1:
                break
            else:
                sen = sen[:start] + sen[end + 1:]
        return sen

    def attr_Chapter(self, a):
        """
        :return: 返回每个样本的篇章情感属性<-(0|1|2)>的字典
        """
        dic = {'-0': 0, '-1': 0, '-2': 0}
        for i in a:
            dic["-0"] = dic["-0"] + i.count("-0") - i.count('-0-')
            dic["-1"] = dic["-1"] + i.count("-1") - i.count('-1-')
            dic["-2"] = dic["-2"] + i.count("-2") - i.count('-2-')
        return dic

    def attr_eveluate(self, a):
        """
        :return: 返回每个样本的属性评价<-(0|1|2)->
        """
        dic = {'-0-': 0, '-1-': 0, '-2-': 0}
        for i in a:
            dic["-0-"] = dic["-0-"] + i.count("-0-")
            dic["-1-"] = dic["-1-"] + i.count("-1-")
            dic["-2-"] = dic["-2-"] + i.count("-2-")
        return dic

    def label(self, a):
        """
        :return:返回每个样本的label
        """
        dic = {"0": 0, "1": 0, "2": 0}
        dic[a[-1]] += 1
        return dic

    def attr_sentiments(self, a):
        """

        :return: 返回每个样本的意见解释<exp-fac|rea|sug|con>
        """
        dic = {'fac': 0, 'rea': 0, 'sug': 0, 'con': 0}
        for i in a:
            dic["fac"] = dic["fac"] + i.count("fac")
            dic["rea"] = dic["rea"] + i.count("rea")
            dic["sug"] = dic["sug"] + i.count("sug")
            dic["con"] = dic["con"] + i.count("con")
        return dic

    def data_clean(self):
        """
        数据清洗，主要包括大小写统一(统一小写)，删除停用词等
        :return:
        """
        x = []
        for sen in self.x:
            sen.lower()
            sen = self.clean_sym(sen)
            sen = self.clean_emoji(sen)
            x.append(sen)

        self.x = x
        print("Successfully clean")

    def write(self, txt="1.txt"):
        with open(txt, "a", encoding="utf-8") as f:
            for x in self.x:
                index = self.x.index(x)
                y = self.y[index]
                line = y + " " + x + "\n"
                f.write(line)

        print("Successfully write")
