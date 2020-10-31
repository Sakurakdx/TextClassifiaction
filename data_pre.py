# coding:utf-8
import numpy
import pandas
import jieba


class Data_pre(object):
    def __init__(self):
        pass

    # def read(txt):
    #     with open(txt, "r", encoding='utf-8') as file:
    #        lines = [line.split() for line in file]
    #
    #     return lines
    def read(txt):
        """
        读取txt文档，将每一个样例和标签组成一个样本以数组形式存入data
        :return:
        """
        with open(txt, "r", encoding="utf-8") as file:
            data = []
            a = []
            for line in file:
                a.append(line.strip())
                if line.isspace():
                    data.append(a)
                    data[-1].pop(-1)
                    a = []
        return data

    def attr_Chapter(a):
        """

        :return: 返回每个样本的篇章情感属性<-(0|1|2)>的字典
        """
        dic = {'-0': 0, '-1': 0, '-2': 0}
        for i in a:
            dic["-0"] = dic["-0"] + i.count("-0") - i.count('-0-')
            dic["-1"] = dic["-1"] + i.count("-1") - i.count('-1-')
            dic["-2"] = dic["-2"] + i.count("-2") - i.count('-2-')
        return dic

    def attr_eveluate(a):
        """

        :return: 返回每个样本的属性评价<-(0|1|2)->
        """
        dic = {'-0-': 0, '-1-': 0, '-2-': 0}
        for i in a:
            dic["-0-"] = dic["-0-"] + i.count("-0-")
            dic["-1-"] = dic["-1-"] + i.count("-1-")
            dic["-2-"] = dic["-2-"] + i.count("-2-")
        return dic

    def label(a):
        """
        :return:返回每个样本的label
        """
        dic = {"0": 0, "1": 0, "2": 0}
        dic[a[-1]] += 1
        return dic

    def attr_sentiments(a):
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

    def data_clean(a):
        """
        数据清洗，主要包括大小写统一(统一小写)，删除停用词等
        :return:
        """
        for i in a:
            i.lower()

    def participle(a):
        for i in a:
            jieba.cut(i, cut_all=False)
