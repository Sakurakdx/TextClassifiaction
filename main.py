# coding:utf-8
from data_pre import Data_pre
import numpy as np
from support import support
import jieba


def attr_con(txt):
    """
    进行属性统计
    :param txt:
    :return:
    """
    lines = Data_pre.read(txt)
    dic_Chapter = {'-0': 0, '-1': 0, '-2': 0}
    dic_eveluate = {'-0-': 0, '-1-': 0, '-2-': 0}
    dic_label = {"0": 0, "1": 0, "2": 0}
    dic_sentiments = {'fac': 0, 'rea': 0, 'sug': 0, 'con': 0}
    for a in lines:
        dic_Chapter = support.sum_dict(dic_Chapter, Data_pre.attr_Chapter(a))  # 情感属性<-(0|1|2)>
        dic_eveluate = support.sum_dict(dic_eveluate, Data_pre.attr_eveluate(a))  # 属性评价<-(0|1|2)->
        dic_label = support.sum_dict(dic_label, Data_pre.label(a))  # label
        dic_sentiments = support.sum_dict(dic_sentiments, Data_pre.attr_sentiments(a))  # 意见解释<exp-fac|rea|sug|con>

    print(dic_Chapter)
    print(dic_eveluate)
    print(dic_label)
    print(dic_sentiments)


def participle(a):
    for i in a:
        jieba.cut(i, cut_all=False)
        print(i)
    # print(a)


if __name__ == '__main__':
    txt = "test.txt"
    # attr_con(txt)
    lines = Data_pre.read(txt)
    for x in lines:
        participle(x)

    # print(lines)
