from data_pre import Data_pre


class support(object):
    def __init__(self):
        pass

    def sum_dict(dica, dicb):
        """
        实现两个字典相加
        :param dictb:
        :return:
        """
        for key in dica:
            if dicb.get(key):
                dica[key] = dica[key] + dicb[key]
        return dica

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
