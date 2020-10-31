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