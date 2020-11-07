from data_pre import Data_pre
import jieba
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def read_vocab(vocab_file):
    """
    读取词典，返回词表映射
    :param vocab_file:
    :return: word2idx
    """
    word2idx = dict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        a = f.readlines()
        for line in a:
            word2idx[line.split()[0]] = int(line.split()[1])

    return word2idx


def read_data(read_file):
    """
    返回sen_list和label
    :param read_file:
    :return:
    """
    label = []
    input = []
    with open(read_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        input.append(line.split()[1])
        label.append(line.split()[0])

    return input, label


def participle(inputs):
    """
    对inputs内的句子进行分词,返回分词后的句子集合
    :return:a：[["词11", "词12", ...], ["词21", "词22", ...], ..., ["词n1", "词n2", ...]]
    """
    a = []
    for sen in inputs:
        sen_list = jieba.lcut(sen, cut_all=False)
        a.append(sen_list)
    return a


def make_data(sentence_length, sentence_list, word_idx, label=[]):
    """
    数据预处理，返回input组成的向量组inputs，返回label
    :param sentence_length:
    :param sentence_list:
    :param word_idx:
    :param label:
    :return:
    """
    inputs = []
    for sen in sentence_list:
        input = [word_idx[n] if n in word_idx.keys() else 0 for n in sen]  # 将单词转换成词表里的映射
        len_input = len(input)
        sentence_length = int(sentence_length)

        # input>平均长度，截断；input<平均长度，补0
        if len_input > sentence_length:
            input = input[:sentence_length]
        else:
            input.extend(0 for x in range(0, sentence_length - len_input))
        inputs.append(input)

    target = []
    for out in label:
        target.append(float(out))
    return inputs, target


def accuracy(input_batch, target_batch, model, model_name):
    pre = predict(model, model_name, input_batch)
    count = 0
    for i in enumerate(target_batch):
        # print("i:", i)
        # print(pre[i[0], 0])
        # print(i[0])
        if i[1] == pre[i[0], 0].item():
            count += 1
    acc = count / len(input_batch)
    return acc


def predict(model, model_file, inputs):
    model.load_state_dict(torch.load(model_file))
    test_batch = torch.LongTensor(inputs).to(device)
    model = model.eval()
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    return predict
