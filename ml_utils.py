import numpy as np
import nltk
import nltk.data
from pyspark.mllib.linalg.distributed import *

nltk.data.path.append("/home/hadoop/nltk_data")


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def KillShortReviews(record):
    mzz = record.split("\t")
    review_body = mzz[13]
    sentence_list = splitSentence(review_body)
    result = len(sentence_list)
    if result >= 2:
        pass
        # print(mzz[1], result, sentence_list)
    return len(sentence_list)


def middle(x):
    return int(x / 2) if x % 2 == 1 else int(x / 2) - 1


def customers_products_sennum(record):
    mzz = record.split("\t")
    user_id = mzz[1]
    products_id = mzz[3]
    sentence_num = len(splitSentence(mzz[13]))
    return (user_id, products_id, sentence_num)


def countReview(record):
    mzz = record.split("\t")
    return (mzz[2], 1)


def countCustomer(record):
    # print(type(record))
    mzz = record.split("\t")

    # print(len(mzz))
    return (mzz[1], 1)


def countProduct(record):
    mzz = record.split("\t")
    return (mzz[3], 1)


def case_sort(x):
    if isinstance(x[1], int) is True:
        return (x[0], np.array([x[1]]))
    else:
        return (x[0], sorted(x[1]))


def explode(row):
    vec, i = row
    for j, v in zip(vec.indices, vec.values):
        yield i, j, v


def Split(x):
    review_id, sentence_list = x
    for i in sentence_list:
        yield review_id, i
