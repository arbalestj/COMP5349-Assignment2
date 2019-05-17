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


def customers_products_sennum(record):
    mzz = record.split("\t")
    user_id = mzz[1]
    products_id = mzz[3]
    sentence_num = len(splitSentence(mzz[13]))
    return (user_id, products_id, sentence_num)


def case_sort(x):
    if isinstance(x[1], int) is True:
        return (x[0], np.array([x[1]]))
    else:
        return (x[0], sorted(x[1]))


def sb(x, index):
    result = []
    # print(type(x))

    for i in np.arange(len(x)):
        i = int(i)
        if x[i] != 0:
            result.append((index, i, x[i]))

    return result


def customers_with_sentence_num(record):
    mzz = record.split("\t")
    user_id = mzz[1]
    sentence_num = [len(splitSentence(mzz[13]))]
    return (user_id, sentence_num)


def products_with_sentence_num(record):
    mzz = record.split("\t")
    products_id = mzz[3]
    sentence_num = [len(splitSentence(mzz[13]))]
    return (products_id, sentence_num)


def MatrixProduct_Spark(mat1, mat2):
    mat1 = CoordinateMatrix(mat1)
    mat2 = CoordinateMatrix(mat2)
    mat1 = mat1.entries.map(lambda entry: (entry.j, (entry.i, entry.value)))
    mat2 = mat2.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    matrix_entries = mat1.join(mat2).values().map(lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1])).reduceByKey(
        lambda x, y: x + y).map(lambda x: MatrixEntry(x[0][0], x[0][1], x[1]))
    matrix = CoordinateMatrix(matrix_entries)
    return matrix


def explode(row):
    vec, i = row
    for j, v in zip(vec.indices, vec.values):
        yield i, j, v


def Split(x):
    review_id, sentence_list = x
    for i in sentence_list:
        yield review_id, i

def Combine(x):
    row, col_data = x
    # for i in col_data:
