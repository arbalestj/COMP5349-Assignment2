import numpy as np
import nltk
import nltk.data


def countthelen(record):
    mzz = record.split("\t")
    return len(mzz)


def haha(record):
    mzz = record.split("\t")
    user_id = mzz[1]


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def KillShortReviews(record):
    mzz = record.split("\t")
    review_body = mzz[13]
    sentence_list = splitSentence(review_body)
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


def split_reviews(record):
    mzz = record.split("\t")
    mzz[13] = splitSentence(mzz[13])
    return mzz


def customers_with_sentence_num(record):
    mzz = record.split("\t")
    user_id = mzz[1]
    sentence_num = [len(splitSentence(record[13]))]
    return (user_id, sentence_num)


def products_with_sentence_num(record):
    mzz = record.split("\t")
    products_id = mzz[3]
    sentence_num = [len(splitSentence(record[13]))]
    return (products_id, sentence_num)
