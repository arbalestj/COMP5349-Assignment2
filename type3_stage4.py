from Music import Music
from ml_utils import *
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
import time

from datetime import datetime

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_211.jdk/Contents/Home"

memory = '6g'
pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

if __name__ == "__main__":
    start = time.time()
    music = Music()
    select_PID = music.Products.sortBy(lambda x: x[1], ascending=False).take(10)[9][0]
    print(select_PID)

    Positive_Reviews = music.og \
        .filter(lambda x: x.split("\t")[3] == select_PID) \
        .filter(lambda x: int(x.split("\t")[7]) >= 4) \
        .flatMap(lambda x: splitSentence(x.split("\t")[13]))
    num_Postive_Sentence = Positive_Reviews.count()
    print("total num of sentences in positive reviews:", num_Postive_Sentence)

    Negative_Reviews = music.og \
        .filter(lambda x: x.split("\t")[3] == select_PID) \
        .filter(lambda x: int(x.split("\t")[7]) <= 2) \
        .flatMap(lambda x: splitSentence(x.split("\t")[13]))
    print("total num of sentences in negative reviews:", Negative_Reviews.count())

    # Positive_Reviews_collect = Positive_Reviews.collect()

    from pyspark.mllib.feature import HashingTF, IDF

    # Load documents (one per line).

    hashingTF = HashingTF()
    tf = hashingTF.transform(Positive_Reviews)
    # tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    # print("the shape of tfidf matrix, row represents num of sentences, column represents num of attributions",np.array(tfidf.collect()).shape)
    # print(tfidf.collect())
    sb = tfidf.map(lambda x: x.numNonzeros())
    # print("each sentence's num of nonzero element: ", sb.collect())
    print("In total there are ", np.sum(np.array(sb.collect())), "nonzero element in tfidf matrix")

    tfidf = tfidf.zipWithIndex()  # .cache()
    tfidf = tfidf.flatMap(explode).cache()

    # print("tfidf shape after coordinate all the nonzero element: ", np.array(tfidf.collect()).shape)
    # print(tfidf.collect())

    # print(tfidf_T.collect())
    tfidf_matrix = CoordinateMatrix(tfidf).toBlockMatrix().toLocalMatrix().toArray()
    print("the shape of tfidf matrix: ", tfidf_matrix.shape)
    dot_product_matrix = tfidf_matrix.dot(tfidf_matrix.T)
    print("the shape of product of tfidf matrix and its transpose: ", dot_product_matrix.shape)
    # print(dot_product_matrix)
    tfidf_norm = np.sqrt(np.sum(tfidf_matrix ** 2, axis=1))[:, np.newaxis]  # /(num_Postive_Sentence - 1)
    print("the shape of tfidf norm vector: ", tfidf_norm.shape)
    norm_matrix = tfidf_norm.dot(tfidf_norm.T)
    print("the shape of tfidf norm matrix: ", norm_matrix.shape)
    # print(norm_matrix)

    cos_dist = 1 - dot_product_matrix / norm_matrix

    print(np.sum(cos_dist, axis=1))

    average_each = np.sum(cos_dist, axis=1) / (num_Postive_Sentence - 1)
    overall = average_each.mean()

    # print(average_each)
    print(overall)
    end = time.time()
    time_spent = end - start
    f = open("sbmzz.txt", "w")
    f.write("average_each: " + str(average_each))
    f.write("overall: " + str(overall))
    f.close()
