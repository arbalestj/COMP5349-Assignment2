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
    #print(tfidf.collect())

    tfidf_T = tfidf.map(lambda x: (x[1], x[0], x[2]))
    # print(tfidf_T.collect())

    tfidf_matrix = CoordinateMatrix(tfidf).toBlockMatrix().toLocalMatrix().toArray()
    #print(dot_product_matrix.toBlockMatrix().toLocalMatrix())

    norm_vector = np.sqrt(np.sum(tfidf_matrix ** 2, axis=1))[:, np.newaxis]

    norm_matrix = norm_vector.dot(norm_vector.T)
    dot_product_matrix = tfidf_matrix.dot(tfidf_matrix.T)
    cosinedistance_matrix = 1 - dot_product_matrix / norm_matrix

    # print(cosinedistance_matrix.shape)
    # print(cosinedistance_matrix)

    distance_for_each = (np.sum(cosinedistance_matrix, axis=1) / cosinedistance_matrix.shape[1])[:, np.newaxis]
    distance_overall = np.mean(distance_for_each)
    end = time.time()
    time_spent = end - start
    f= open("Stage4_numpy_v.txt","w")
    f.write("time spent:"+str(time_spent))
    f.write("distance for each:"+str(sorted(distance_for_each)))
    f.close()