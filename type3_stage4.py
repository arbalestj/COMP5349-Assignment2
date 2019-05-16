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
    print(tfidf.collect())

    tfidf_T = tfidf.map(lambda x: (x[1], x[0], x[2]))
    # print(tfidf_T.collect())

    dot_product_matrix = CoordinateMatrix(tfidf_T)
    #print(dot_product_matrix.toBlockMatrix().toLocalMatrix())
    cosine_similarity = dot_product_matrix.toIndexedRowMatrix().columnSimilarities()  # .toBlockMatrix().toLocalMatrix().toArray()
    print(type(cosine_similarity))
    #print(cosine_similarity.entries.collect())
    print(cosine_similarity.numCols(), cosine_similarity.numRows())
    #print(cosine_similarity.toBlockMatrix().toLocalMatrix())
    # print(cosine_similarity.entries.collect())
    #NP_cos_sim = cosine_similarity.toBlockMatrix().toLocalMatrix().toArray().copy()

    sum_of_similarity = cosine_similarity.entries \
    .filter(lambda x: x.value > 1e-6) \
    .map(lambda x: (x.i, x.j, 1 - x.value)) \
    .map(lambda x: (0, x[2])) \
    .reduceByKey(lambda x, y: x + y) \
    .collect()
    print(sum_of_similarity)
    the_1 = (cosine_similarity.numRows()*cosine_similarity.numCols() - cosine_similarity.numRows())/2 - cosine_similarity.entries.count()
    final = (sum_of_similarity[0][1]+the_1)*2/(num_Postive_Sentence*(num_Postive_Sentence-1))
    print(final)
    end = time.time()
    time_spent = end - start
    f = open("Stage4.txt","w")
    f.write("time_spent: "+str(time_spent)+"\n")
    f.write("the final is:" + str(final) +"\n")
    f.close()

