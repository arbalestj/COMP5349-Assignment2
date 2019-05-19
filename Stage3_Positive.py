from Music import Music
from ml_utils import *
import os
import tensorflow as tf
import tensorflow_hub as hub
import time

np.set_printoptions(threshold=np.inf)
model_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(model_url)

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
        .map(lambda x: (x.split("\t")[2], splitSentence(x.split("\t")[13]))) \
        .flatMap(Split) \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0]))

    Positive_Reviews_body = Positive_Reviews.map(lambda x: x[1][0])
    print(Positive_Reviews.count())

    Positive_Reviews_collect = Positive_Reviews_body.collect()

    Positive_Sentence_Matrix = np.array([])
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = np.array(session.run(embed(Positive_Reviews_collect)))
        print(message_embeddings.shape)
        Positive_Sentence_Matrix = message_embeddings.copy()

    print(Positive_Sentence_Matrix.shape)
    norm_vector = np.sqrt(np.sum(Positive_Sentence_Matrix ** 2, axis=1))[:, np.newaxis]
    norm_matrix = norm_vector.dot(norm_vector.T)
    dot_product_matrix = Positive_Sentence_Matrix.dot(Positive_Sentence_Matrix.T)
    cosinedistance_matrix = 1 - dot_product_matrix / norm_matrix

    distance_for_each = (np.sum(cosinedistance_matrix, axis=1) / (cosinedistance_matrix.shape[1] - 1))
    center_index = np.argmin(distance_for_each)
    Closet10_Index = np.argsort(cosinedistance_matrix[center_index, :])[:11]
    distance_overall = np.mean(distance_for_each)
    end = time.time()
    time_spent = end - start
    f = open("Stage3_Positive.txt", 'w')
    # f.write(str(Positive_Reviews.collect()))
    f.write("total time spent: " + str(time_spent) + "s" + "\n")
    # f.write("the distance matrix:" + str(cosinedistance_matrix) + "\n")
    f.write("the index of central sentence: " + str(center_index) + "\n")
    f.write("the center sentence for positive reviews: " + str(Positive_Reviews.lookup(center_index)) + "\n")
    f.write("the 10 closest sentences for positive central sentence: " + "\n")
    for i in Closet10_Index:
        f.write(str(Positive_Reviews.lookup(i)) + "\n")
    f.write("Average distance between one sentence with others" + "\n" + str(distance_for_each) + "\n")
    f.write("Overall Average distance: " + str(distance_overall))
    f.close()
