from Music import Music
from ml_utils import *
import os
import tensorflow as tf
import tensorflow_hub as hub
import time
from datetime import datetime

model_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(model_url)

if __name__ == "__main__":
    start = time.time()
    music = Music()
    select_PID = music.Products.sortBy(lambda x: x[1], ascending=False).take(10)[0][0]
    print(select_PID)

    Positive_Reviews = music.og \
        .filter(lambda x: x.split("\t")[3] == select_PID) \
        .filter(lambda x: int(x.split("\t")[7]) >= 4) \
        .map(lambda x: splitSentence(x.split("\t")[13]))

    print(Positive_Reviews.count())

    Negative_Reviews = music.og \
        .filter(lambda x: x.split("\t")[3] == select_PID) \
        .filter(lambda x: int(x.split("\t")[7]) <= 2) \
        .map(lambda x: splitSentence(x.split("\t")[13]))

    print(Negative_Reviews.count())

    # print(type(np.array(Positive_Reviews.collect())[0]))

    # for i in np.array(Positive_Reviews.collect()):
    #    print(i)
    vector_set = np.array([])
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        index = 0
        for i in np.array(Negative_Reviews.collect()):
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            # print(i)
            message_embeddings = np.array(session.run(embed(i)))
            # print(message_embeddings.shape)
            print(date_time, ": ", "Num: ", index, message_embeddings.shape)
            index += 1
            for i in np.arange(message_embeddings.shape[0]):
                vector_set = np.append(vector_set, message_embeddings[i, :])
    vector_set = vector_set.reshape(-1, 512)

    norm_vector = np.sqrt(np.sum(vector_set ** 2, axis=1))[:, np.newaxis]

    norm_matrix = norm_vector.dot(norm_vector.T)
    dot_product_matrix = vector_set.dot(vector_set.T)
    cosinedistance_matrix = 1 - dot_product_matrix / norm_matrix

    # print(cosinedistance_matrix.shape)
    # print(cosinedistance_matrix)

    distance_for_each = (np.sum(cosinedistance_matrix, axis=1) / cosinedistance_matrix.shape[1])[:, np.newaxis]
    distance_overall = np.mean(distance_for_each)
    # print(distance)
    # print(distance.shape)
    end = time.time()
    time_spent = end - start
    '''
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(Positive_Reviews[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
    '''
    f = open("Stage3.txt", 'w')
    |
    # f.write(str(Positive_Reviews.collect()))
    f.write("total time spent: " + str(time_spent) + "s" + "\n")
    f.write("Average distance between one sentence with others" + "\n" + str(distance_for_each) + "\n")
    f.write("Overall Average distance: " + str(distance_overall))
    f.close()
