from Music import Music
from ml_utils import *
import os

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_211.jdk/Contents/Home"

if __name__ == "__main__":
    music = Music()
    # Num_of_Reviews = len(music.Reviews.collect())
    # print(Num_of_Reviews)

    # print(music.Reviews.take(10))
    '''
    lens = music.og.map(countthelen).collect()
    for i in np.arange(len(lens)):
        if lens[i] != 15:
            print(i)
            print(lens[i])
    print(lens)
    '''

    # Reviews_filter_short = music.og.map(split_reviews).filter(lambda x: len(x[13]) >= 2)
    Reviews_filter_short = music.og.filter(lambda x: KillShortReviews(x) >= 2)
    Num_of_Reviews_after_Kill = len(Reviews_filter_short.collect())
    print(Num_of_Reviews_after_Kill)

    Median_of_Reviewers = music.Customers.sortBy(lambda x: x[1], ascending=False) \
        .collect()[middle(len(music.Customers.collect()))][1]
    Median_of_Products = music.Products.sortBy(lambda x: x[1], ascending=False) \
        .collect()[middle(len(music.Products.collect()))][1]

    # print(Median_of_Reviewers)
    # print(Median_of_Products)

    Customers_above_median = music.Customers.filter(lambda x: x[1] >= Median_of_Reviewers).keys().collect()
    Products_above_median = music.Products.filter(lambda x: x[1] >= Median_of_Products).keys().collect()

    print(Customers_above_median)
    print(Products_above_median)

    Filter_Result = Reviews_filter_short \
        .filter(lambda x: x.split("\t")[1] in Customers_above_median) \
        .filter(lambda x: x.split("\t")[3] in Products_above_median)
    print(len(Filter_Result.collect()))
    # print(len(Reviews_filter_users.collect()))

    Top10_Customers = Filter_Result \
        .map(customers_with_sentence_num) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], x[1][middle(len(x[1]))])) \
        .sortBy(lambda x: x[1], ascending=False) \
        .take(10)

    Top10_Products = Filter_Result \
        .map(products_with_sentence_num) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], x[1][middle(len(x[1]))])) \
        .sortBy(lambda x: x[1], ascending=False) \
        .take(10)

    print(Top10_Customers)
    print(Top10_Products)
