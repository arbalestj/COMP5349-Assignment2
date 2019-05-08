from Music import Music
from ml_utils import *
import os

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_211.jdk/Contents/Home"

memory = '6g'
pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

if __name__ == "__main__":
    music = Music()
    Num_of_Reviews = music.Reviews.count()
    print("Num_of_Reviews: ", Num_of_Reviews)

    # Num_of_Customers = music.Customers.count()
    # print(Num_of_Customers)
    # print(middle(Num_of_Customers))

    # print(music.Reviews.take(10))
    '''
    lens = music.og.map(countthelen).collect()
    for i in np.arange(len(lens)):
        if lens[i] != 15:
            print(i)
            print(lens[i])
    print(lens)
    '''

    Reviews_filter_short = music.og.filter(lambda x: KillShortReviews(x) >= 2)
    # Num_of_ReviewsAfterKill = Reviews_filter_short.count()
    # print(Num_of_ReviewsAfterKill)
    # mid = middle(Num_of_ReviewsAfterKill)
    Num_of_Customer = music.Customers.count()
    mid = middle(Num_of_Customer)
    print("Num_of_Customer： ", Num_of_Customer)
    print(mid)
    # print(mid)
    Median_of_Reviewers = music.Customers \
        .sortBy(lambda x: x[1], ascending=False) \
        .zipWithIndex() \
        .filter(lambda x: x[1] == mid)
    Median_num_of_reviews_a_user_published = Median_of_Reviewers.collect()[0][0][1]
    # print(Median_of_Reviewers.collect()[0][0][1])

    Num_of_Products = music.Products.count()
    mid = middle(Num_of_Products)
    print("Num_of_Products", Num_of_Products)
    print(mid)
    # print(mid)
    Median_of_Products = music.Products \
        .sortBy(lambda x: x[1], ascending=False) \
        .zipWithIndex() \
        .filter(lambda x: x[1] == mid)
    Median_num_of_reviews_a_product_received = Median_of_Products.collect()[0][0][1]

    # print(Median_of_Products.collect()[0][0][1])
    # print(Median_of_Reviewers.collect())
    '''
    Median_of_Reviewers = music.Customers.sortBy(lambda x: x[1], ascending=False).zipWithIndex().filter(lambda (index, content): index ==middle(Num_of_Customer) \
        .collect()[middle(len(music.Customers.collect()))][1]
    Median_of_Products = music.Products.sortBy(lambda x: x[1], ascending=False) \
        .collect()[middle(len(music.Products.collect()))][1]

    # print(Median_of_Reviewers)
    # print(Median_of_Products)
    '''
    Customers_below_median = music.Customers.filter(lambda x: x[1] <= Median_num_of_reviews_a_user_published)
    Products_below_median = music.Products.filter(lambda x: x[1] <= Median_num_of_reviews_a_product_received)
    # print(Customers_above_median)
    '''
    print(Customers_above_median)
    print(Products_above_median)
    '''
    Filter_Result = Reviews_filter_short \
        .map(lambda x: (x.split("\t")[1], x)) \
        .subtractByKey(Customers_below_median) \
        .map(lambda x: x[1]) \
        .map(lambda x: (x.split("\t")[3], x)) \
        .subtractByKey(Products_below_median) \
        .map(lambda x: x[1])
    print("Filter_Result: ", Filter_Result.count())
    # .filter(lambda x: Customers_above_median.lookup(x.split("\t")[1]) is not None) \
    # .filter(lambda x: Products_above_median.lookup(x.split("\t")[3]) is not None)
    # print(Filter_Result.count())
    '''
    # print(len(Reviews_filter_users.collect()))
    '''
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