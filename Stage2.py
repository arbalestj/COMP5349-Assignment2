from Music import Music
from ml_utils import *
import os
import time

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_211.jdk/Contents/Home"

# memory = '6g'
# pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
# os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

if __name__ == "__main__":
    start = time.time()
    music = Music()
    Num_of_Reviews = music.Reviews.count()
    print("Num_of_Reviews: ", Num_of_Reviews)

    # filter the reviews whose body contains less than 2 sentences
    Reviews_filter_short = music.og.filter(lambda x: KillShortReviews(x) >= 2)

    '''
    determine the median number of reviews that a customer published
    The main logic is: we sort the customers by the number of reviews they published, then find the median number.
    In step 1 we determine the index of the median:  
    Then we sort the customer and extract the target.
    '''

    '''
    Step1: determine the location of the median
    '''
    Num_of_Customer = music.Customers.count()
    mid = middle(Num_of_Customer)

    '''
    Step2: sort the customers and extract what we want:
    
    1. access the RDD: (Customer_id, Num_of_reviews_this_customer_published), this tuple is treated as a Key-Value Pair,
    where the Customer_id is the Key, the number of reviews this customer published is the Value.
    2. Sort by values
    3. Give index to each Key-Value pair: (Key, Value) ---> ((Key, Value), Index)
    4. Choose the specific median one: filter(lambada x: the_index_of_x is mid)
    Now we got the result
    
    After we apply .collect() to the result, we got: [((Key, Value), Index)], 
    we should access the Value, that is, the num of reviews this customer published,
    So we apply [0][0][1]
    '''

    Median_of_Reviewers = music.Customers \
        .sortBy(lambda x: x[1], ascending=False) \
        .zipWithIndex() \
        .filter(lambda x: x[1] == mid)
    Median_num_of_reviews_a_user_published = Median_of_Reviewers.collect()[0][0][1]

    '''
    For product, it's likewise.
    '''

    Num_of_Products = music.Products.count()
    mid = middle(Num_of_Products)
    Median_of_Products = music.Products \
        .sortBy(lambda x: x[1], ascending=False) \
        .zipWithIndex() \
        .filter(lambda x: x[1] == mid)
    Median_num_of_reviews_a_product_received = Median_of_Products.collect()[0][0][1]

    '''
    Then we select the Customers who published reviews less than the median level
    Then we select the Products which received reviews less than the median level
    '''
    Customers_below_median = music.Customers.filter(lambda x: x[1] <= Median_num_of_reviews_a_user_published)
    Products_below_median = music.Products.filter(lambda x: x[1] <= Median_num_of_reviews_a_product_received)

    '''
    We filter out such Customers and Products
    Note that for each row, there are 15 attributions. The 2th is the customer_id, while the 4th is the Product_id 
    '''
    Filter_Result = Reviews_filter_short \
        .map(lambda x: (x.split("\t")[1], x)) \
        .subtractByKey(Customers_below_median) \
        .map(lambda x: x[1]) \
        .map(lambda x: (x.split("\t")[3], x)) \
        .subtractByKey(Products_below_median) \
        .map(lambda x: x[1])
    print("Filter_Result: ", Filter_Result.count())

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
    end = time.time()

    print(end - start)
