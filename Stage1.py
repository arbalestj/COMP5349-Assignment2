from Music import Music
from ml_utils import *
import os

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_211.jdk/Contents/Home"

if __name__ == "__main__":
    music = Music()

    # Stage 1: Overall
    Num_Of_Reviews = len(music.Reviews.collect())
    Num_of_Customers = len(music.Customers.collect())
    Num_of_Products = len(music.Products.collect())

    # Stage 1: user-review distribution
    most_reviews_a_user_create = max(music.Customers.values().collect())

    # Sorted_Customers = music.Customers.map(swap).sortByKey(ascending=False).map(swap)
    Sorted_Customers = music.Customers.sortBy(lambda x: x[1], ascending=False)
    Top10_Reviewers = Sorted_Customers.take(10)
    Median_of_Reviewers = Sorted_Customers.collect()[middle(Num_of_Customers)]

    # Stage 1: product-review distribution
    most_reviews_a_product_has = max(music.Products.values().collect())

    # Sorted_Products = music.Products.map(swap).sortByKey(ascending=False).map(swap)
    Sorted_Products = music.Products.sortBy(lambda x: x[1], ascending=False)
    Top10_Products = Sorted_Products.take(10)
    Median_of_Products = Sorted_Products.collect()[middle(Num_of_Products)]

    f = open("Stage1.txt", "w")
    f.write("Num of Reviews: " + str(Num_Of_Reviews) + "\n")
    f.write("Num of Products: " + str(Num_of_Products) + "\n")
    f.write("Num of Customers: " + str(Num_of_Customers) + "\n")
    f.write("most_reviews_a_user_create: " + str(most_reviews_a_user_create) + "\n")
    f.write("Top10_Reviewers:" + str(Top10_Reviewers) + "\n")
    f.write("Median_of_Reviewers" + str(Median_of_Reviewers) + "\n")
    f.write("most_reviews_a_product_has" + str(most_reviews_a_product_has) + "\n")
    f.write("Top10_Products" + str(Top10_Products) + "\n")
    f.write("Median_of_Products:" + str(Median_of_Products))

    f.close()
