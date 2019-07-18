import csv
import pandas as pd
import time

import Recommenders as Recommender
import Evaluation as Eval
from sklearn.cross_validation import train_test_split
import scipy
df = pd.read_csv("Reviews.csv")
#print(df.head(5))
#print(len(df))

df1=df.head(50000)
#df1=df
#print(df1.head(5))

"""

Product_grouped = df1.groupby(['ProductId']).agg({'Score': 'count'}).reset_index()
grouped_sum = Product_grouped['Score'].sum()

Product_grouped.sort_values(['Score', 'ProductId'], ascending = [0,1])
"""
users = df1['UserId'].unique()


train_data, test_data = train_test_split(df1, test_size = 0.20, random_state=0)
#print(train_data.head(5))
#print(test_data.head(5))
"""
pm = Recommender.popularity_recommender_py()
pm.create(train_data, 'Score', 'ProductId')
UserId = users[5]
pm.recommend(UserId)
"""


is_model = Recommender.item_similarity_recommender_py()
is_model.create(train_data,'UserId','ProductId')  #module 1
UserId=users[19603]


user_items = is_model.get_user_items(UserId) #module 2

#print(user_items)

print("------------------------------------------------------------------------------------")
print("Training data Products for the user userid: %s:" % UserId)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend products for the user using personalized model
print(is_model.recommend(UserId)) #module 3

"""
start = time.time()

#Define what percentage of users to use for precision recall calculation
user_sample = 0.05

#Instantiate the precision_recall_calculator class
pr = Eval.precision_recall_calculator(test_data, train_data,  is_model)

#Call method to calculate precision and recall values
(ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)

end = time.time()
print(end - start)

"""
