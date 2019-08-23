# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:08:37 2019

@author: SG0223885
"""

# Start data exploration of the Expedia Data Sets 

import pandas as pd # Start using Pandas for Data Exploration

# Read the CSV Files for the competition

destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Explore the volume of Data in the training and test data sets 
# The program took over 2 Mins 30 Seconds to Print the results the performance is very poor need
# to explore on how I can optimize 

print(train.shape) # Training Data Set has 37670293 rows (37 Million )  and 24 columns 

print(test.shape) # Test Data Set has 2.5 Million Rows and 22 Columns 

print(train.head(5)) # Look at the first 5 rows of the training data set to glean information.

#Date & Time could be useful so we may need to convert it
#What is Feature Engineering : https://en.wikipedia.org/wiki/Feature_engineering
#Feature engineering is hard as all data for country , continent is numbers probably this has been done by expedia to mask the real production and sales data  

# What are we trying to Predict ?
# Our Objective is to predict which hotel cluster a user will book after doing a search on expedia
# How will I know that the prediction is correct ?
# Need to make 5 cluster predictions for each row 
# And will be judged if the correct prediction is made 
# So if corret cluster is 3 and we predict {4,43,60,3,20} our score will be lower if we predict {3,4,43,60,20} 
# Bottomline is we need to order predictions based on the certainity with which the algorithm feels 
# they will be selected !!

print(train["hotel_cluster"].value_counts()) # Distribution of hotels to clusters is fairly even , why is this important ?? I need to explore more ?

#Explore train and test user id's to see if all test user ids are found in train Data Frame

#Create a set of all unique test user id
test_ids = set(test.user_id.unique())
#Create Set of all unique train user id 
train_ids = set(train.user_id.unique()) 
#Figure out the intersection of the 2 sets and see if it matches ??
intersection_count = len(test_ids & train_ids)
if(intersection_count == len(test_ids)):
    print("True") 

# Our hypothesis was correct the user id's in test data set are a subset of training data set so there is no spurious data

# Break the training data set has 37 million records this is unmanageable consider  assign into a manageable chunk via random sampling as it is 
# And break it in 2 parts the test data set and training dataset from within the training data set     

# Differentiate the test and train datasets via year and month 
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month
    