#!/usr/bin/env python
# coding: utf-8

# # Tools Installation

# In[1]:


get_ipython().system('pip install scikit-surprise')


# In[1]:


import pandas as pd
import re
import numpy as np
import random
import scipy
import scipy.io
import scipy.sparse as sp


# In[2]:


from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import KNNBasic
from surprise import SlopeOne
from surprise import CoClustering
from surprise import BaselineOnly
from surprise import NMF
from surprise.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as skFold

from implementations import *
from als import *


# # Work On Given Data

# In[ ]:


data = pd.read_csv("Datasets/data_train.csv")
sample = pd.read_csv("Datasets/sample_submission.csv")

seed = 211 # as exercice 10
random.seed = seed


# This is to provide data in the format surprise library wants it to be.

# In[ ]:


cleanedFrame = split3columns(data)
sampleFrame = split3columns(sample)


# In[6]:


reader = Reader(rating_scale=(1, 5))

#Here we call surprise function 

dataCleaned = Dataset.load_from_df(cleanedFrame[['userId', 'movieId', 'rating']], reader)
sampleCleaned = Dataset.load_from_df(sampleFrame[['userId', 'movieId', 'rating']], reader)


# ## Build Trainsets and Testsets

# In[7]:


trainset = dataCleaned.build_full_trainset()
testset = sampleCleaned.build_full_trainset().build_testset()


# Here we need to do that because surprise build_testset()
# modify the order of original testset hence, for ALS algorithm we need to
# reorder it.

# In[18]:


testset_reordered = reorderTestset(testset)
testset_reordered.to_csv("testset_reordered.csv", index = False) 


# # Train Algorithms

# Based on each gridsearch, we apply the same parameters for each algorithms on 
# sample test set to get individual predictions.

# ## SVD

# In[ ]:


#SVD with baselines

algo = SVD()
algo.n_factors = 400
algo.verbose = False
algo.biased = True
algo.reg_all = 0.1
algo.lr_all = 0.01
algo.n_epochs = 500
algo.random_state = seed

print("Training SVD...")
algo.fit(trainset)

print("Computing predictions for SVD... \n")
test_predictions_svd = algo.test(testset) #Get real predictions to append to big final matrix


# In[ ]:


test_predictions_svd = np.asarray(test_predictions_svd)
test_predictions_svd_filtered = test_predictions_svd[:, 3]


# ## SVD Without Baselines

# In[ ]:


#SVD without baselines

algo = SVD()
algo.n_factors = 1
algo.verbose = False
algo.biased = False
algo.reg_all = 0.001 # from gridsearch in other notebook
algo.lr_all = 0.01
algo.n_epochs = 500
algo.random_state = seed

print("Training SVD no baselines...")
algo.fit(trainset)

print("Computing predictions for SVD no baselines...\n")
test_predictions_svd_noB = algo.test(testset) #Get real predictions to append to big final matrix


# In[ ]:


test_predictions_svd_noB = np.asarray(test_predictions_svd_noB)
test_predictions_svd_noB_filtered = test_predictions_svd_noB[:, 3]


# ## SVD++

# In[ ]:


algo = SVDpp()
algo.n_factors = 2
algo.n_epochs = 50
algo.verbose = True
algo.random_state = seed

print("Training SVD++...")
algo.fit(trainset)

print("Computing predictions for SVD++...\n")
test_predictions_svdpp = algo.test(testset)


# In[ ]:


test_predictions_svdpp = np.asarray(test_predictions_svdpp)
test_predictions_svdpp_filtered = test_predictions_svdpp[:, 3]


# ## Slope One

# In[ ]:


#SlopeOne
algo = SlopeOne()

print("Training Slope One...")
algo.fit(trainset)

print("Computing predictions for Slope One...\n")

test_predictions_slope = algo.test(testset)


# In[ ]:


test_predictions_slope = np.asarray(test_predictions_slope)
test_predictions_slope_filtered = test_predictions_slope[:, 3]


# ## KNN Items

# In[ ]:


#KNN

sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }

bsl_options = {'method': 'als',
               'n_epochs': 50
               }

algo = KNNBasic(k=220, sim_options=sim_options, bsl_options=bsl_options)
print("Training KNN Items...")
algo.fit(trainset)

print("Computing predictions for KNN Items...\n")

test_predictions_knn_items = algo.test(testset)


# In[ ]:


test_predictions_knn_items = np.asarray(test_predictions_knn_items)
test_predictions_knn_items_filtered = test_predictions_knn_items[:, 3]


# ## KNN Users

# In[ ]:


#KNN users

sim_options = {'name': 'pearson_baseline',
               'user_based': True  # compute  similarities between users
               }

bsl_options = {'method': 'als',
               'n_epochs': 50
               }

algo = KNNBasic(k=220, sim_options=sim_options, bsl_options=bsl_options)

print("Training KNN Users...")
algo.fit(trainset)

print("Computing predictions for KNN Users...\n")

test_predictions_knn_users = algo.test(testset)


# In[44]:


test_predictions_knn_users = np.asarray(test_predictions_knn_users)
test_predictions_knn_users_filtered = test_predictions_knn_users[:, 3]


# ## Baselines Only

# In[ ]:


#Baselines
algo = BaselineOnly()
print("Training Baselines...")
algo.fit(trainset)

print("Computing predictions for Baselines...\n")
test_predictions_baselines = algo.test(testset)


# In[ ]:


test_predictions_baselines = np.asarray(test_predictions_baselines)
test_predictions_baselines_filtered = test_predictions_baselines[:, 3]


# ## Global Mean

# In[13]:


print("Computing Global Mean...\n")
test_predictions_global_filtered = globalMean(data, len(testset))


# ## User Mean and Movie Mean

# In[14]:


copie_validation = testset.copy()

validation_frame = pd.DataFrame(copie_validation)
validation_frame.columns= ['userId', 'movieId', 'rating']


# In[16]:


print("Computing User mean & Movie mean...\n")
test_predictions_users = userMean(cleanedFrame, validation_frame)
test_predictions_items = itemMean(cleanedFrame, validation_frame)


# ## Matrix Factorization - ALS

# In[ ]:


from helpers import load_data, preprocess_data
path_dataset = "Datasets/data_train.csv"
path_testset = "testset_reordered.csv"

testset = pd.read_csv(path_testset)
ratings = load_data(path_dataset)

ratings.shape


# In[14]:


testFrame = split3columns(testset)


# In[ ]:


print("Computing Matrix Factorization with ALS...\n")
user, item = ALSWithoutTest(ratings)

predictions = item.T.dot(user)
getPredictionsInPlace(testFrame, predictions)
test_predictions_als = testFrame['rating'].values


# # Blending

# ## Stacking Matrices for Regression

# After many submissions, we notices that stacking every algorithms wasn't optimal. Only
# a few was optimal: SVD, SVD++, KNN Users, ALS and Item Mean. Notice that they represent a mix
# of 3 different classes of algorithms: Matrix Factorization, Clustering, Statistics. Blending will
# take the best of each.

# In[ ]:


stacked_test_predictions = np.column_stack((
    test_predictions_svd_filtered,
    test_predictions_svdpp_filtered,
    test_predictions_slope_filtered, 
    test_predictions_knn_items_filtered,
    test_predictions_knn_users_filtered, 
    test_predictions_baselines_filtered, 
    test_predictions_global_filtered,
    test_predictions_users,
    test_predictions_items,
    test_predictions_svd_noB_filtered,
    test_predictions_als
    ))

print("Stacking only best models...\n")
stacked_test_pred_matrix = np.column_stack((
    test_predictions_svd_filtered,
    test_predictions_svdpp_filtered,
    test_predictions_knn_users_filtered, 
    test_predictions_als
    ))


# In[ ]:


#stacked_test_pred = pd.DataFrame(stacked_test_predictions, columns=('Model1', 'Model2','Model3','Model4','Model5','Model6','Model7','Model8', 'Model9', 'Model10'))

#predictions_test.to_csv("stacked_predictions_test_set.csv", index=False)


# In[ ]:


#stacked_test_pred.to_csv("all_models.csv", index = False)


# In[ ]:


#path = "SafeguardModels/all_models_updated_by_tintin.csv"
#stacked_test_pred = pd.read_csv(path)

#stacked_test_pred.head(5)


# In[ ]:


#best_models = stacked_test_pred.drop(columns=['Model6','Model10','Model7','Model3', 'Model4', 'Model9','Model8'])
#stacked_test_pred_matrix = best_models.values

#best_models.head()


# # Feature Expansion

# We apply feature expansion from Scikit.

# In[69]:


from sklearn.preprocessing import PolynomialFeatures


# It would be great if we could do a grid search on the degree but we had not enough time for that.

# In[70]:


print("Feature expansion of degre 2 on data matrix...")

poly = PolynomialFeatures(2, interaction_only=False)
stacked_test_pred_matrix = poly.fit_transform(stacked_test_pred_matrix)

print("done. Here is the new shape: ")
print(stacked_test_pred_matrix.shape)


# Take the weights from the validation set which performed a cross validation of ridge regression
# on a matrix of predictions.

# In[73]:



#take wegight from other notebook (validationset_gridsearch where we have a nice validation set)
weights_opt = np.array([-1.79605641e-01, -7.63823414e-02, 7.09880620e-01, 6.40220837e-01,
-1.89104275e-01, -4.68966506e-02, -6.71453589e-02, 2.24760590e-01,
-2.05275807e-02, -5.50000332e-04, -4.41913407e-02, -1.24416930e-02,
-1.89475553e-01, 7.94738754e-02, 7.06314694e-02])

print("Multiply with weights from validation set...")
targets = stacked_test_pred_matrix.dot(weights_opt)
targets = np.clip(targets, 1, 5)
print("done. \n")
targets = np.asarray(targets)
targets_rounded = np.around(targets.astype(np.double))

print(targets_rounded.shape)


# # Build Final Output Frame

# In[74]:


print("Building submission file.\n")

result_blending = reconstructSampleSubmissionFormat(test_predictions_baselines, targets_rounded)


# In[75]:


result_blending.to_csv("result_blending_expanded.csv", index = False)


# In[ ]:




