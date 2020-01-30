import pandas as pd
import numpy as np
import re


#This function reorder the testset for ALS algorithm.
def reorderTestset(testset):
    d = []

    for row in testset:
        d.append(("r"+str(row[0])+"_c"+str(row[1]), row[2]))

    return pd.DataFrame(d, columns = ('Id', 'Prediction'))


#Reformat the data according to surpise library.
def split3columns(data):
    cleanedFrame = pd.DataFrame({
        'userId': data["Id"].apply(lambda x: int(re.search('r(.*)_' , x).group(1))),
        'movieId': data["Id"].apply(lambda x: int(re.search('c(.*)' , x).group(1))),
        'rating' : data["Prediction"]
    })
    return cleanedFrame

#get the validation labels
def getValidationLabels(validation_test):
    yvalid = []
    for i in range(len(validation_test)):
        yvalid.append(validation_test[i][2])

    yvalid = np.asarray(yvalid)

    return yvalid

def globalMean(data, length):
    return [data["Prediction"].mean(axis = 0)] * length

def userMean(trainingset, validation_frame):
    dictionary = trainingset.groupby(['userId']).mean()['rating']
    join_frame = validation_frame.merge(dictionary, on='userId')
    user_frame = join_frame.drop(columns=['rating_x'])

    return user_frame['rating_y'].values

def itemMean(trainingset, validation_frame):
    dictionary_items = trainingset.groupby(['movieId']).mean()['rating']
    item_frame = validation_frame.drop(columns=['userId'])
    pred_items = item_frame['rating'].array

    for index, row in enumerate(item_frame.values):
      pred_items[index] = dictionary_items[row[0]]

    return pred_items

#put all targets rounded in the sample_submission format.
def reconstructSampleSubmissionFormat(test_predictions_example, targets_rounded):
    d = []

    for index, p in enumerate(test_predictions_example):
        pred = np.round(targets_rounded[index])
        d.append(("r"+str(p[0])+"_c"+str(p[1]), pred))

    return pd.DataFrame(d, columns = ('Id', 'Prediction'))
