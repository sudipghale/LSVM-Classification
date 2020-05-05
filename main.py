'''
Sudip Ghale (1001557881, sxg7881)
CSE - 4334-002
Programming Assignment 2
04/28/2020
'''

import pandas as pd
import warnings


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

#The dataset contains attributes such as player name and team name. 
#We know that they are not useful for classification and thus do not 
#include them as features. 
feature_columns = ['G', 'GS', 'MP',  'FG','FG%', '3P', '3PA', \
    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']

#all given features for reference
'''
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
'''

#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]


#split our data into a 75% training and a 25% test set so we can evaluate generalization performance: 
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)
    
    
#The LinearSVC classifer       
with warnings.catch_warnings():
    warnings.simplefilter('ignore')    
    linearsvm = LinearSVC( loss='hinge', intercept_scaling=5, max_iter=10000, random_state=0).fit(train_feature, train_class)
print("Test set score (LinearSVC): {:.3f}".format(linearsvm.score(test_feature, test_class)))

    
#The Confusion matrix
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    prediction = linearsvm.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))



#The 10-fold stratified cross-validation
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    linearsvm = LinearSVC(loss='hinge', intercept_scaling=5, max_iter=10000, random_state=0).fit(train_feature, train_class)
    scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.3f}".format(scores.mean()))