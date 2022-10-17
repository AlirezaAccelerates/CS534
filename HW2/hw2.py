'''
CS 534: Machine Learning
Alireza Rafiei - Fall 2022 - HW2
'''

'''
Description: 
    In this assignment,  classical ML models for categorizing social and news media 
    posts as "fake" vs. "real" is experienced. In this regard, several ML models were
    developed to classify tweets as fake or real based on this dataset:
    Covid19 News dataset: https://github.com/diptamath/covid_fake_news
    
    For every single developed model, accuracy and F1 on test fold, range of hyperparameter 
    values compared, the final setting of hyperparameters is reported, while optimization of the 
    hyperparameters were on the validation fold only.
    
    This script is written for homework #2 machine learning course.
    It is available on https://github.com/AlirezaRafiei9/CS534/tree/main/HW2 repo
    with the required libraries for running the code.
    
    
IMPORTANT note for running the code:
    The code is modular and comprises different functions.
    A function is dedicated to every part of the problem, including ML models and processing steps.
    You can activate/deactivate running functions and classes via commenting/uncommenting the execution line.
    
    The functions are as follows:
        Data_loader()
        Data_exploration()
        preprocessing() 
        feature_engineering() 
        Naive_Bayes()
        Random_Forest()
        GDBT()
        SVM()

        
    The complete working steps of each function are described in their own sections. I've used
    "hyperopt" library to discover the optimized hyperparameters for the ML models. The procedure
    of finding the most optimized hyperparameters started with several trials and errors to estimate
    an initial range. Afterward, I've implemented a targeted search using "hyperopt" library to discover 
    the most optimized number for the hyperparameters. This library can be easily installed using the 
    following line code:
    !pip install hyperopt
    *ref: http://hyperopt.github.io/hyperopt/
    
    As running the search for the optimal hyperparameters is a time-consuming process, I've 
    commented on this process in the ML models' functions and just put the most optimal hyperparameters
    to run the models and show the results. However, you can easily run them by uncommenting those
    parts if you are interested (just best_hyperparams_... functions through the script). To prevent 
    any different results because of random processes of training ML models, I've set a specific 
    random_state for each model. The considered loss for finding the most optimized hyperparameters is 
    "-(accuracy+f1_score)/2". Besides, for every single model, the accuracy and f1-score are provided 
    using the shared SciKit Learn performance metrics links. Also, classification_report and confusion 
    matrix for each model is reported.
    
    Final results:
        
        Naive Bayes:
            Accuracy on test fold: 0.9187
            F1 score on test fold: 0.9138
            Classification report:
            
                          precision    recall  f1-score   support

                       0       0.93      0.91      0.92      1142
                       1       0.90      0.92      0.91       998

                accuracy                           0.92      2140
               macro avg       0.92      0.92      0.92      2140
            weighted avg       0.92      0.92      0.92      2140
            
            Range of hyperparameter values compared:
            alpha = (0.000005, 10)
            
            The final setting of hyperparameters:
            alpha = 0.03714265336014352
            
            
        Random Forest:
            Accuracy on test fold: 0.9121
            F1 score on test fold: 0.9093
            Classification report:

                          precision    recall  f1-score   support

                       0       0.90      0.93      0.91      1088
                       1       0.92      0.90      0.91      1052

                accuracy                           0.91      2140
               macro avg       0.91      0.91      0.91      2140
            weighted avg       0.91      0.91      0.91      2140
            
            Range of hyperparameter values compared:
            n_estimators = (1, 500), min_samples_leaf = (1, 10)
            
            The final setting of hyperparameters:
            n_estimators = 463, min_samples_leaf = 3
            
            
        GDBT:
            Accuracy on test fold: 0.9168
            F1 score on test fold: 0.9128
            Classification report:

                          precision    recall  f1-score   support

                       0       0.92      0.92      0.92      1118
                       1       0.91      0.91      0.91      1022

                accuracy                           0.92      2140
               macro avg       0.92      0.92      0.92      2140
            weighted avg       0.92      0.92      0.92      2140
            
            Range of hyperparameter values compared:
            learning_rate = (0, 0.95),n_estimators = (1, 500), min_samples_leaf = (1, 6)
            
            The final setting of hyperparameters:
            learning_rate= 0.4031859174188665, min_samples_leaf= 6, n_estimators= 444
            
            
        SVM:
            Accuracy on test fold: 0.9364
            F1 score on test fold: 0.9325
            Classification report:

                          precision    recall  f1-score   support

                       0       0.95      0.93      0.94      1146
                       1       0.92      0.94      0.93       994

                accuracy                           0.94      2140
               macro avg       0.94      0.94      0.94      2140
            weighted avg       0.94      0.94      0.94      2140
            
            Range of hyperparameter values compared:
            kernel = ['linear', 'poly', 'sigmoid'], C = (1, 100)
            
            The final setting of hyperparameters:
            kernel = 'linear' C = 1
            
        
'''


## Let's import the data

# Importing generally required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# Loading different subsets
def Data_loader():
    Constraint_Train = pd.read_csv("Constraint_Train.csv")
    Constraint_Val = pd.read_csv("Constraint_Val.csv")
    Constraint_Test = pd.read_csv("english_test_with_labels.csv")
    return Constraint_Train, Constraint_Val, Constraint_Test
    
Constraint_Train, Constraint_Val, Constraint_Test = Data_loader()

## Let's explore the data

def Data_exploration(train, val, test):
    print("\nConstraint_Train set\n")
    print("===================================")
    print(train.head(10))
    print(train.tail(10))
    print(train.describe())
    print(train.dtypes)
    print(train.isnull().sum())
    print(train.info())
    
    print("\nConstraint_Val set\n")
    print("===================================")
    print(val.head(10))
    print(val.tail(10))
    print(val.describe())
    print(val.dtypes)
    print(val.isnull().sum())
    print(val.info())
    
    print("\nConstraint_Test set\n")
    print("===================================")
    print(test.head(10))
    print(test.tail(10))
    print(test.describe())
    print(test.dtypes)
    print(test.isnull().sum())
    print(test.info())
    shape_train = train.shape
    print('\nThere are {} tweets and {} columns for each tweet in the train set.'.format(shape_train[0], shape_train[1]))
    print('The distribution of the train data into classes is:')
    print(train['label'].value_counts())

    shape_val = val.shape
    print('\nThere are {} tweets and {} columns for each tweet in the validation set.'.format(shape_val[0], shape_val[1]))
    print('The distribution of the validation data into classes is:')
    print(val['label'].value_counts())

    shape_test = test.shape
    print('\nThere are {} tweets and {} columns for each tweet in the test set.'.format(shape_test[0], shape_test[1]))
    print('The distribution of the test data into classes is:')
    print(test['label'].value_counts())
    

# Investigating all three sets
Data_exploration(Constraint_Train, Constraint_Val, Constraint_Test)


# Adding libraries for the preprocessing stepts
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import re
from collections import Counter
from numpy import array

# Dropping the id column of the data sets
try:
    column_to_drop = ['id']
    Constraint_Train = Constraint_Train.drop(column_to_drop, axis = 1)
    Constraint_Val = Constraint_Val.drop(column_to_drop, axis = 1)
    Constraint_Test = Constraint_Test.drop(column_to_drop, axis = 1)
except:
    None
    

stop_words = set(stopwords.words("english"))

def preprocessing(tweets):
    '''
    This function preprocesses the data using a specific workflow. I've examined all of the 
    presented following steps with several different combinations and orders to find the best
    process. The commented lines have also been checked in different orders with other processes, 
    but they did not show any improvement in the final performance of the ML models. As a result, 
    the keeped steps are:
    *Lower casing
    *Replace "http", "www" in urls with space
    *Replace ampersand(&) with "and"
    *Remove non-alphanumeric characters
    *Replace 'covid-19' and 'covid_19' with 'covid19'
    *Remove stop words
    '''
    
    # lowercasing the tweets
    text = tweets.lower().split()
    
    # Expanding contractions -- NOT useful
    #text = contractions.fix(text)
    
    #Replace "http", "www" in urls with space
    text = " ".join(text)
    text = re.sub(r"http(\S)+",' ',text)    
    text = re.sub(r"www(\S)+",' ',text)
    
    # Replace ampersand(&) with "and"
    text = re.sub(r"&",' and ',text)  

    # Remove non-alphanumeric characters
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text)
    
    # Replace 'covid-19' and 'covid_19' with 'covid19'
    covid = 'covid-19'
    covid_19 = 'covid_19'
    text = re.sub(covid, 'covid19', text)
    text = re.sub(covid_19, 'covid19', text)
    
    # Remove stop words
    text = text.split()
    text = [w for w in text if not w in stop_words]
        
    # Lemmatization and Stemming -- NOT useful
    #lemmatizer = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    #words_input = [lemmatizer.lemmatize(word) for word in text]
    #words_input = [stemmer.stem(word) for word in text]
    
    # filtering out short tokens -- NOT useful
    #words_input = [word for word in text if len(word) > 2]
    
    text = " ".join(text)
    return text

# Preprocessing the sets
Constraint_Train_Data = Constraint_Train.copy()
Constraint_Val_Data = Constraint_Val.copy()
Constraint_Test_Data = Constraint_Test.copy()
Constraint_Train_Data['tweet'] = Constraint_Train['tweet'].map(lambda x: preprocessing(x))
Constraint_Val_Data['tweet'] = Constraint_Val['tweet'].map(lambda y: preprocessing(y))
Constraint_Test_Data['tweet'] = Constraint_Test['tweet'].map(lambda z: preprocessing(z))

# Preprocessing results
print(Constraint_Train_Data.head())
print(Constraint_Val_Data.head())
print(Constraint_Test_Data.head())


## Feature Engineering

# BOW model and TF-IDF Representation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def feature_engineering(train_data_fe, val_data_fe, test_data_fe):
    '''
    This function is dedicated to feature engineering of the avaiable tweets. First, Bag of Words (BoW) 
    strategy is applied for converting the text document into numbers. Second, TfidfTransformer
    is considered to transform a count matrix to a normalized term-frequency or term-frequency times 
    inverse document-frequency representation. The extracted feature of this function are considered
    as the input of the ML models.
    '''
    
    # BOW
    vectorizer_train = CountVectorizer()
    vectorizer_val = CountVectorizer()
    vectorizer_test = CountVectorizer()

    X_train_bow = vectorizer_train.fit_transform(train_data_fe)
    X_val_bow = vectorizer_val.fit_transform(val_data_fe)
    X_test_bow = vectorizer_test.fit_transform(test_data_fe)

    print("There are {} tweets in the train set, each represented as a {} dim feature vector"          .format(X_train_bow.shape[0], X_train_bow.shape[1]))
    print("There are {} tweets in the test set, each represented as a {} dim feature vector"          .format(X_val_bow.shape[0], X_train_bow.shape[1]))
    print("There are {} tweets in the validation set, each represented as a {} dim feature vector"          .format(X_test_bow.shape[0], X_train_bow.shape[1]))

    # TF-IDF Representation
    tf_transformer_train = TfidfTransformer(smooth_idf=True,use_idf=True).fit(X_train_bow)
    tf_transformer_val = TfidfTransformer(smooth_idf=True,use_idf=True).fit(X_val_bow)
    tf_transformer_test = TfidfTransformer(smooth_idf=True,use_idf=True).fit(X_test_bow)

    X_train_tf = tf_transformer_train.transform(X_train_bow)
    X_val_tf = tf_transformer_val.transform(X_val_bow)
    X_test_tf = tf_transformer_test.transform(X_test_bow)


    print('\nThere are {} tweets in the train set and {} is the size of the vocabulary for all the tweets'          .format(X_train_tf.shape[0], X_train_tf.shape[1]))
    print('There are {} tweets in the validation set and {} is the size of the vocabulary for all the tweets'          .format(X_val_tf.shape[0], X_val_tf.shape[1]))
    print('There are {} tweets in the test set and {} is the size of the vocabulary for all the tweets\n'          .format(X_test_tf.shape[0], X_test_tf.shape[1]))
    
    return X_train_tf, X_val_tf, X_test_tf
    
X_train_tf, X_val_tf, X_test_tf = feature_engineering(Constraint_Train_Data['tweet'], 
                                                      Constraint_Val_Data['tweet'], Constraint_Test_Data['tweet'])

Constraint_Train_Label_Cat = Constraint_Train_Data['label'] 
Constraint_Train_Label_Cat = Constraint_Train_Data['label'].replace(['real', 'fake'],[0, 1])
Constraint_Val_Label_Cat = Constraint_Val_Data['label'] 
Constraint_Val_Label_Cat = Constraint_Val_Data['label'].replace(['real', 'fake'],[0, 1])
Constraint_Test_Label_Cat = Constraint_Test_Data['label'] 
Constraint_Test_Label_Cat = Constraint_Test_Data['label'].replace(['real', 'fake'],[0, 1])


## Let's import libraries for developing ML models
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
'''
if needed:
!pip install hyperopt
'''
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


### ........................ Problem #1 ........................ ###

# Naive Bayes
print("\nNaive Bayes\n")
# Exploring to discover the most optimal hyperparameters
'''
Hyperparameters to optimize: alpha
Range: (0.000005, 10)
'''
# As the searching steps are time-consuming, I've commented this part of code
# and put the best value obtained for evaluating the model on the test set. You
# can easily uncomment this part, and put "best_hyperparams_NB['alpha']" command 
# instead of the set value for alpha in the related pipeline.
print("\nExploring to discover the most optimal hyperparameters")
space_NB={'alpha': hp.uniform('alpha', 0.000005, 10)}

def objective_NB(space_NB):
    pipeline_NB = Pipeline([
            ('bow', CountVectorizer()),  
            ('tfidf', TfidfTransformer()),  
            ('clf', MultinomialNB(alpha = space_NB['alpha']))
        ])

    pipeline_NB.fit(Constraint_Train_Data['tweet'],Constraint_Train_Label_Cat)
    
    prediction_NB = pipeline_NB.predict(Constraint_Val['tweet'])
    accuracy = accuracy_score(prediction_NB, Constraint_Val_Label_Cat)
    f1 = f1_score(prediction_NB, Constraint_Val_Label_Cat)
  
    print ("SCORE: Accuracy_val {}, F1_val {}".format(accuracy, f1))
    return {'loss': -(accuracy+f1)/2, 'status': STATUS_OK }

trials_NB = Trials()
'''
best_hyperparams_NB = fmin(fn = objective_NB,
                        space = space_NB,
                        algo = tpe.suggest,
                        max_evals = 3,
                        trials = trials_NB)

print("\nThe best hyperparameters for the NB model are : ")
print(best_hyperparams_NB)
'''

print('\nRange of hyperparameter values compared: ')
print('alpha: (0.000005, 10)\n')           
print('The final setting of hyperparameters: ')
print('alpha = 0.03714265336014352')

pipeline_NB = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ('clf', MultinomialNB(alpha = 0.03714265336014352))
        ])

pipeline_NB.fit(Constraint_Train_Data['tweet'],Constraint_Train_Label_Cat)
    
prediction_NB = pipeline_NB.predict(Constraint_Test['tweet'])
print("\nThe accuracy of Naive Bayes model is: {:.4f}".format(accuracy_score(prediction_NB, Constraint_Test_Label_Cat)))
print("The f1_score of Naive Bayes model is: {:.4f}\n".format(f1_score(prediction_NB, Constraint_Test_Label_Cat)))
print("Naive Bayes classification report:\n")
print(classification_report(prediction_NB, Constraint_Test_Label_Cat))
confusion_matrix_NB = pd.crosstab(
    Constraint_Test_Label_Cat, 
    prediction_NB, 
    rownames = ['True Label'], 
    colnames = ['Predicted Label']
    )
fig, ax = plt.subplots(figsize=(6,4))
heatmap_NB = sns.heatmap(confusion_matrix_NB, annot = True, xticklabels= ['Fake', 'Real'], yticklabels = ['Fake', 'Real'], 
                      linewidths = 1.5, fmt = 'd', cmap="Blues")
plt.title("Confusion matrix of Naive Bayes on Test Data")
plt.show()


### ........................ Problem #2 ........................ ###

# Random Forest
print("\nRandom Forest\n")
# Exploring to discover the most optimal hyperparameters
'''
Hyperparameters to optimize: n_estimators, min_samples_leaf
Range: n_estimators = (1, 500), min_samples_leaf = (1, 10)
'''
# As the searching steps are time-consuming, I've commented this part of code
# and put the best value obtained for evaluating the model on the test set. You
# can easily uncomment this part, and put "best_hyperparams_RF['n_estimators']"
# and "best_hyperparams_RF['min_samples_leaf']" commands instead of the set value 
# for n_estimators and min_samples_leaf in the related pipeline.
# I've also set a random_state number to prevent different results because of needed random processes.
print("\nExploring to discover the most optimal hyperparameters")
space_RF={'n_estimators': hp.quniform('n_estimators', 1, 500, 1)
      ,'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
      ,'random_state': 3}

def objective_RF(space_RF):
    pipeline_RF = Pipeline([
            ('bow', CountVectorizer()),  
            ('tfidf', TfidfTransformer()),  
            ('clf', RandomForestClassifier(n_estimators = int(space_RF['n_estimators']), min_samples_leaf = int(space_RF['min_samples_leaf'])
                                          , random_state = space_RF['random_state']))])

    pipeline_RF.fit(Constraint_Train_Data['tweet'],Constraint_Train_Label_Cat)
    
    prediction_RF = pipeline_RF.predict(Constraint_Val['tweet'])
    accuracy = accuracy_score(prediction_RF, Constraint_Val_Label_Cat)
    f1 = f1_score(prediction_RF, Constraint_Val_Label_Cat)
    
    print ("SCORE: Accuracy_val {}, F1_val {}".format(accuracy, f1))
    return {'loss': -(accuracy+f1)/2, 'status': STATUS_OK }

trials_RF = Trials()
'''
best_hyperparams_RF = fmin(fn = objective_RF,
                        space = space_RF,
                        algo = tpe.suggest,
                        max_evals = 3,
                        trials = trials_RF)

print("\nThe best hyperparameters for the RF model are : ")
print(best_hyperparams_RF)
'''
print('\nRange of hyperparameter values compared: ')
print('n_estimators: (1, 500)')
print('min_samples_leaf: (1, 10)')  
print('\nThe final setting of hyperparameters: ')
print('n_estimators = 463')
print('min_samples_leaf = 1')  

pipeline_RF = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ('clf', RandomForestClassifier(min_samples_leaf = 1, n_estimators = 463, random_state= 3))
        ])

pipeline_RF.fit(Constraint_Train_Data['tweet'],Constraint_Train_Label_Cat)
    
prediction_RF = pipeline_RF.predict(Constraint_Test['tweet'])
print("\nThe accuracy of Random Forest is: {:.4f}".format(accuracy_score(prediction_RF, Constraint_Test_Label_Cat)))
print("The f1_score of Random Forest is: {:.4f}\n".format(f1_score(prediction_RF, Constraint_Test_Label_Cat)))
print("Random Forest classification report:\n")
print(classification_report(prediction_RF, Constraint_Test_Label_Cat))
confusion_matrix_RF = pd.crosstab(
    Constraint_Test_Label_Cat, 
    prediction_RF, 
    rownames = ['True Label'], 
    colnames = ['Predicted Label']
    )
fig, ax = plt.subplots(figsize=(6,4))
heatmap_RF = sns.heatmap(confusion_matrix_RF, annot = True, xticklabels= ['Fake', 'Real'], yticklabels = ['Fake', 'Real'], 
                      linewidths = 1.5, fmt = 'd', cmap="Blues")
plt.title("Confusion matrix of Random Forest on Test Data")
plt.show()


# GDBT
print("\nGDBT\n")
# Exploring to discover the most optimal hyperparameters
'''
Hyperparameters to optimize: learning_rate, n_estimators, min_samples_leaf
Range: learning_rate = (0, 0.95), n_estimators = (1, 500), min_samples_leaf = (1, 6)
'''
# As the searching steps are time-consuming, I've commented this part of code
# and put the best value obtained for evaluating the model on the test set. You
# can easily uncomment this part, and put "best_hyperparams_GDBT['learning_rate']" 
# "best_hyperparams_GDBT['n_estimators']" and "best_hyperparams_GDBT['min_samples_leaf']" 
# commands instead of the set value for n_estimators and min_samples_leaf in the related pipeline.
# I've also set a random_state number to prevent different results because of needed random processes.
print("\nExploring to discover the most optimal hyperparameters")
space_GDBT={'learning_rate': hp.uniform('learning_rate', 0, 0.95)
            ,'n_estimators': hp.quniform('n_estimators', 1, 500,1)
            ,'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 6,1)
            ,'random_state' : 3}

def objective_GDBT(space_GDBT):
    pipeline_GDBT = Pipeline([
            ('bow', CountVectorizer()),  
            ('tfidf', TfidfTransformer()),  
            ('clf', GradientBoostingClassifier(learning_rate = space_GDBT['learning_rate'], n_estimators = int(space_GDBT['n_estimators']), 
                                               min_samples_leaf = int(space_GDBT['min_samples_leaf']), random_state = space_GDBT['random_state']))])

    pipeline_GDBT.fit(Constraint_Train_Data['tweet'], Constraint_Train_Label_Cat)
    
    prediction_GDBT = pipeline_GDBT.predict(Constraint_Val['tweet'])
    accuracy = accuracy_score(prediction_GDBT, Constraint_Val_Label_Cat)
    f1 = f1_score(prediction_GDBT, Constraint_Val_Label_Cat)
    
    print ("SCORE: Accuracy {}, F1 {}".format(accuracy, f1))
    return {'loss': -(accuracy+f1)/2, 'status': STATUS_OK }

trials_GDBT = Trials()
'''
best_hyperparams_GDBT = fmin(fn = objective_GDBT,
                        space = space_GDBT,
                        algo = tpe.suggest,
                        max_evals = 3,
                        trials = trials_GDBT)

print("\nThe best hyperparameters for the GDBT model are : ")
print(best_hyperparams_GDBT)
'''
print('\nRange of hyperparameter values compared: ')
print('learning_rate: (0, 0.95)')
print('n_estimators: (1, 500)')
print('min_samples_leaf: (1, 6)')  
print('\nThe final setting of hyperparameters: ')
print('learning_rate = 0.4031859174188665')
print('n_estimators = 463')
print('min_samples_leaf = 1')  

pipeline_GDBT = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ('clf', GradientBoostingClassifier(learning_rate= 0.4031859174188665, min_samples_leaf= 6, n_estimators= 444))
        ])

pipeline_GDBT.fit(Constraint_Train_Data['tweet'], Constraint_Train_Label_Cat)
    
prediction_GDBT = pipeline_GDBT.predict(Constraint_Test['tweet'])
print("\nThe accuracy of GDBT is: {:.4f}".format(accuracy_score(prediction_GDBT, Constraint_Test_Label_Cat)))
print("The f1_score of GDBT  is: {:.4f}\n".format(f1_score(prediction_GDBT, Constraint_Test_Label_Cat)))
print("GDBT classification report:\n")
print(classification_report(prediction_GDBT, Constraint_Test_Label_Cat))

confusion_matrix_GDBT = pd.crosstab(Constraint_Test_Label_Cat, 
    prediction_GDBT, 
    rownames = ['True Label'], 
    colnames = ['Predicted Label']
    )
fig, ax = plt.subplots(figsize=(6,4))
heatmap_RF = sns.heatmap(confusion_matrix_GDBT, annot = True, xticklabels= ['Fake', 'Real'], yticklabels = ['Fake', 'Real'], 
                      linewidths = 1.5, fmt = 'd', cmap="Blues")
plt.title("Confusion matrix of GDBT on Test Data")
plt.show()



### ........................ Problem #3 ........................ ###

# SVM
print("\nSVM\n")
# Exploring to discover the most optimal hyperparameters
'''
Hyperparameters to optimize: kernel, C
Range: kernel = ['linear', 'poly', 'sigmoid'], C = (1, 100)
'''
# As the searching steps are time-consuming, I've commented this part of code
# and put the best value obtained for evaluating the model on the test set. You
# can easily uncomment this part, and put "best_hyperparams_GDBT['learning_rate']" 
# "best_hyperparams_GDBT['n_estimators']" and "best_hyperparams_GDBT['min_samples_leaf']" 
# commands instead of the set value for n_estimators and min_samples_leaf in the related pipeline.
# I've also set a random_state number to prevent different results because of needed random processes.
print("\nExploring to discover the most optimal hyperparameters")
space_SVM = {'kernel': hp.choice('svm_kernel', ['linear', 'poly', 'sigmoid']),'C': hp.quniform('C', 1, 100,1)}

def objective_SVM(space_SVM):
    pipeline_SVM = Pipeline([
            ('bow', CountVectorizer()),  
            ('tfidf', TfidfTransformer()),  
            ('clf', SVC(kernel = space_SVM['kernel'],C = space_SVM['C']))])

    pipeline_SVM.fit(Constraint_Train_Data['tweet'],Constraint_Train_Label_Cat)
    
    prediction_SVM = pipeline_SVM.predict(Constraint_Val['tweet'])
    accuracy = accuracy_score(prediction_SVM, Constraint_Val_Label_Cat)
    f1 = f1_score(prediction_SVM, Constraint_Val_Label_Cat)
    
    print ("SCORE: Accuracy {}, F1 {}".format(accuracy, f1))
    return {'loss': -(accuracy+f1)/2, 'status': STATUS_OK }

trials_SVM = Trials()
'''
best_hyperparams_SVM = fmin(fn = objective_SVM,
                        space = space_SVM,
                        algo = tpe.suggest,
                        max_evals = 3,
                        trials = trials_SVM)

print("\nThe best hyperparameters for the GDBT model are : ")
print(best_hyperparams_SVM)
'''
print('\nRange of hyperparameter values compared: ')
print("kernel: ['linear', 'poly', 'sigmoid']")
print('C: (1, 100)')
print('\nThe final setting of hyperparameters: ')
print('kernel = linear')
print('C = 1') 

pipeline_SVM = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ('clf', SVC(kernel = 'linear', C = 1))
        ])

pipeline_SVM.fit(Constraint_Train_Data['tweet'], Constraint_Train_Label_Cat)
    
prediction_SVM = pipeline_SVM.predict(Constraint_Test['tweet'])
print("\nThe accuracy of SVM is: {:.4f}".format(accuracy_score(prediction_SVM, Constraint_Test_Label_Cat)))
print("The f1_score of SVM is: {:.4f}\n".format(f1_score(prediction_SVM, Constraint_Test_Label_Cat)))
print("SVM classification report:\n")
print(classification_report(prediction_SVM, Constraint_Test_Label_Cat))
confusion_matrix_SVM = pd.crosstab(Constraint_Test_Label_Cat, 
                                   prediction_SVM, 
                                    rownames = ['True Label'], 
                                    colnames = ['Predicted Label']
                                    )
fig, ax = plt.subplots(figsize=(6,4))
heatmap_SVM = sns.heatmap(confusion_matrix_SVM, annot = True, xticklabels= ['Fake', 'Real'], yticklabels = ['Fake', 'Real'], linewidths = 1.5, fmt = 'd', cmap="Blues")
plt.title("Confusion matrix of SVM on Test Data")
plt.show()