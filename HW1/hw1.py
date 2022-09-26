'''
CS 534: Machine Learning
Alireza Rafiei - Fall 2022 - HW1
'''

'''
IMPORTANT note for running the code:
    This script is written for homework #1 machine learning course.
    It is available on https://github.com/AlirezaRafiei9/CS534/tree/main/HW1 repo 
    with the required libraries for running the code.
    The code is modular and comprises different functions.
    A function is dedicated to every part of problem #1.
    Problems #2 and #3 can be run using a dedicated python class.
    You can activate/deactivate running functions and classes via comment/uncomment the execution line.
    
    The functions and classes are:
        Data_exploration()
        Data_visualization() default: inactive
        Pre_precossing() default: inactive
        Extraction()
        P1A()
        P1B()
        P1C()
        P1D()
        FakeRidgeRegression()
        LinearRegressionWithKnowledge()
        
    The Pre_processing function preprocesses the raw data. As the questions did not ask,
    the default data for running functions and classes is the unpreprocessed one.
    However, you can easily activate the related function and add "_preprocessed" after your 
    data name to start working with preprocessed data (e.g., Train_preprocessed instead of Train)
        
'''

### ........................ Problem #1 ........................ ###


"""

The data has 27 attributes for each 10 minute interval, which are described in detail on the UCL ML repository, 
Applicances energy prediction dataset: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction.

Attribute Information:

date time year-month-day hour:minute:second
Appliances, energy use in Wh
lights, energy use of light fixtures in the house in Wh
T1, Temperature in kitchen area, in Celsius
RH_1, Humidity in kitchen area, in %
T2, Temperature in living room area, in Celsius
RH_2, Humidity in living room area, in %
T3, Temperature in laundry room area
RH_3, Humidity in laundry room area, in %
T4, Temperature in office room, in Celsius
RH_4, Humidity in office room, in %
T5, Temperature in bathroom, in Celsius
RH_5, Humidity in bathroom, in %
T6, Temperature outside the building (north side), in Celsius
RH_6, Humidity outside the building (north side), in %
T7, Temperature in ironing room , in Celsius
RH_7, Humidity in ironing room, in %
T8, Temperature in teenager room 2, in Celsius
RH_8, Humidity in teenager room 2, in %
T9, Temperature in parents room, in Celsius
RH_9, Humidity in parents room, in %
To, Temperature outside (from Chievres weather station), in Celsius
Pressure (from Chievres weather station), in mm Hg
RH_out, Humidity outside (from Chievres weather station), in %
Wind speed (from Chievres weather station), in m/s
Visibility (from Chievres weather station), in km
Tdewpoint (from Chievres weather station), Â°C
rv1, Random variable 1, nondimensional
rv2, Random variable 2, nondimensional

The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. 
Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless 
data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy 
meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from 
a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets 
using the date and time column. Two random variables have been included in the data set for testing the 
regression models and to filter out non predictive attributes (parameters).

"""
## Let's import the data

# Importing generally required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading different subsets
Train = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")
Validation = pd.read_csv("validation.csv")


## Let's explore the data
def Data_exploration(data):
    print(data.head(10))
    print(data.tail(10))
    print(data.describe())
    ## Is there any null values in the data?! (Organization, cleaning, and data treatment)
    # Checking all data is numerical
    print(data.dtypes)
    print(data.isnull().sum())
    print(data.info())

# Investigating the train set
Data_exploration(Train)

## Let's do data visualization
def Data_visualization(data):
    his = data.hist(figsize = (20,20))
    # Correlation between the attributes
    df = data.drop(['date', 'Appliances'], axis=1)
    plt.figure(figsize=(20, 15))
    heatmap = df.corr()
    sns.heatmap(data=heatmap, annot=True)

    
# Visualizing the train set
#Data_visualization(Train)


def Pre_precossing(Train, Test, Validation):
    '''
    This function preprocesses the data using the following steps:
    * Retrieve the year, month, week, day, and hour of every single record.
    * Calculate the correlation between attributes.
        discarding the attributes that repeat more than 2 times with a correlation of
        more than 85% with another attribute. Not more than 2 attributes from a 
        particular type of sensor.
    * removing attributes with less than 5% correlation with the target.
    * Standardizing the data
    '''
    global Train_preprocessed
    global Test_preprocessed
    global Validation_preprocessed
    
    # Pre-processing steps
    # Retrieving year, month, week, day, and hours for the provided date
    Train_preprocessed = Train.copy()
    Train_preprocessed['date'] = pd.to_datetime(Train_preprocessed['date'])
    Train_preprocessed['yaer'] = Train_preprocessed.date.dt.year
    Train_preprocessed['month'] = Train_preprocessed.date.dt.month
    Train_preprocessed['weekday'] = Train_preprocessed.date.dt.weekday
    Train_preprocessed['hour'] = Train_preprocessed.date.dt.hour
    Train_preprocessed['week'] = Train_preprocessed.date.dt.week
    Train_preprocessed = Train_preprocessed.drop(['date'], axis=1)
    
    Test_preprocessed = Train.copy()
    Test_preprocessed['date'] = pd.to_datetime(Test['date'])
    Test_preprocessed['yaer'] = Test_preprocessed.date.dt.year
    Test_preprocessed['month'] = Test_preprocessed.date.dt.month
    Test_preprocessed['weekday'] = Test_preprocessed.date.dt.weekday
    Test_preprocessed['hour'] = Test_preprocessed.date.dt.hour
    Test_preprocessed['week'] = Test_preprocessed.date.dt.week
    Test_preprocessed = Test_preprocessed.drop(['date'], axis=1)
    
    Validation_preprocessed = Train.copy()
    Validation_preprocessed['date'] = pd.to_datetime(Validation['date'])
    Validation_preprocessed['yaer'] = Validation_preprocessed.date.dt.year
    Validation_preprocessed['month'] = Validation_preprocessed.date.dt.month
    Validation_preprocessed['weekday'] = Validation_preprocessed.date.dt.weekday
    Validation_preprocessed['hour'] = Validation_preprocessed.date.dt.hour
    Validation_preprocessed['week'] = Validation_preprocessed.date.dt.week
    Validation_preprocessed = Validation_preprocessed.drop(['date'], axis=1)
    
    # Calculating the correlation between attributes
    def Correlated_pairs(df):
        Correlated_pairs = set()
        col = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                Correlated_pairs.add((col[i], col[j]))
        return Correlated_pairs

    def top_correlations(df,n):
        corrs = df.corr().abs().unstack()
        correlated_labels = Correlated_pairs(df)
        corrs = corrs.drop(labels=correlated_labels).sort_values(ascending=False)
        return corrs[0:n]

    print("TopcCorrelation pairs")
    print(top_correlations(Train_preprocessed,50))
    
    # Removing the correlated pairs
    Train_preprocessed = Train_preprocessed.drop(['RH_8', 'RH_9', 'T1'], axis=1)
    Test_preprocessed = Test_preprocessed.drop(['RH_8', 'RH_9', 'T1'], axis=1)
    Validation_preprocessed = Validation_preprocessed.drop(['RH_8', 'RH_9', 'T1'], axis=1)
    
    # Removing the attributes that have less than 5% correlation to the target variable
    corr_matrix = Train_preprocessed.corr()
    attributes_to_keep = corr_matrix.loc[~corr_matrix['Appliances'].between(-0.05, 0.05)]
    print('\nThe attributes to keep are:\n')
    print(attributes_to_keep.index.tolist())
    Train_preprocessed = Train_preprocessed[attributes_to_keep.index.tolist()]
    Validation_preprocessed = Train_preprocessed[attributes_to_keep.index.tolist()]
    Test_preprocessed = Train_preprocessed[attributes_to_keep.index.tolist()]
    
    # Standardization of the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() 
    Train_preprocessed=pd.DataFrame(scaler.fit_transform(Train_preprocessed))
    Test_preprocessed=pd.DataFrame(scaler.fit_transform(Test_preprocessed))
    Validation_preprocessed=pd.DataFrame(scaler.fit_transform(Validation_preprocessed))

    X_train_preprocessed = Train_preprocessed.drop([0], axis=1)
    y_train_preprocessed = Train_preprocessed[0]
    X_validation_preprocessed = Validation_preprocessed.drop([0], axis=1)
    y_validation_preprocessed = Validation_preprocessed[0]
    X_test_preprocessed = Test_preprocessed.drop([0], axis=1)
    y_test_preprocessed = Test_preprocessed[0]
    X_TrainVal_preprocessed = pd.concat([X_train_preprocessed, X_validation_preprocessed])
    y_TrainVal_preprocessed = pd.concat([y_train_preprocessed, y_validation_preprocessed])
    
    return X_train_preprocessed, y_train_preprocessed, X_validation_preprocessed, y_validation_preprocessed,            X_test_preprocessed, y_test_preprocessed, X_TrainVal_preprocessed, y_TrainVal_preprocessed

#X_train_preprocessed, y_train_preprocessed, X_validation_preprocessed, y_validation_preprocessed, X_test_preprocessed, \
#y_test_preprocessed, X_TrainVal_preprocessed, y_TrainVal_preprocessed = Pre_precossing(Train, Test, Validation)


# Extracting data and labels for different subsets without preprocessing steps
def Extraction(Train, Validation,Test):
    X_train = Train.drop(['date', 'Appliances'], axis=1)
    y_train = Train['Appliances']
    X_validation = Validation.drop(['date', 'Appliances'], axis=1)
    y_validation = Validation['Appliances']
    X_test = Test.drop(['date', 'Appliances'], axis=1)
    y_test = Test['Appliances']
    X_TrainVal = pd.concat([X_train, X_validation])
    y_TrainVal = pd.concat([y_train, y_validation])
    return X_train, y_train, X_validation, y_validation, X_test, y_test, X_TrainVal, y_TrainVal

X_train, y_train, X_validation, y_validation, X_test, y_test, X_TrainVal, y_TrainVal = Extraction(Train, Validation, Test)


## Let's start training our models
#### P1A

# Training the Regression models
from sklearn import linear_model
from sklearn import metrics

def P1A(X_train,y_train, X_validation, y_validation, X_test, y_test):
    print('\nProblem 1 -- Part A\n')
    
    # Training the model
    LinReg = linear_model.LinearRegression()
    LinReg.fit(X_train,y_train)
    print("The coefficients for the developed model are {}\n".format(LinReg.coef_))
    print("The intercept for the developed model is {}\n".format(LinReg.intercept_))
    # Calculating RMSE and R2 and other metrics for the subsets
    # Train set
    y_pred_train = LinReg.predict(X_train)
    print("MAE train is {}".format(metrics.mean_absolute_error(y_train,y_pred_train)))
    print("MSE train is {}".format(metrics.mean_squared_error(y_train,y_pred_train)))
    print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))))
    print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train)))

    # Validation set
    y_pred_validation = LinReg.predict(X_validation)
    print("MAE validation is {}".format(metrics.mean_absolute_error(y_validation,y_pred_validation)))
    print("MSE validation is {}".format(metrics.mean_squared_error(y_validation,y_pred_validation)))
    print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))))
    print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation))) 

    # test set
    y_pred_test = LinReg.predict(X_test)
    print("MAE test is {}".format(metrics.mean_absolute_error(y_test,y_pred_test)))
    print("MSE test is {}".format(metrics.mean_squared_error(y_test,y_pred_test)))
    print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))))
    print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test)))
    
P1A(X_train,y_train, X_validation, y_validation, X_test, y_test)

#### P1B
def P1B(X_TrainVal, y_TrainVal, X_train,y_train, X_validation, y_validation, X_test, y_test):
    print('\nProblem 1 -- Part B\n')
    
    # Training the model
    LinReg2 = linear_model.LinearRegression()
    LinReg2.fit(X_TrainVal, y_TrainVal)
    print("The coefficients for the developed model are {}\n".format(LinReg2.coef_))
    print("The intercept for the developed model is {}\n".format(LinReg2.intercept_))
    # Calculating RMSE and R2 and other metrics for the subsets
    # Train set
    y_pred_train2 = LinReg2.predict(X_train)
    print("MAE train is {}".format(metrics.mean_absolute_error(y_train,y_pred_train2)))
    print("MSE train is {}".format(metrics.mean_squared_error(y_train,y_pred_train2)))
    print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train2))))
    print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train2)))

    # Validation set
    y_pred_validation2 = LinReg2.predict(X_validation)
    print("MAE validation is {}".format(metrics.mean_absolute_error(y_validation,y_pred_validation2)))
    print("MSE validation is {}".format(metrics.mean_squared_error(y_validation,y_pred_validation2)))
    print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation2))))
    print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation2))) 

    # test set
    y_pred_test2 = LinReg2.predict(X_test)
    print("MAE test is {}".format(metrics.mean_absolute_error(y_test,y_pred_test2)))
    print("MSE test is {}".format(metrics.mean_squared_error(y_test,y_pred_test2)))
    print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test2))))
    print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test2))) 
    
P1B(X_TrainVal, y_TrainVal, X_train,y_train, X_validation, y_validation, X_test, y_test)


p1b = '''Adding the validation set for the training purpose enhances the number of training data. 
This enhancement in the training data could increase the performance of machine learning algorithms. 
Thus, it would increase the R2 score and decrease RMSE values.
As shown in the results, the performance of the developed models on the train set was roughly similar. 
However, the performance of the second model, which was developed by both training and validation sets, 
on the validation set dramatically increased. This is because the model saw this set during the training 
phase. Besides, the second model outperforms the first one on the test set. That is, the first model does 
not follow the data trend in the test set (negative R2), while the second has better performance. Therefore, 
with an increment in the number of training data, we had a growth in the model's performance on the test set. '''
print(p1b)


#### P1C
def P1C(X_train,y_train, X_validation, y_validation, X_test, y_test):
    print('\nProblem 1 -- Part C\n')
    
    # Training a Ridge regression model only on the training data based on the lowest RMSE on the validation set. 
    # I aim to find the optimal regularization parameter using a grid search between 0.01 to 20 with 0.01 step size.
    print('\nRidge Regression\n')
    Lambda_vals_Ridge = []
    RMSE_vals = []
    for L in np.arange(0.01,20,0.01):
        print("Lambda {}".format(L))
        RidReg = linear_model.Ridge(alpha=L)
        RidReg.fit(X_train,y_train)

        #Train set
        y_pred_train = RidReg.predict(X_train)
        print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))))
        print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train)))

        # Validation set
        y_pred_validation = RidReg.predict(X_validation)
        print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))))
        print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation))) 

        # test set
        y_pred_test = RidReg.predict(X_test)
        print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))))
        print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test))) 

        Ridge_RMSE_val = np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))
        print("Criteria for selecting the optimal Lambda (Validation RMSE): {}".format(Ridge_RMSE_val))
        print("--------------------------------------------------------------------------------------\n")
        RMSE_vals.append(Ridge_RMSE_val)
        Lambda_vals_Ridge.append(L)

    min_val, min_inx = min((min_val, min_inx) for (min_inx, min_val) in enumerate(RMSE_vals))
    L_optimal_Ridge = Lambda_vals_Ridge[min_inx]
    print('The optimal regularization parameter for the Ridge regression model is {}\n'.format(L_optimal_Ridge))
    
    # Training a Lasso regression model only on the training data based on the lowest RMSE on the validation set. 
    # I aim to find the optimal regularization parameter using a grid search between 0.01 to 20 with 0.01 step size.
    print('\nLasso Regression\n')
    Lambda_vals_Lasso = []
    RMSE_vals = []
    for L in np.arange(0.01,20,0.01):
        print("Lambda {}".format(L))
        LassoReg = linear_model.Lasso(alpha=L)
        LassoReg.fit(X_train,y_train)

        #Train set
        y_pred_train = LassoReg.predict(X_train)
        print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))))
        print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train)))

        # Validation set
        y_pred_validation = LassoReg.predict(X_validation)
        print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))))
        print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation))) 

        # test set
        y_pred_test = LassoReg.predict(X_test)
        print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))))
        print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test))) 

        Lasso_RMSE_val = np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))
        print("Criteria for selecting the optimal Lambda (Validation RMSE): {}".format(Lasso_RMSE_val))
        print("--------------------------------------------------------------------------------------\n")
        RMSE_vals.append(Lasso_RMSE_val)
        Lambda_vals_Lasso.append(L)

    min_val, min_inx = min((min_val, min_inx) for (min_inx, min_val) in enumerate(RMSE_vals))
    L_optimal_Lasso = Lambda_vals_Lasso[min_inx]
    print('The optimal regularization parameter for the Lasso regression model is {}\n'.format(L_optimal_Lasso))
    
    return L_optimal_Ridge, L_optimal_Lasso

    
L_optimal_Ridge, L_optimal_Lasso = P1C(X_train,y_train, X_validation, y_validation, X_test, y_test)

#### P1D
def P1D(X_TrainVal, y_TrainVal, X_train,y_train, X_validation, y_validation, X_test, y_test):
    print('\nProblem 1 -- Part D\n')
    
    # Training a Ridge regression on both the training and validation set with the optimal regularization parameter.
    RidReg = linear_model.Ridge(alpha=L_optimal_Ridge)
    RidReg.fit(X_TrainVal, y_TrainVal)
    W = RidReg.coef_
    Ridge_reg_coefs = RidReg.coef_
    print("The coefficients for the developed Ridge model are {}\n".format(RidReg.coef_))
    print("The intercept for the developed model is {}\n".format(RidReg.intercept_))

    # Calculating RMSE and R2 for the developed optimal Ridge model on the subsets.

    # Train set
    y_pred_train = RidReg.predict(X_train)
    print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))))
    print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train)))

    # Validation set
    y_pred_validation = RidReg.predict(X_validation)
    print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))))
    print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation))) 

    # test set
    y_pred_test = RidReg.predict(X_test)
    print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))))
    print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test))) 

    # Training a Lasso regression on both the training and validation set with the optimal regularization parameter.
    LassoReg = linear_model.Lasso(alpha=L_optimal_Ridge)
    LassoReg.fit(X_TrainVal, y_TrainVal)
    print("The coefficients for the developed Lasso model are {}\n".format(LassoReg.coef_))
    print("The intercept for the developed model is {}\n".format(LassoReg.intercept_))

    # Calculating RMSE and R2 for the developed optimal Lasso model on the subsets

    # Train set
    y_pred_train = LassoReg.predict(X_train)
    print("RMSE train is {}".format(np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))))
    print("R2 train is {}\n".format(metrics.r2_score(y_train,y_pred_train)))

    # Validation set
    y_pred_validation = LassoReg.predict(X_validation)
    print("RMSE validation is {}".format(np.sqrt(metrics.mean_squared_error(y_validation,y_pred_validation))))
    print("R2 validation is {}\n".format(metrics.r2_score(y_validation,y_pred_validation))) 

    # test set
    y_pred_test = LassoReg.predict(X_test)
    print("RMSE test is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))))
    print("R2 test is {}\n".format(metrics.r2_score(y_test,y_pred_test))) 
    return W, Ridge_reg_coefs
    
W, Ridge_reg_coefs = P1D(X_TrainVal, y_TrainVal, X_train,y_train, X_validation, y_validation, X_test, y_test)


p1d = '''Similar to the standard linear regression part, when we added the validation set to the training data, 
the performance metrics of the Ridge and Lasso models on the test set increased. This enhancement 
was higher for the Ridge regression model. In addition, the performance metrics of the models on 
the validation set had a growth. Although the performance of the Ridge model had not a considerable 
change when we added the validation data on the training set, the Lasso model experienced modest 
growth. '''
print(p1d)


### ........................ Problem #2 ........................ ###
class FakeRidgeRegression() :
    print('\nProblem 2\n')
    
    '''
    This class aims to show that the ridge regression estimates can be obtained 
    by ordinary least squares regression on an augmented data set. In this regard,
    We augment the centered matrix X with k additional rows √λI and augment y with
    k zeros. The idea is that by introducing artificial data having response value zero, 
    the fitting procedure is forced to shrink the coefficients towards zero.
  
  
Parameters:
    -----------
    Lambda  -float
        Penalty term for the Ridge regression model, which is used for data augmentation as well.
    k  -int 
        The additional number of rows with the aim of using for data augmentation.

Return:
    ----------
    The coefficients of attributes of the developed models.
    
    '''
      
    def __init__( self, Lambda, k) :
                        
        self.Lambda = Lambda
        self.k = k
        
    def Data_augmentation(self)  :

        Aug_data = pd.DataFrame(np.sqrt(self.Lambda)*np.identity(self.k))
        Aug_label = pd.DataFrame(np.zeros((self.k, 1)))
        Fake_data = pd.DataFrame(np.concatenate((X_train, Aug_data),axis=0))
        Fake_label = pd.concat([self.y, Aug_label], axis=0,ignore_index=True)
        return Fake_data, Fake_label
        
    # Training of the model           
    def fit(self, X, y) :
        
        # Number of training examples, number of attributes 
        self.m, self.n = X.shape
        self.X = X        
        self.y = y  
        Fake_data, Fake_label = self.Data_augmentation()
        
        LinReg = linear_model.LinearRegression()
        LinReg.fit(Fake_data,Fake_label)
        
        RidReg = linear_model.Ridge(self.Lambda)
        RidReg.fit(self.X,self.y)
        
        LC = pd.DataFrame(LinReg.coef_.T)
        RC = pd.DataFrame(RidReg.coef_.T)
        Cofficients = pd.concat([LC, RC], axis=1, ignore_index=True)
        Cofficients.columns = ["Fake coefficients learned", "Real cofficients learned"]
        pd.set_option('display.colheader_justify', 'center')
        print(Cofficients)
        
fake = FakeRidgeRegression(Lambda = L_optimal_Ridge, k = 25)
fake.fit(X_train, y_train)


### ........................ Problem #3 ........................ ###
class LinearRegressionWithKnowledge() :
    print('\nProblem 3\n')
    
    '''
    If we use the regular Ridge Regression, then many of the coefficients will be estimated near
    zeros, which doesn’t conform with the experts’ domain knowledge by incorporating an expert’s
    intuition about importance of coefficients. The idea is to modify the penalty term of the loss 
    function to reflect the prior knowledge. A new loss function is as follows:
    np.sum(np.square(y - np.dot(X, coef))) + lmbd * np.sum(np.square(coef - coef_prior))
    This class implement regression using prior knowlendge.
    
Parameters:
    -----------
    learning_rate  -float
        The step size at each iteration.
    Lambda  -float
        Penalty term.
    W_expt  -list
        The weights for attributes that are determined by experts.
    num_iter_no_change  -int
        The number of iterations with no improvement to wait before stopping fitting.
    max_iter  -int
        The maximum number of passes over the training data (aka epochs). 
    tol  -float
        The stopping criterion.

Return:
    ----------
    The weights of attributes of the developed model.
    
    '''
      
    def __init__( self, learning_rate, Lambda, W_expt, max_iter=1000, num_iter_no_change=3, tol = 0.001) :
          
        self.learning_rate = learning_rate                
        self.Lambda = Lambda
        self.W_expt = W_expt
        self.best_loss = np.inf
        self.max_iter = max_iter
        self.num_iter_no_change = num_iter_no_change
        self.tol = tol
        self.stop = True
          
    # Training of the model           
    def fit( self, X, Y ) :
          
        # Number of_training examples, number of attributes       
        self.m, self.n = X.shape
          
        # Parameter initialization        
        self.W = np.zeros( self.n )
        self.b = 0
        self.W_best = np.zeros( self.n )
        self.b_best = 0
        self.iteration = 0
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        while self.stop or (self.max_iter <= self.iteration):           
            self.update_weights()            
        return self
      
    # Updating weights in gradient descent
      
    def update_weights( self ) :           
        Y_pred = self.predict( self.X )
          
        # calculate new gradients      
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) + ( 2 * self.Lambda * (self.W - self.W_expt) ) ) / self.m     
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m
          
        # updating weights and checking the stop criteria 
        self.W = self.W - self.learning_rate * dW  
        self.b = self.b - self.learning_rate * db
        loss = np.sum(np.square(self.Y - Y_pred)) + self.Lambda * np.sum(np.square(self.W - self.W_expt))
        self.iteration += 1
        if (self.best_loss - self.tol) < loss:
            self.n_iter_no_change += 1
            self.W_best = self.W
            self.b_best = self.b
            if self.n_iter_no_change > self.num_iter_no_change:
                self.stop = False
        if (self.best_loss - self.tol) > loss:
            self.n_iter_no_change = 0
            self.best_loss = loss
            self.W_best = self.W
            self.b_best = self.b
        return self
      
    # Hypothetical function
    def predict( self, X ) :    
        return X.dot( self.W_best ) + self.b_best
    # Showing the final weights compared to the developed Ridge regression model
    def weights(self):
        Weights = pd.DataFrame(self.W)
        Weights.columns = ["Experts’ Model Coffecients"]
        Weights.insert(1, "Ridge Regression Coffecients", Ridge_reg_coefs, True)
        pd.set_option('display.colheader_justify', 'center')
        print(Weights)

# The results can vary based on the expert’s intuition about the importance of coefficients and the penalty term.
# Nevertheless, the impact of other parameters, such as learning rate, is not notable on the outcome.
model_expt = LinearRegressionWithKnowledge(learning_rate = 0.0005, Lambda = 0.9, W_expt = W)
model_expt.fit(X_train, y_train)
Weights = model_expt.weights()
