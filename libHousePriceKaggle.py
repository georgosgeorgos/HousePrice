import time
import random
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# under this line there is the library that I use directly in the script.

#---------------------------------------------------------------------------------------------
# X_new is the concatenation of the train(without the labels) and test set. In this way it is possible to preprocess in exactly the same
# form the two dataset.
# n is the number of initial numerical features.


def featuresCleaning(X_new, n, fill = "mean"):
    
    '''
    input: X_new (DataFrame), n (integer), fill (string)
    
    action: create binary variables for the categorical(non numerical) features and fill the nan in the DataFrame 
            with the mean or the median of the relative column and control if all the nan are filled.
    
    output: X_new (Dataframe)
    '''
    
    X_new = pd.get_dummies(X_new)
    
    if fill == "median":
        
        
        X_new = X_new.fillna(X_new.median(axis=0),inplace=True)
        
    else:
        
        X_new = X_new.fillna(X_new.mean(axis=0),inplace=True)
        
    
    
    tot_nan_num = X_new.iloc[:,:n].isnull().sum().sum()
    tot_nan_cat = X_new.iloc[:,n:].isnull().sum().sum()
    
    if tot_nan_num != 0:
        
        print("Problem numeric features")
        
    if tot_nan_cat != 0:
        
        print("Problem categoric features")
        
    
    return X_new



def Outliers(X_new,Y, outliers):

    
    '''
    input: X_new (Dataframe), Y (Series), outliers (Dict)
    
    action: drop the outliers from the DataFrame and the output(only if the outlier is 
            in the original train set). The outliers are been choosen
            inspecting the scatter plots of the numerical variables.
    
    output: X_new (Dataframe), Y (Series)
    '''

    points = set()
    
    for feature in outliers.values():
        for outlier in feature: 
            if outlier < len(Y):
                points.add(outlier)
    try:

        X_new = X_new.drop(points)
        Y = Y.drop(points)
        
    except:
        
        print("outliers problem")
    
    return X_new, Y



def nearZeroVariance(X_new, t, n):
    
    '''
    input: X_new (DataFrame), t (integer), n (integer) 
    
    action: return the X_new DataFrame without the binary features with a variance smaller than 
    a given threshold
    
    output X_new (DataFrame)
    '''
    
    for feature in X_new.columns[n:]:

        if X_new[feature].var() <= t:

            del X_new[feature]
    
    return X_new



def preprocessBoxLog(X_new, n, t, c = "log"):

    '''
    input : X_new (DataFrame), n (integer), t (float),  c (string)
    
    action: perform a feature trasformation (on the original numerical features) using the boxcox/log/sqrt or simply the log,
            considering only the features with a skewness bigger(or smaller) than a certain threshold t
    
    output : X_new (DataFrame)
    '''
    
    
    if c == "box":
     
        for feature in X_new.columns[:n]:


            if skew(X_new[feature]) < -t or skew(X_new[feature]) > t:


                box = np.abs(skew(boxcox(X_new[feature]+1)[0]))
                log = np.abs(skew(np.log(X_new[feature]+1)))
                sq = np.abs(skew(np.sqrt(X_new[feature]+1)))

                g = boxcox(X_new[feature]+1)[1]

                cond = (g < 5 and g > -5)

                x = np.argmin([box, log , sq])  


                if x == 0 and  cond :

                    X_new[feature] = boxcox(X_new[feature]+1)[0]

                elif x == 1 or (x==0 and not cond) :

                    X_new[feature] = np.log(X_new[feature]+1)

                elif x == 2:

                    X_new[feature] = np.sqrt(X_new[feature]+1)
    
    else:
        
        for feature in X_new.columns[:n]:

            if skew(X_new[feature]) < -t or skew(X_new[feature]) > t:
        
                    X_new[feature] = np.log(X_new[feature]+1)
                   
    return X_new
            



def split(X_new, Y, f = 1, state = 1, g = "log"):
    
    '''
    input: X_new (DataFrame), Y (Series), f (float), state (int), g (string)
    
    action: split the DataFrame into the original train and test set. If f !=1 split also the original train set in a (sub)train set 
            and a cross-validation set. Also transform the labels with log or sqrt 
    
    output: x_train (DataFrame), x_cv (DataFrame), y_train (Series), y_cv (Series), x_test (DataFrame)
    '''

    s = len(Y)
    
    
    X = X_new.iloc[:s]
    
    x_train = X.sample(frac=f,random_state=state)
    x_cv = X.drop(x_train.index)
    
    if g == "sqrt":
        
        y_train = np.sqrt(Y[x_train.index])
        y_cv = np.sqrt(Y.drop(x_train.index))
        
    else:
        
        y_train = np.log(Y[x_train.index])
        y_cv = np.log(Y.drop(x_train.index))
    
    
    x_test = X_new.iloc[s:]

    if f != 1:

        return x_train, x_cv, y_train, y_cv, x_test

    else:

        return x_train, y_train, x_test





def RMSE(clf, x_train, y_train, k):
    
    '''
    input: clf (regression model), x_train (Dataframe), y_train (Series), k (integer)
    
    action: perform a k-fold cross-validation for a given model and return the mean of the rmse of the k subsets
    
    output: Rmse (float)
    '''
    
    while k <2:
        
        print("Please a number k bigger that 1 to perform a k-fold cross-validation")
        k = int(input())
    
    rmse = np.zeros(k)

    t = len(y_train)//k

    for j in range(k):

        x_cv = x_train[j*t:(j+1)*t]
        y_cv = y_train[j*t:(j+1)*t]

        x = x_train.drop(x_cv.index)
        y = y_train.drop(y_cv.index)

        clf.fit(x, y)
        mse = mean_squared_error(clf.predict(x_cv),y_cv)


        rmse[j] = np.sqrt(mse)
        
    Rmse = rmse.mean()

    return Rmse


# The functions that I use directly in my final script are concluded.
#-----------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------
# Under this line there is a "manual preprocessing": in practice I have written manually 
# the methods that I use to understand better what I'm doing. So more or less the relation is:

# pd.get_dummy ----> convertObjectNumeric

# pd.fillna    ----> objectNaN + numericNaNCleaner

# subroutineGeneralPreprocess ----> trasformation(box/log) and scaling (scale) 

# generalPreprocess ----> inside here there is the whole manual preprocess and it is possible 
# to decide the degree of the model(linear or polynomial) with the parameter d.

# Using this manual library I have obtained approx a Leaderboard score of 0.126.


def objectNaN(X):


    '''

    input: X (DataFrame)

    action: remove categorical features with a lot of nan and substitute the residuals nan with "NaN"

    output: X_obj (DataFrame)

    '''
    X_obj = X.select_dtypes([np.object])
    d = X_obj.columns[X_obj.isnull().sum()  > (X.shape[0])//20]
    X_obj = X_obj.drop(d,axis = 1)
    d = X_obj.columns[X_obj.isnull().sum()  > 0]

    for col in d:

        x = X_obj[col]

        key = "NaN"
        X_obj[col] = X_obj[col].fillna(key)

    return X_obj


def numericNaNCleaner(X):

    '''
    input: X (DataFrame)

    action: remove numerical features with a lot of nan and 
            fill the residuals nan with the column mean

    output: X_num (DataFrame)
    '''

    X_num = X.select_dtypes([np.number])
    d = X_num.columns[X_num.isnull().sum()  > (X.shape[0])//20]
    X_num = X_num.drop(d,axis = 1)
    X_num = X_num.fillna(X_num.mean(axis=0))
    
    return X_num


def convertObjectNumeric(X_obj, name):

    
    '''
    input: X_obj (DataFrame) ,name (string)

    action: clean categorical(non numerical) feature and create the binary features

    output: new (DataFrame
    '''

    vector = X_obj[name].values
    n = X_obj.shape[0]
    
    M = {s:np.zeros(n) for s in set(vector) if s not in ["NaN"]}
    
    for i in range(0,n):
        if vector[i] not in ["NaN"]:
    
            M[vector[i]][i] = 1
    
    new = pd.DataFrame(M, index = X_obj.index)
    
    return new



def generalPreprocess(X, value, d=1):


    '''
    input: X (DataFrame), value (float), d (int)

    action: whole preprocess and model complexity(linear of polynomial)

    output : X_new (DataFrame)
    '''

    X_new = pd.DataFrame()

    X_obj = objectNaN(X)
    z = X_obj.columns

    numeric = numericNaNCleaner(X)

    if d == 2: 

        poly = PolynomialFeatures(degree=d)
        p = poly.fit_transform(numeric)[:,1:]
        numeric = pd.DataFrame(p, index = numeric.index)
    
    w = numeric.columns

    for name in w:
    
        temp = subroutineGeneralPreprocess(numeric,name,value)
        X_new = pd.concat([X_new,temp], axis=1)

    for name in z:
    
        temp = convertObjectNumeric(X_obj,name)
        X_new = pd.concat([X_new,temp],axis=1)
        
    return X_new



def subroutineGeneralPreprocess(numeric, feature, value):

    '''
    input: numeric (DataFrame), feature (string), value(float)

    action: trasform(box/log) and scale the feature

    output: new (DataFrame)
    '''
     
    if skew(numeric[feature]) > value:
        
        
        box = np.abs(skew(boxcox(numeric[feature]+1)[0]))
        log = np.abs(skew(np.log(numeric[feature]+1)))
        sq = np.abs(skew(np.sqrt(numeric[feature]+1)))

        g = round(boxcox(numeric[feature]+1)[1])

        cond = (g < 5 and g > -5)

        x = np.argmin([box, log , sq])  

        if x == 0 and  cond :

            new = boxcox(numeric[feature]+1)[0]

        elif x == 1 or (x==0 and not cond) :

            new = np.log(numeric[feature]+1)
        else:

            new = np.sqrt(numeric[feature])
    else:
            new =  numeric[feature]
            
    new = preprocessing.scale(new)
                   
    new = pd.DataFrame({feature : new}, index = numeric.index)
    return new



#------------------------------------------------------------------------------------------------
# this is a function that implements a manual feature selection using a greedy algorithm with
# a simple linear regression. With this method, one by one, the feature that return the smaller
# rmse is added. And you continue until it is possible to obtain a smaller rmse.
# Obviously with this method it is possible to obtain only a local minimum(to obtain the absolute
# minimun it should necessary to try all the possible feature combination and this is not possible
# for a big number of features).
# With this method I have obtained a LeaderBoard score of approx 0.128.


def featureSelection_sk(X_train, Y_train):

    '''
    input: X_train (DataFrame), Y_train (Series)

    action: perform a greedy algorithm to obtain a local optimal feature selection with a standard
            linear regression

    output: features (list)

    '''

    features = []
    X_train = X_train.values
    
    n,m = X_train.shape
    oldMin = 1
    count = 0
    while len(features) < m:
        
        RMSE = []  
        
        count +=1
        if count%20 == 0:
            print("pause")
            time.sleep(10)

        for j in range(m):
            if j not in features:

                index = features+[j]

                clf = linear_model.LinearRegression()
                rmse = np.sqrt(-cross_val_score(clf, X_train[:,index], Y_train, scoring="neg_mean_squared_error", cv = 5)).mean()
                
                RMSE.append(rmse)

            else:
                RMSE.append(1)
        
        RMSE = np.array(RMSE)
        newMin = RMSE.min()
        indexMin = np.argmin(RMSE)
        
        if newMin > oldMin:
            print(features)
            
            print("cv error",min(RMSE))
            return features

        oldMin = newMin
        features.append(indexMin)
        print("cv error",newMin)
        
    return features

#-------------------------------------------------------------------------------------------------------