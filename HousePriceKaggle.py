##-------------------------------------------


##--------------------------------------------
import sys
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
from libHousePriceKaggle import featuresCleaning, Outliers, nearZeroVariance, preprocessBoxLog, split, RMSE

#-------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------

string = sys.argv

train_csv = string[1]
test_csv = string[2]

# Load train set and test set

data = pd.read_csv(train_csv, index_col = 0)
test = pd.read_csv(test_csv, index_col = 0)

X = data.iloc[:,:-1]    # train set without labels
Y = data.iloc[:,-1]     # labels

#---------------------------------------------------------------------------------------------------

# X_new is the concatenation of the train(without the labels) and test set. 
#In this way it is possible to preprocess in exactly the same form the two dataset.

X_new = pd.concat([X,test],axis=0)


# n is the total number of numerical(ordinal and not-ordinal) features

n = X_new.select_dtypes([np.number]).shape[1]


# convert categorical in binary features and fill NaNs in numerical feature 
# with the median of the column (the median return a better result when there are outliers)


X_new = featuresCleaning(X_new, n,"median")    



# remove the outliers respect to the labels(not respect the relative feature distribution) only of 
# the numerical (not-ordinal) features

index_ordinal = [0,3,4,14,16,17,18,19,20,21,22,23,25,30,32,34,35]
index_numerical = [1, 2, 5, 6, 7, 8,9, 10, 11, 12, 13, 15, 24, 26, 27, 28, 29, 31, 33]


outliers = {X_new.columns[1]:[935,1299],X_new.columns[2]:[250,314,336,707],
            X_new.columns[8]:[1299],X_new.columns[11]:[1299],
            X_new.columns[12]:[1299,2550],X_new.columns[15]:[524,1299,2550],
            X_new.columns[27]:[54],X_new.columns[28]:[496,584,1329],
            X_new.columns[29]:[198],"Price": [524,1183]}


X_new, Y = Outliers(X_new, Y, outliers)


# remove the feature with near-zero variance

X_new = nearZeroVariance(X_new, 0.0, n)



# transform all the numerical features using the boxcox transformation (or the log)
# to obtain less skewed features and less problem with outliers


X_new = preprocessBoxLog(X_new, n, 0.0, "log")


# scale all the features with robust_scale that is a better scaling  when there are a lot of outliers
# because it uses the median instead of the mean


X_new_scaled = pd.DataFrame(preprocessing.robust_scale(X_new), 
	index = X_new.index, columns = X_new.columns)


#----------------------------------------------------------------------------------------------------


# split the dataset into the original train and test set and compute the log of the output variable

x_train, y_train, Xtest  = split(X_new_scaled,Y,1,1, "log")


# subtract the mean of the output

m = y_train.mean()
y_train = y_train-m


# perform two parametric regressions with regularization(ridge and lasso), using cross-validation to tune 
# the alpha e lambda parameters, and choose the method that returns the smallest root mean squared error
# using a K-folds(5 folds) cross-validation(this is done with the RMSE function).

rmse = {}

# these two models use a different approach to control the overfitting and the complexity of the model.


# the ridge regression(L2 regularization) 
#regularizes with the sum of the square of the parameters(shrinkage of the parameters)

ridge = linear_model.RidgeCV(alphas = (1,5,10,15,20,30,40,50,100),fit_intercept=True, normalize=False, cv=5)
rmse["ridge"] = RMSE(ridge, x_train, y_train, 5)


# the lasso regression(L1 regularization) 
#regularizes with the sum of the module of the parameters(non-linear function and automatic feature selection)

lasso = linear_model.LassoCV(alphas = (0.0001,0.0005,0.001,0.01,0.1,1),fit_intercept=True, normalize=False, max_iter=1e4, cv=5)
rmse["lasso"] = RMSE(lasso, x_train, y_train, 5)


# predict and trasform the output prediction summing the mean and exponentiate


if rmse["ridge"] < rmse["lasso"]:
    
    
    ridge.fit(x_train, y_train)
    prediction = ridge.predict(Xtest)
    prediction = prediction+m
    prediction = np.exp(prediction)
    
    
else:
    
    lasso.fit(x_train, y_train)
    prediction = lasso.predict(Xtest)
    prediction = prediction+m
    prediction = np.exp(prediction)


# save the prediction in a csv file.

# on final submission I use Ridge Regression

ridge.fit(x_train, y_train)
prediction = ridge.predict(Xtest)
prediction = prediction+m
prediction = np.exp(prediction)

prediction = np.round(prediction,3)

pred = pd.DataFrame({"SalePrice": prediction}, index = test.index)
pred.to_csv("pred.csv")                                              # 0.11625

#---------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
