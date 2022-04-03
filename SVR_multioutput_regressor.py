# example of evaluating direct multioutput regression with an SVM model
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score,RepeatedKFold,RandomizedSearchCV


# define dataset
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
# Importing dataset
# Importing or Loading dataset
data = "C:/Users/Omomule Taiwo G/Desktop/PhD Project Dataset/Regression/Parkinsons/parkinsons_disease.csv"
df = pd.read_csv(data, delimiter=',')
# Inspect data
print('Inspect Data')
#print(df.head(50).to_string())
# Check Shape
print(df.shape)
# Statistical Summary
print(df.describe())


# Data Types
print(df.info())

# Each feature summary
for i in df:
    print(df[i].describe())
# Inspecting the statistical summary of the data shows the need for rescaling.

# Check Missing Values: To delete columns having missing values more than 30% or to input values--------
# Check missing values
df3 = df.isnull().sum()
print('Missing values in each feature \n:-------------------------------')
print(df3) # There are no missing values in the data

# Check feature relevance to the target through correlation matrix
df_corr = df.corr()
print('Feature Correlation Table')
print(df_corr.to_string)

# Through inspection, the parkinsons disease dataset is a multivariate dataset, having float and integer feature values
# with no missing values.

X = df.drop(['motor_UPDRS','total_UPDRS'],axis=1)
y = df[['motor_UPDRS','total_UPDRS']].copy()

cv_method = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
#model = BaggingRegressor(SVR())
# Support Vector Machines
SVM_clasf = SVR()
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_SVR = {'kernel':['rbf'],'C':np.linspace(0.1,1.0),
              'gamma':['scale','auto']} #np.linspace(0.1,1.0)}

#params_SVR = {'kernel':['rbf','poly','linear','sigmoid'],'C':np.linspace(0.1,1.0),
              #'gamma':['scale','auto'], 'degree':[2,3,4,5,6,7,8]} #np.linspace(0.1,1.0)}
# Using Random Search to explore the best parameter for the a SVM model
SVR_Grid = RandomizedSearchCV(SVM_clasf,params_SVR,scoring='r2',cv=cv_method)
# Fitting the parameterized model
SVR_Grid = MultiOutputRegressor(SVR_Grid)
SVR_Grid.fit(X,y)
# Print the best parameter values
print('SVR Best Parameter Values:', SVR_Grid.best_params_)
SVR_wrapper = SVR(**SVR_Grid.best_params_)
SVR_wrap = BaggingRegressor(SVR_wrapper)
#SVR = MultiOutputRegressor(SVR_wrap)
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(SVR_wrap)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
