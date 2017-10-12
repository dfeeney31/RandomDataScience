import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from sklearn.utils import resample
%matplotlib inline
os.chdir('/Users/danielfeeney/Documents/DataScience/carinsurance')

df = pd.read_csv('train.csv')
df_majority = df[df.target==0]
df_minority = df[df.target==1]
len(df_majority)
len(df_minority)

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=573518,    # to match majority class
                                 random_state=123) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


X = df_upsampled.loc[:, (df_upsampled.columns != 'target') & (df_upsampled.columns != 'id')]
y = df_upsampled['target']

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

test = pd.read_csv('test.csv')
X_test = test.loc[:, (test.columns != 'id')]

preds = model.predict_proba(X_test)
#Save data into 2 DFs, concatenate
df1 = pd.DataFrame(preds[:,1], columns = ['target'])
IDs = test.loc[:, (test.columns == 'id')]
out = pd.concat([IDs,df1], axis = 1)
out.head()

out.to_csv('Submission_new.csv', index= False, header = True)
