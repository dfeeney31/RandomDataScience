import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')
X = df.loc[:, (df.columns != 'target') & (df.columns != 'id')]
y = df['target']

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
