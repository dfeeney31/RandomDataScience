import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

df = pd.read_csv('train.csv')

X = df.loc[:, (df.columns != 'target') & (df.columns != 'id')]
y = df['target']

X = df.loc[:, (df.columns != 'target') & (df.columns != 'id')]
y = df['target']
x_train = np.array(X)
y_train = np.array(y)
x_train = x_train[0:100000,0:57]
y_train = y_train[0:100000]

#Input dimension and output dimension to build model. Takes trial and error usually.
in_dim = X.shape[1]
#Sequential layer. Output of 128 nodes. Activation. Another dense layer. Activation that is sigmoid. Then compile
# create model
model = Sequential()
model.add(Dense(12, input_dim= in_dim, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = np.array(X)
y_train = np.array(y)

model.fit(x_train, y_train, batch_size = 50)

#Read in testing data
test = pd.read_csv('test.csv')
X_test = test.loc[:, (test.columns != 'id')]
x_test = np.array(X_test)

#Make predictions on the testing data
predicts = model.predict(x_test)
df1 = pd.DataFrame(predicts, columns = ['target'])
idout = IDs = test.loc[:, (test.columns == 'id')]
out = pd.concat([idout,df1], axis = 1)
