# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:43:24 2018

@author: arunn
"""

## Import Required lib's
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mydata = pd.read_csv("housing.csv")

mydata.info()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
mydata[['total_bedrooms']]=imputer.fit_transform(mydata[['total_bedrooms']])

mydata.info()

iv = mydata.iloc[:,:9]
dv = mydata.iloc[:,-1]

iv = pd.get_dummies(iv,drop_first=True)

from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv_train = sc.fit_transform(iv_train)
iv_test = sc.transform(iv_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
regression = Sequential()
regression.add(Dense(7, input_dim = 12,kernel_initializer='uniform',activation='relu'))
regression.add(Dense(7,kernel_initializer='uniform',activation='relu'))
regression.add(Dense(7,kernel_initializer='uniform',activation='relu'))
regression.add(Dense(7,kernel_initializer='uniform',activation='relu'))
regression.add(Dense(1,kernel_initializer='uniform'))

regression.compile(optimizer='adam',loss = 'mean_squared_error',metrics=['accuracy'])

regression.fit(iv_train,dv_train, batch_size=5,nb_epoch=1000)

#y_pred = classifier.predict(iv_test)












