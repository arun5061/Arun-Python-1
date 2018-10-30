# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 08:04:30 2018

@author: arunn
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Churn_Modelling.csv')

iv = dataset.iloc[:,3:13]
dv = dataset.iloc[:,13]

iv = pd.get_dummies(iv,drop_first=True)

iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)

sc = StandardScaler()
iv_train = sc.fit_transform(iv_train)
iv_test = sc.transform(iv_test)

#classifier = Sequential()
#classifier.add(Dense(6, input_dim = 11,kernel_initializer='uniform',activation='relu'))
#classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
#classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

#classifier.fit(iv_train,dv_train, batch_size=5,nb_epoch=100)

#y_pred = classifier.predict(iv_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)

#accuracies = cross_val_score(estimator=classifier,X = iv_train, y = dv_train, cv = 2,n_jobs = 1 )

#Hypetunning parameters

params = {
        'batch_size': [25, 32],
        'nb_epoch' : [100, 500]
        'optimizer' : ['adam', 'rmsprop']
        }

grid_search = GridSearchCV(estimator = build_classifier,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(iv_train,dv_train)
grid_search.best_params_
grid_search.best_score_


