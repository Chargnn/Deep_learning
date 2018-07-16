'''
Dataset:
    An experimental drug was tested on individuals from 13 to 100yo.
    The trial had 10000 participants.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv')
samples = dataset.iloc[:, 1:6].values
labels = dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
samples[:, 0] = label_encoder_X.fit_transform(samples[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
samples = onehotencoder.fit_transform(samples).toarray()

from sklearn.model_selection import train_test_split
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.5, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_samples = scaler.fit_transform(train_samples)
test_samples = scaler.transform(test_samples)

#########################

import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential([
        Dense(units=16, activation='relu', kernel_initializer='uniform'),   #Input layer
        Dense(units=32, activation='relu', kernel_initializer='uniform'),   #Hidden layer
        Dense(units=1, activation='sigmoid', kernel_initializer='uniform')  #Output layer
        ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_samples, train_labels, batch_size=10, epochs=100)

#######################

# model.summary()

























