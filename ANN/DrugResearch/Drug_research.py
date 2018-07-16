'''
Dataset:
    An experimental drug was tested on individuals from 13 to 100yo.
    The trial had 2100 participants. Half were under 65yo, half were over 65yo.
    95% of patientes 65 or older experienced side effects
    95% of patients under 65 experienced no side effects
'''
# Creating the dataset 
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)
    
for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

#########################

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

model = Sequential([
        Dense(16, input_shape=(1,), activation='relu'), #Input layer
        Dense(32, activation='relu'),                   #Hidden layer
        Dense(2, activation='softmax')                  #Output layer
        ])

model.summary()

model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(scaled_train_samples, train_labels, batch_size=10, validation_split=0.4, epochs=20, verbose=2)

########################

























