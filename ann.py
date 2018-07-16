# Artificial Neural Network

#### Préparation des données
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

####

#### Construire réseau de neurones
import keras

from keras.models import Sequential
from keras.layers import Dense

# Initiation
classifier = Sequential()

classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))

classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

####

#### Entrainer reseau neurones

classifier.fit(X_train, y_train, batch_size=10, epochs=200)

####

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.6)

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.6)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#########################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)
precision = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
moyenne = precision.mean()



#####################################

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer, units):
    classifier = Sequential()
    
    classifier.add(Dense(units=units, activation="relu", kernel_initializer="uniform", input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=units, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    print("Optimizer=" + str(optimizer) + "units=" + str(units))
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size": [25, 50, 100],
              "epochs" : [100, 200, 400, 800],
              "optimizer": ["adam", "rmsprop"],
              "units": [2, 4, 8, 16]}

grid_search = GridSearchCV(estimator=KerasClassifier(build_fn=build_classifier), param_grid=parameters, scoring="accuracy",cv=10)

grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_






















