import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle
import mediapipe as mp
import cv2
import csv
import os
import time
import numpy as np
import statistics
# to train
df=pd.read_csv('coords.csv')
print(df.head())

x=df.drop('class',axis=1)
y=df['class']
print(x)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
print(y_test)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),

    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
fit_models = {}
print("training")
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
print("trained")

print(fit_models)

fit_models['rc'].predict(X_test)
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

print(fit_models['rf'].predict(X_test))
print(y_test)


with open('smart_ai3.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)