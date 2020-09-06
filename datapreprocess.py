healthy_pd = create_dataset(healthy_path,healthy,0)
diseased_pd = create_dataset(diseased_path,diseased,1)
frames=[healthy_pd, diseased_pd]
dataset=pd.concat(frames)
dataset.shape
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.head(20)
#export this dataset to csv file
dataset.to_csv('dataset.csv')
!cp '/content/dataset.csv' '/content/drive/My Drive/project/dataset.csv'
import numpy as np
import pandas as pd
import os
import string
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from tqdm import tqdm_notebook
import math
dataset = pd.read_csv("/content/drive/My Drive/project/dataset.csv")
y = np.array(dataset['output'])
print(y)
q=y
c=0
t=0
for each in y:
  t=t+1
  if each==0:
    c=c+1
print(c/t)
dataset=dataset.drop(['output'],axis=1)
X = dataset.iloc[:,1:]
from sklaearn.model_selection import train_test_split
# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)
#X_train.head(5)
from sklearn import preprocessing
# Feature scaling
scaler = preprocessing.StandardScaler().fit(X_train)
scaler
scaler.mean_
scaler.scale_
X_train=scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
scaler
scaler.mean_
scaler.scale_
X_test=scaler.transform(X_test)


#from sklearn.preprocessing import StandardScaler
print(type(X_train))
print(type(X_test))
X_train
