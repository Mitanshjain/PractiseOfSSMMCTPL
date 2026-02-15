# Train_test_split
# We split data to check whether our model actually learned something — or is just memorizing.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(3))

# print(df.isna().sum())


df['fever'] = df['fever'].fillna(0)
# print(df.isna().sum())

df['gender'] = df['gender'].map({'Female':0,'Male':1})
df['cough'] = df['cough'].map({'Mild':0,'Strong':1})

df = df.drop(columns=['city'])

# print(df.head(4))


x = df.drop(columns=['has_covid'])
y = df['has_covid']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)