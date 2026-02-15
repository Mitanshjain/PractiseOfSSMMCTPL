# StandardScaler standardizes data around mean 0.
# Data is normally distributed
# It transforms data so that:-
# Mean = 0
# Standard Deviation = 1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(3))



# print(df.isna().sum()) # to find the missing value. so there is column with the name of fever has 10 missing value.

df['fever'] = df['fever'].fillna(0) # we can fill missing value with way.
# print(df.isna().sum()) 





# another one is SimpleImputer.
si = SimpleImputer(strategy='mean')  # This is the next way to fill missing value.
df[['fever']] = si.fit_transform(df[['fever']])

df['gender'] = df['gender'].map({"Male":0 , "Female":1})
df['cough'] = df['cough'].map({"Mild":0 , "Strong":1})

df = df.drop(columns = ['city'])

x = df.drop(columns=['has_covid'])  # Target Column
y = df['has_covid'] #Independent column.

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(df.head(2))




# By using this np.round(x_train.describe(),2) you can see our previous data before train the data.
# print(np.round(x_train.describe(),2))

# So we are going to do standard scaler.
sc = StandardScaler()  # making object of standard scaler.


# In our num_cols we numerical data. but we can also apply on categorical as well.
# Yes  — you also train on categorical features, but only after converting them into numbers.

# Make variable x_train_sc in which we perform fit_transform.
x_train_sc = sc.fit_transform(x_train)
# And always apply fit_transform on training data.

# print(x_train_sc)
# If i print x_train_sc it gives me array.
# when i run this x_train_sc it gives error as convert string to float like in our training datasets we have gender column so absolutly we have categorical data
# If we want to run this first we convert this gender column to numerical dataset.


x_train_new = pd.DataFrame(x_train_sc, columns=x_train.columns)

print(np.round(x_train_new.describe(),2))
# So this will gives mean as 0 and standard deviation as 1.
