import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(3))
df = df.drop(columns=['city'])
# print(df.head(3))

# Convert categorical to numerical column
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['cough'] = df['cough'].map({'Mild':0,'strong':1})

# print(df.head(7))

x = df.drop(columns=['has_covid'])
y = df['has_covid']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(np.round(x_train.describe(),2)) # you can see our previous data before train the data.

# So we are going to do MinMax Scaler.
# First we create a object for minmax scaler.
mn = MinMaxScaler()

x_train_mn = mn.fit_transform(x_train)
# print(x_train_mn)
# So this will give as array.

# So we convert this array into DataFrame

x_train_new = pd.DataFrame(x_train_mn,columns=x_train.columns)
print(np.round(x_train_new.describe(),2))
# So we get max is 1 and another one gets nearby 0 value.