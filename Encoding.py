# What is encoding?
# Encoding is the process of converting categorical (text) data into numbers so that a machine learning model can understand it, Because ML model
# Understand numbers not text.

# There are four types of encoding are as follow:-
# 1. Label Encoding.
# 2. Ordinal Encoding.
# 3. OneHot Encoding.
# 4. Get_dummies Encoding.



# 1. Label Encoding:- If order matters → Use Label/Ordinal Encoding.
# Gives each category a number.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(4))

# print(df.isna().sum())

df = df.dropna() # It removes rows that contain missing values (NaN) from your dataset.
# print(df.isna().sum()) 


lb = LabelEncoder()

# df['cough'] = lb.fit_transform(df['cough'])
# df['gender'] = lb.fit_transform(df['gender'])
# df['has_covid'] = lb.fit_transform(df['has_covid'])
# df['has_covid'] = lb.fit_transform(df['has_covid'])
# df['city'] = lb.fit_transform(df['city'])

# print(df.head(4))


# One-Hot Encoding:-Creates new columns for each category. No number is bigger than another. No order problem.

ohe = OneHotEncoder(drop = 'first', sparse_output=False, dtype=np.int32)

df = ohe.fit_transform(df[['gender','cough','city','has_covid']])
# print(df) # so when we run this it gives array as you see that. We cannot use head or any another keyword with df because array have didnt keywords like head and all.
print(type(df)) # <class 'numpy.ndarray'> you will get this.


# So we can convert this array into DataFrame 
df_ohe = pd.DataFrame(
    df,
    columns=ohe.get_feature_names_out(['gender','cough','city','has_covid'])
)
# print(df_ohe.head(4))


# 3. Ordinal Encoding:-Used when categories have a clear order.

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(4))

df = df.drop(columns = [ 'city','age', 'fever'])
# print(df.head(2))

df_new = df.copy()  # In this df_new variable we copy the df.
oe = OrdinalEncoder(categories=[['Male','Female'],['Mild','Strong'],['No','Yes']])

oe_df = oe.fit_transform(df)
# print(oe_df) # So you will get array from this..

# So we convert this array into DataFrame
df_oe = pd.DataFrame(oe_df,columns=df_new.columns)
# print(df_oe.head(5))


# 4. Get_dummies:- pd.get_dummies() is a Pandas function that performs One-Hot Encoding.

# It converts text categories into numbers.

# Separate features and target

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(4))
X = df[['gender','cough','city']] # Model learns from X
y = df['has_covid'] # Model predicts y

# Encode only features
X_encoded = pd.get_dummies(X, drop_first=True, dtype='int32')

print(X_encoded.head(4))