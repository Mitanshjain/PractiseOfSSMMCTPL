# A Pipeline is a tool in scikit-learn that lets you
# combine multiple steps of machine learning into one single process.

# Instead of doing:-
                # Fill missing values
                # Encode data
                # Scale data
                # Train model

# A pipeline does everything in order automatically.

# Pipeline = A system that connects all preprocessing steps and the model together in one line.

# First we seprate the data (categorical and numerical).
# Categorical data == > SimpleImputer fill === > Encoding
# Numerical data == > SimpleImputer fill === > Standardize.

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('covid_toy - covid_toy.csv')
# print(df.head(5))

# Train_test_split the data.
x = df.drop(columns=['has_covid'])
y = df['has_covid']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Define the columns that need to be preprocessed.
categorical_features = ['gender','city','cough']
numerical_features = ['age','fever']
# So Our data divide in categorical and numerical columns.


# Create transformer
numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])



#  Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

print(preprocessor)