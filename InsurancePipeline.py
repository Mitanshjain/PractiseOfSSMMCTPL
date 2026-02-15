import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = pd.read_csv('insurance - insurance.csv')
# print(df.head(5))

# Train_test_split
x = df.drop(columns=['charges'])
y = df['charges']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Define the columns that need to be preprocessed.
categorical_features = ['sex','smoker','region']
numerical_features = ['age','children']
# So our data divide in categorical and numerical

# create transformer
numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

#  Combine transformers
preproccesor = ColumnTransformer(
    transformers=[
        ('num',numerical_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

print(preproccesor)