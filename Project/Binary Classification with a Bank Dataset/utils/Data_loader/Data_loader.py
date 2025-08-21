import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.impute import KNNImputer
# This is very dangerous since it might kill the terminal
# imputer = KNNImputer(n_neighbors=1)
# # impute the missing values
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Therefore we might adopt dimensional reduction first and then conduct imputation
from sklearn.decomposition import PCA



class Data_loader():
    def __init__(self, name: str | None = None, data: pd.DataFrame | None = None):
        # read pd.csv
        if name is not None:
            self.data = pd.read_csv(name)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either 'name' or 'data' must be provided.")

        if 'y' in self.data.columns:
            self.data_character = "train"
            self.X = self.data.drop('y', axis=1)
            self.y = self.data['y']
        else:
            self.data_character = "test"
            self.X = self.data
            self.y = None

    def pipeline(self, *args, **kwargs):
        # a following pipeline of data processing
        for func in args:
            if callable(func):
                func(self, **(kwargs.get(func.__name__, {})))
        return self.X, self.y
    
    def preprocessing(self):
        df = self.data.copy()
        # Create a new column 'education_unknown'
        df['education_unknown'] = (df['education'] == 'unknown').astype(int)

        # Define the mapping for ordinal encoding
        education_map = {'primary': 1, 'secondary': 2, 'tertiary': 3, "unknown": np.nan}

        # Apply the mapping to the 'education' column
        # 'unknown' values will become NaN because they are not in the map
        df['education'] = df['education'].map(education_map)
        df['pdays_contacted'] = (df['pdays'] > -1).astype(int)
        df['month'] = df['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                               'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

        df = pd.get_dummies(df, columns=['job', 'marital', 'contact', 'poutcome'], drop_first=True)
        for col in ['default', 'housing', 'loan']:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0}) 

        df_X = self.X.copy()
        column = "education"
        n_components = 5
        neighbors = 7
        df1 = df_X.copy()
        df2 = df_X.copy()
        df2 = df2.drop(columns=[column])
        df1 = df1[column]
        pca = PCA(n_components=n_components)
        df2_reduced = pca.fit_transform(df2)
        df_reduced = pd.concat([pd.DataFrame(df2_reduced, columns=[f'PC{i+1}' for i in range(df2_reduced.shape[1])]), df1], axis=1)

        imputer1 = KNNImputer(n_neighbors=neighbors)
        df_reduced_1 = imputer1.fit_transform(df_reduced)
        # clip to the set {1, 2, 3}
        df_reduced_1 = np.clip(df_reduced_1, 1, 3)
        # Select the imputed 'education' column (the last column)
        imputed_education_values = df_reduced_1[:, n_components].astype(int)
        # Create a new DataFrame for the imputed values, preserving the original index from df_X
        df_reduced_education = pd.DataFrame(imputed_education_values, columns=['education'], index=df_X.index)
        df_X['education'] = df_reduced_education['education']

        self.X = df_X

    def output(self, *args, **kwargs):
        if args is None:
            self.pipeline(self.preprocessing)
            return self.data.X, self.data.Y

