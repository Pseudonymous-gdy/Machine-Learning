import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

class Data_loader():
    def __init__(self, name: str| None=None, df: pd.DataFrame | None=None, select: float = 0.1, random_state: int =42):
        '''select is the proportion we select from the original df for effective investigation on the subsets.'''
        if name is not None:
            self.name = name
            self.df = pd.read_csv(name)
        if df is not None:
            self.df = df
            self.name = "DataFrame"
        self.__check__()
        self.df = self.df.sample(n=int(len(self.df) * select), random_state=random_state).reset_index(drop=True)
        self.X = self.df.drop(columns=['y'])
        self.y = self.df['y']

    def __check__(self):
        '''
        Check since name and df are mutually exclusive.
        '''
        if (self.name is not None) and (self.df is not None):
            if self.name != "DataFrame":
                if not self.df.equals(pd.read_csv(self.name)):
                    raise ValueError("Name and Dataframe mismatch.")
        else:
            pass

    def merge_columns(self):
        '''
        A copied column merge function in preprocessing.
        '''
        dataframe = self.df.copy()
        # life status
        dataframe["socioeconomic_status"] = dataframe["job"] + " " + dataframe["education"]
        dataframe["age_group"] = pd.cut(dataframe["age"],
                                        bins=[0, 25, 45, 65, 100],
                                        labels=["young", "young adult", "adult", "senior"])
        dataframe["age_marital"] = dataframe["age_group"].astype(str) + " " + dataframe["marital"]

        # contact-related information
        dataframe["last_contact_info"] = dataframe["contact"] + " " + dataframe["day"].astype(str) + " " + dataframe["month"]
        dataframe["contacted"] = (dataframe["pdays"] != -1).astype(int)
        dataframe["contact_outcome"] = dataframe["previous"].astype(str) + " " + dataframe["poutcome"]
        dataframe["total_contacts"] = dataframe["campaign"] + dataframe["previous"]

        # credit and loan information
        dataframe["credit_info"] = dataframe["default"] + " " + dataframe["housing"] + " " + dataframe["loan"]
        self.df = dataframe.copy()
        self.X = self.df.drop(columns=['y'])
    
    def feature_selection(self):
        '''
        Select possible effective columns. A method is by LogisticRegression
        '''
        data_no_low_importance = self.df.copy()
        X_df = data_no_low_importance.drop(columns=['y'])
        X_num_df = X_df.select_dtypes(include=[np.number])
        X_cat_df = X_df.select_dtypes(exclude=[np.number])

        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_encoded = oe.fit_transform(X_cat_df)

        ss = StandardScaler()
        X_num_scaled = ss.fit_transform(X_num_df)

        X_processed = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        all_features = X_num_df.columns.tolist() + X_cat_df.columns.tolist()

        y = data_no_low_importance['y']

        # Use a solver that supports 'l1' penalty
        selector = SelectFromModel(LogisticRegression(penalty="l1", C=.01, solver='liblinear', random_state=42))
        
        X_new = selector.fit_transform(X_processed, y)

        # To see the selected features
        selected_features = np.array(all_features)[selector.get_support()]
        print("Selected features:", selected_features.tolist())
        print(X_processed.shape, "->", X_new.shape)
        self.X = self.df[selected_features.tolist()]

    def splitted_data(self):
        self.merge_columns()
        self.feature_selection()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state = 42, stratify=self.y)
        # Ignore unknown categories during transform to avoid errors when X_test
        # contains categories not seen in X_train (common in small samples).
        oe = OneHotEncoder(handle_unknown='ignore')
        X_train = oe.fit_transform(X_train)
        X_test = oe.transform(X_test)
        ss = StandardScaler(with_mean=False)
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dl = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv')
    X_train, X_test, y_train, y_test = dl.splitted_data()
    print(X_train, X_test, y_train, y_test)
