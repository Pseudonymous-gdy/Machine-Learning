import numpy as np
import scipy.sparse as sp
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
        # placeholders for fitted transformers/selectors for reuse on test data
        self.oe = None
        self.ss = None
        self.selector = None
        self.selected_features = None
        self.num_cols = None
        self.cat_cols = None

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
        # If 'y' exists (training), drop it for X; otherwise keep all columns for test-time
        if 'y' in self.df.columns:
            self.X = self.df.drop(columns=['y'])
        else:
            self.X = self.df.copy()
    
    def feature_selection(self):
        '''
        Select possible effective columns. A method is by LogisticRegression
        '''
        data_no_low_importance = self.df.copy()
        X_df = data_no_low_importance.drop(columns=['y'])
        X_num_df = X_df.select_dtypes(include=[np.number])
        X_cat_df = X_df.select_dtypes(exclude=[np.number])

        # store numeric / categorical column lists
        self.num_cols = X_num_df.columns.tolist()
        self.cat_cols = X_cat_df.columns.tolist()

        # fit encoders/scalers on training data and keep them for later transform
        self.oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # ensure 2D for single categorical column
        if isinstance(X_cat_df, pd.Series):
            X_cat_df = X_cat_df.to_frame()
        X_cat_encoded = self.oe.fit_transform(X_cat_df)

        self.ss = StandardScaler()
        X_num_scaled = self.ss.fit_transform(X_num_df)

        X_processed = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        all_features = self.num_cols + self.cat_cols

        y = data_no_low_importance['y']

        # Use a solver that supports 'l1' penalty
        self.selector = SelectFromModel(LogisticRegression(penalty="l1", C=.013, solver='liblinear', random_state=42))
        X_new = self.selector.fit_transform(X_processed, y)

        # To see the selected features
        selected_features = np.array(all_features)[self.selector.get_support()]
        self.selected_features = selected_features.tolist()
        print("Selected features:", self.selected_features)
        print(X_processed.shape, "->", X_new.shape)
        # set self.X to a dataframe of selected features
        self.X = self.df[self.selected_features].copy()

    def encode_and_scale(self):
        if self.selected_features is None or self.oe is None or self.ss is None or self.selector is None:
            raise ValueError("feature_selection must be run on training data before calling encode_and_scale on new data")

        # derive numeric and categorical slices from stored column lists
        X_df = self.df.copy()
        # if 'y' present, drop it
        if 'y' in X_df.columns:
            X_df = X_df.drop(columns=['y'])

        X_num_df = X_df[self.num_cols]
        X_cat_df = X_df[self.cat_cols]

        # ensure 2D for single-column slices
        if isinstance(X_cat_df, pd.Series):
            X_cat_df = X_cat_df.to_frame()
        if isinstance(X_num_df, pd.Series):
            X_num_df = X_num_df.to_frame()

        X_cat_encoded = self.oe.transform(X_cat_df)
        X_num_scaled = self.ss.transform(X_num_df)

        X_processed = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        X_selected = self.selector.transform(X_processed)

        # if selector returns sparse matrix, convert to dense array
        if hasattr(X_selected, 'toarray'):
            X_selected = X_selected.toarray()

        # store as DataFrame for downstream code that expects DataFrame-like API
        self.X = pd.DataFrame(X_selected, columns=self.selected_features)

    def splitted_data(self):
        # Prepare merged columns first
        self.merge_columns()

        # Work on a copy to allow restoring self.df later
        full_df = self.df.copy()

        if 'y' not in full_df.columns:
            raise ValueError("Dataframe must contain 'y' column for splitted_data")

        # Split first to avoid leaking test labels into feature selection/encoders
        train_df, test_df = train_test_split(full_df, test_size=0.015, random_state=42, stratify=full_df['y'])

        # Fit feature selection and encoders on training portion only
        self.df = train_df.reset_index(drop=True)
        self.feature_selection()
        # encode_and_scale will populate self.X based on selected_features
        self.encode_and_scale()
        X_train = self.X
        y_train = self.df['y']

        # Now transform the test portion using the fitted transformers/selector
        self.df = test_df.reset_index(drop=True)
        self.encode_and_scale()
        X_test = self.X
        y_test = self.df['y']

        # Restore full dataframe into the loader instance
        self.df = full_df.reset_index(drop=True)
        # Also keep X/y consistent with original interface
        self.X = full_df.drop(columns=['y'])
        self.y = full_df['y']

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dl = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv')
    X_train, X_test, y_train, y_test = dl.splitted_data()
    print(X_train, X_test, y_train, y_test)
