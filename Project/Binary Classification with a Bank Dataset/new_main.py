from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.Models.XgBoost import XGB_Classifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from utils.Data_loader.Data_loader import Data_loader

# ----------------------------------------------------------------
# 1. Data Preprocessing
# ----------------------------------------------------------------
df = pd.read_csv("Project/Binary Classification with a Bank Dataset/train.csv")
dl = Data_loader(df=df.copy())

dl.merge_columns()
dl.feature_selection()
dl.df = df.copy()
dl.merge_columns()
dl.encode_and_scale()
X, X_test, y, y_test = dl.splitted_data()


# ----------------------------------------------------------------
# 2. Data Training
# ----------------------------------------------------------------
model = XGB_Classifier(eval_metric="auc")
model.fit(X, y)

y_pred = model.predict_proba(X_test)[:, 1]
print("Roc-Auc Score:", roc_auc_score(y_test, y_pred))


# ----------------------------------------------------------------
# 3. Data Fitting
# ----------------------------------------------------------------
df = pd.read_csv("Project/Binary Classification with a Bank Dataset/test.csv")
dl.df = df.copy()
dl.merge_columns()
dl.encode_and_scale()
X = dl.X

out_path = 'Project/Binary Classification with a Bank Dataset/new_submission.csv'
y_pred = model.predict_proba(X)[:, 1]
submission = pd.DataFrame({'id': df['id'], 'y': y_pred})
submission.to_csv(out_path, index=False)
print(f'Submission saved to {out_path}')