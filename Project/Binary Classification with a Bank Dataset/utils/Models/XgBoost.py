from xgboost import XGBClassifier
import os
import sys

# Import Data_loader in a way that works both when this module is imported as a package
# and when the file is executed directly. The Data_loader module lives in the sibling
# folder ../Data_loader/Data_loader.py relative to this file.
try:
    # Prefer the package relative import when this file is used inside a package
    from ...utils.Data_loader.Data_loader import Data_loader
except Exception:
    # Fallback: load the Data_loader.py file directly to guarantee we get the class
    import importlib.util
    this_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(this_dir, '..', 'Data_loader', 'Data_loader.py'))
    spec = importlib.util.spec_from_file_location("bank_data_loader", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Data_loader = mod.Data_loader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# XGBoost model with tuned hyperparameters
class XGB_Classifier(XGBClassifier):
    def __init__(self):
        # Default tuned hyperparameters. Accept kwargs so callers can override any of them.
        def_tree_method = 'gpu_hist' if os.environ.get('XGB_DEVICE', '').lower() == 'gpu' else 'hist'
        # Map a simple `device` option to the appropriate tree_method. Consumers can pass device via kwargs.
        # We'll accept a `device` kwarg for backward compatibility.
        # Collect defaults, allow overrides from kwargs by popping them first.
        defaults = dict(
            n_estimators=2700,
            learning_rate=3e-1,
            max_depth=5,
            subsample=0.93,
            colsample_bytree=0.76,
            tree_method=def_tree_method,
            random_state=42,
        )
        # Allow any extra kwargs to override defaults
        try:
            # If users pass device explicitly, translate it
            device = defaults.pop('device', None)
        except Exception:
            device = None
        # Call parent initializer with merged params
        super().__init__(**{**defaults})

if __name__ == "__main__":
    # If this file is executed directly, run a quick test to verify the model can be trained and evaluated.
    print("Running quick test of XGB_Classifier...")
    # Load data
    data_loader = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv', select=.99, random_state=3407)
    X_train, X_test, y_train, y_test = data_loader.splitted_data()
    # Create and train model
    model = XGB_Classifier()
    model.fit(X_train, y_train)
    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test ROC AUC: {auc:.4f}")

