from sklearn.linear_model import LogisticRegression
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

class Logistic_Regression(LogisticRegression):
    # Explicit parameters (no *args) so scikit-learn can inspect them.
    def __init__(
        self,
        penalty: str = 'l2',
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 0.05,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight=None,
        random_state: int | None = 42,
        solver: str = 'lbfgs',
        max_iter: int = 10000,
        multi_class: str = 'auto',
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int | None = None,
        l1_ratio: float | None = None,
    ) -> None:
        # Call parent with the explicit parameters (project defaults set above)
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

if __name__ == "__main__":
    dl = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv', select=.1, random_state=3407)
    X_train, X_test, y_train, y_test = dl.splitted_data()

    model = Logistic_Regression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred = np.round(y_pred)
    print("AUC:", roc_auc_score(y_test, y_pred))
    print("Trained AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))