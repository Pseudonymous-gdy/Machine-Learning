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
from typing import Literal, Any
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class RandomForest_Classifier(RandomForestClassifier):
    def __init__(
        self,
        n_estimators: int = 500,
        *,
        criterion: Literal['gini', 'entropy', 'log_loss'] = 'log_loss',
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float | Literal['sqrt', 'log2'] = 'sqrt',
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Any = None,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ) -> None:
        # Call parent initializer with the sanitized arguments
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, *args, **kwargs):
        """Fit and remember training labels for OOB evaluation convenience.

        Stores a copy of the training labels in self._train_y so oob_roc_auc can
        be called without passing y each time.
        """
        # Call parent fit
        res = super().fit(X, y, *args, **kwargs)
        try:
            self._train_y = np.asarray(y)
        except Exception:
            # fallback: do not block training if conversion fails
            self._train_y = None
        return res

    def oob_roc_auc(self, y: np.ndarray | None = None) -> float:
        """Compute ROC AUC using out-of-bag predictions.

        If y is provided, it will be used as the true labels; otherwise the
        labels recorded during `fit` will be used. Raises ValueError if OOB
        predictions are not available or true labels are missing.
        """
        if getattr(self, 'oob_decision_function_', None) is None:
            raise ValueError('oob_decision_function_ is not available. Make sure the forest was trained with oob_score=True')

        oob_pred = self.oob_decision_function_
        # Determine probability for positive class
        if getattr(oob_pred, 'ndim', None) == 2 and oob_pred.shape[1] > 1:
            oob_proba = oob_pred[:, 1]
        else:
            # shape (n_samples,) or similar
            oob_proba = oob_pred

        true_y = None
        if y is not None:
            true_y = np.asarray(y)
        elif getattr(self, '_train_y', None) is not None:
            true_y = self._train_y

        if true_y is None:
            raise ValueError('True labels are required to compute OOB ROC AUC; pass y to oob_roc_auc or call fit(X, y) before.')

        # Ensure shapes align
        if len(true_y) != len(oob_proba):
            raise ValueError(f'Length mismatch between true labels ({len(true_y)}) and OOB predictions ({len(oob_proba)}).')

        return float(roc_auc_score(true_y, oob_proba))


if __name__ == "__main__":
    dl = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv', select=.1, random_state=3407)
    dl.merge_columns()
    dl.feature_selection()
    dl.encode_and_scale()
    X_train, y_train = dl.X, dl.y
    # Create and train model
    # enable out-of-bag scoring if you want OOB ROC AUC
    model = RandomForest_Classifier(oob_score=True, n_jobs=-1)
    model.fit(X_train, y_train)
    # Predict and evaluate
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
    # auc = roc_auc_score(y_test, y_pred_proba)
    # print(f"Test ROC AUC: {auc:.4f}")
    # If OOB was enabled, compute ROC AUC on out-of-bag predictions (training data)
    try:
        oob_auc = model.oob_roc_auc()
    except Exception as e:
        print('OOB ROC AUC not available:', e)
    else:
        print(f"OOB ROC AUC: {oob_auc:.4f}")