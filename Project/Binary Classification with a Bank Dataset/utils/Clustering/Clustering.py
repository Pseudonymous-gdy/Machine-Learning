from joblib import Memory
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Any, Literal, Optional
import os
import scipy.sparse as sp
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

class clustering(OPTICS):
    def __init__(
        self,
        *,
        min_samples: int = 5,
        max_eps: float = np.inf,
        metric: str | Callable[..., Any] = "minkowski",
        p: float = 2.0,
        metric_params: Optional[dict] = None,
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Optional[int] = None,
        algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto",
        leaf_size: int = 30,
        memory: Optional[Memory] | str = None,
        n_jobs: Optional[int] = None,
        dataframe: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            p=p,
            metric_params=metric_params,
            cluster_method=cluster_method,
            eps=eps,
            xi=xi,
            predecessor_correction=predecessor_correction,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            leaf_size=leaf_size,
            memory=memory,
            n_jobs=n_jobs,
        )
        self.df = dataframe
        # Also expose the parameter as an attribute so sklearn's get_params
        # and other code that expects the __init__ args to become attributes
        # will not raise AttributeError.
        self.dataframe = dataframe


    def plotting(self):
        ...

    def cluster(
        self,
        X: Optional[pd.DataFrame | np.ndarray] = None,
        *,
        method: Literal['optics', 'kmeans'] = 'optics',
        n_clusters: Optional[int] = None,
        label_col: str = 'cluster',
        compute_scores: bool = True,
        random_state: Optional[int] = None,
        true: Optional[np.ndarray|pd.DataFrame] = None
    ) -> dict:
        """Run clustering and attach labels to self.df[label_col].

        Args:
            X: optional data to cluster (DataFrame or ndarray). If None, uses
               numeric columns from `self.df` (must exist).
            method: 'optics' to use this OPTICS instance, or 'kmeans' to run KMeans.
            n_clusters: required when method='kmeans'.
            label_col: column name to write labels into `self.df`.
            compute_scores: whether to compute silhouette and NMI (when possible).
            random_state: passed to KMeans when used.

        Returns:
            dict with keys: 'labels', 'silhouette', 'nmi' (values or None).
        """
        # Resolve input matrix
        if X is None:
            if self.df is None:
                raise ValueError('No data provided: set self.df or pass X')
            if isinstance(self.df, pd.DataFrame):
                X_use_df = self.df.select_dtypes(include=[np.number])
                X_use = X_use_df.values
            else:
                # assume array-like
                # handle sparse
                if sp.issparse(self.df) or hasattr(self.df, 'toarray'):
                    X_use = self.df.toarray()
                else:
                    X_use = np.asarray(self.df)
        else:
            if isinstance(X, pd.DataFrame):
                X_use_df = X.select_dtypes(include=[np.number])
                X_use = X_use_df.values
            else:
                # handle sparse inputs
                if sp.issparse(X) or hasattr(X, 'toarray'):
                    X_use = X.toarray()
                else:
                    X_use = np.asarray(X)

        # Basic sanitation: ensure numeric float matrix, fill NaNs, scale features
        try:
            from sklearn.preprocessing import StandardScaler
        except Exception:
            StandardScaler = None

        # Convert to float array if necessary
        if not np.issubdtype(getattr(X_use, 'dtype', np.array(X_use).dtype), np.number):
            X_use = X_use.astype(float)

        # If there are NaNs, replace with column means
        if np.isnan(X_use).any():
            col_mean = np.nanmean(X_use, axis=0)
            inds = np.where(np.isnan(X_use))
            X_use[inds] = np.take(col_mean, inds[1])

        # Standardize features when possible to help k-means/optics
        if StandardScaler is not None:
            try:
                scaler = StandardScaler()
                X_use = scaler.fit_transform(X_use)
            except Exception:
                # if scaling fails, continue with raw features
                pass

        # Run chosen clustering
        labels = None
        if method == 'optics':
            # use the inherited OPTICS estimator: fit returns self with labels_
            self.fit(X_use)
            labels = getattr(self, 'labels_', None)
            if labels is None:
                raise RuntimeError('OPTICS did not produce labels_')
        elif method == 'kmeans':
            if n_clusters is None:
                raise ValueError('n_clusters must be provided for kmeans')
            # Use more robust KMeans parameters and multiple inits
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, init='k-means++')
            labels = km.fit_predict(X_use)
        else:
            raise ValueError(f'Unknown clustering method: {method}')

        labels = np.asarray(labels)

        # Provide diagnostic cluster counts
        try:
            unique, counts = np.unique(labels, return_counts=True)
            cluster_counts = dict(zip(unique.tolist(), counts.tolist()))
        except Exception:
            cluster_counts = None

        # Ensure self.df exists and has correct length to assign labels
        if self.df is None:
            # create a minimal dataframe from X_use
            cols = [f'c{i}' for i in range(X_use.shape[1])]
            self.df = pd.DataFrame(X_use, columns=cols)

        # If self.df has more rows than X_use (e.g., original df kept non-numeric cols),
        # try to align length by using the numeric-subset index if available.
        try:
            if isinstance(self.df, pd.DataFrame):
                if len(self.df) == len(labels):
                    self.df[label_col] = labels
                else:
                    # try to write into numeric-only slice if possible
                    numeric_idx = self.df.select_dtypes(include=[np.number]).index
                    if len(numeric_idx) == len(labels):
                        self.df.loc[numeric_idx, label_col] = labels
                    else:
                        raise ValueError('Cannot align labels to self.df: row count mismatch')
            else:
                raise ValueError('self.df must be a DataFrame to attach labels')
        except Exception:
            # Bubble up as ValueError with context
            raise

        silhouette = None
        nmi = None
        if compute_scores:
            try:
                # For silhouette, need at least 2 clusters (excluding noise label -1)
                mask = labels != -1
                if np.unique(labels[mask]).size >= 2:
                    silhouette = float(silhouette_score(X_use[mask], labels[mask]))
                # If the dataframe has a true label column 'y' compute NMI
                if len(true) == len(labels):
                    nmi = float(normalized_mutual_info_score(true, labels))
            except Exception:
                # keep scores as None if computation fails
                silhouette = silhouette
                nmi = nmi

        return {'labels': labels, 'silhouette': silhouette, 'nmi': nmi, 'counts': cluster_counts}

if __name__ == "__main__":
    dl = Data_loader(name='Project/Binary Classification with a Bank Dataset/train.csv', select=.01, random_state=3407)
    dl.merge_columns()
    dl.feature_selection()
    dl.encode_and_scale()
    X, y = dl.X, dl.y
    # Avoid concatenating sparse X and y; pass X directly to cluster
    clusterer = clustering(dataframe=None)
    result = clusterer.cluster(X=X, method='kmeans', n_clusters = 8, random_state=42, true=y)
    print(result)