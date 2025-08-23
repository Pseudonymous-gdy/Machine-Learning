import argparse
import os
from utils.Data_loader.Data_loader import Data_loader
from utils.Models.Random_Forest import RandomForest_Classifier
from utils.Models.Logistic_Regression import Logistic_Regression
from utils.Models.XgBoost import XGB_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd


def main(full: bool = False, select: float | None = None, out_path: str = 'Project/Binary Classification with a Bank Dataset/submission.csv'):
	# Safety defaults
	if full:
		select_val = 1.0 if select is None else select
		rfc_trees = 2700
		xgb_trees = 2700
		stack_n_jobs = -1
		cv = 5
	else:
		select_val = 0.05 if select is None else select
		rfc_trees = 200
		xgb_trees = 200
		stack_n_jobs = 1
		cv = 5

	print(f"Running with select={select_val}, rfc_trees={rfc_trees}, xgb_trees={xgb_trees}, stack_n_jobs={stack_n_jobs}")

	dl = Data_loader(name="Project/Binary Classification with a Bank Dataset/train.csv", select=select_val)
	X_train, X_test, y_train, y_test = dl.splitted_data()

	estimators = [
		('rfc', RandomForest_Classifier(n_estimators=rfc_trees, n_jobs=1)),
		('lr', Logistic_Regression()),
		('xgb', XGB_Classifier(n_estimators=xgb_trees, n_jobs=1)),
	]

	stack = StackingClassifier(estimators=estimators, final_estimator=Logistic_Regression(), cv=cv, n_jobs=stack_n_jobs)

	# Fit stacking model on preprocessed features
	stack.fit(X_train, y_train)

	# Evaluate
	proba = stack.predict_proba(X_test)
	if proba.ndim == 2 and proba.shape[1] > 1:
		y_pred_pos = proba[:, 1]
	else:
		y_pred_pos = proba.ravel()
	validation_auc = roc_auc_score(y_test, y_pred_pos)
	print('Validation ROC AUC:', validation_auc)

	# Prepare test set using training transformers
	df_full = pd.read_csv("Project/Binary Classification with a Bank Dataset/test.csv")
	dl.df = df_full
	dl.merge_columns()
	dl.encode_and_scale()
	X = dl.X

	# Predict on test set
	y_pred_proba = stack.predict_proba(X)
	if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
		y_pred_pos = y_pred_proba[:, 1]
	else:
		y_pred_pos = y_pred_proba.ravel()

	submission = pd.DataFrame({'id': df_full['id'], 'y': y_pred_pos})
	submission.to_csv(out_path, index=False)
	print(f'Submission saved to {out_path}')
	# If running full job, also print validation AUC and submission head together
	if full:
		print('\n=== Full run summary ===')
		print(f'Validation ROC AUC: {validation_auc}')
		print('\nSubmission head:')
		# show first 10 rows
		print(submission.head(10).to_string(index=False))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--full', action='store_true', help='Run full dataset (higher resource usage)')
	parser.add_argument('--select', type=float, default=None, help='Override sample fraction (0-1)')
	parser.add_argument('--out', type=str, default='Project/Binary Classification with a Bank Dataset/submission.csv', help='Submission output path')
	args = parser.parse_args()
	main(full=args.full, select=args.select, out_path=args.out)