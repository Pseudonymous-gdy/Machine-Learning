import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 安全读取数据
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("Dataset read successfully! Length of dataset:", len(df))
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

target = 'death'

X = df.drop(columns=[target])
y = df[target]

# 去除可能导致标签泄露的列
leakage_cols = ['hospdead', 'dnrday', 'totcst', 'totmcst', 'charges', 
                'totcst_log', 'totmcst_log', 'hday', 'long_term_diff', 
                'short_term_diff', 'surv2m', 'dnr', 'prg6m']
X = X.drop(columns=leakage_cols, errors='ignore')

# 目标变量类型转换
if y.dtype == object:
    y = y.astype('category').cat.codes
elif len(y.unique()) == 2:
    y = y.astype(int)

print(f"\nTarget variable '{target}' distribution:\n{y.value_counts(normalize=True)}")

# 模型训练 --------------------------------------------------------
print("\n=== 模型训练 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== 数据类型检查 ===")
print("训练集数据类型分布:")
print(X_train.dtypes.value_counts())

non_numeric_cols = X_train.select_dtypes(include=['object', 'category']).columns
if not non_numeric_cols.empty:
    print(f"\n发现非数值列: {list(non_numeric_cols)}")
    X_train = pd.get_dummies(X_train, columns=non_numeric_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=non_numeric_cols, drop_first=True)
    print(f"独热编码后特征数: {X_train.shape[1]}")

# 构建带预处理的管道
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=42, max_iter=10000, tol=1e-4))
])

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# 类别权重
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# 创建处理管道
pipeline = Pipeline([
    ('clf', LogisticRegression(
        random_state=42, 
        max_iter=10000,
        tol=1e-4
    ))
])

# 优化超参数网格
param_grid = [
    {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-3, 2, 6),
        'clf__solver': ['saga'],
        'clf__class_weight': [class_weights, 'balanced']
    },
    {
        'clf__penalty': ['elasticnet'],
        'clf__C': np.logspace(-3, 2, 6),
        'clf__solver': ['saga'],
        # 'clf__class_weight': [None, class_weights],
        'clf__class_weight': [class_weights, 'balanced'],
        'clf__l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    }
]

# 网格搜索优化
print("\nStarting grid search...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    # scoring='f1_weighted',
    scoring='f1_macro', # 使用宏平均F1分数，更关注少数类
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Hyperparameter search completed!")

# 评估最佳模型
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print("\n=== Best Model ===")
print(f"Best params: {best_params}")
print(f"Best CV F1: {best_score:.4f}")

# 模型在训练集上的评估
y_train_pred = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)[:, 1]

print("\n=== Train Performance ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Train AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))
# 可视化训练集结果
plt.figure(figsize=(15, 12))
# 混淆矩阵
plt.subplot(2, 2, 1)
cm_train = confusion_matrix(y_train, y_train_pred)
cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)
sns.heatmap(cm_train_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (Train)')

# 测试集评估
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== Test Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 可视化结果
plt.figure(figsize=(15, 12))

# 混淆矩阵
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')

# 特征重要性
plt.subplot(2, 2, 2)
clf = best_model.named_steps['clf']
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': np.abs(clf.coef_[0])
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top Feature Importances')
plt.xlabel('Absolute Coefficient Value')

# ROC曲线
plt.subplot(2, 2, 3)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300)
plt.show()

# 保存结果
joblib.dump(best_model, 'best_logistic_model.pkl')
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nModel saved as 'best_logistic_model.pkl'")
print("Feature importance saved as 'feature_importance.csv'")