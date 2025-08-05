import numpy as np
import pandas as pd
# from prometheus_client import Counter
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
from matplotlib.gridspec import GridSpec
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

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
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(drop='first'), non_numeric_cols)
    ])
preprocessor.fit(X)
# 然后拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 应用预处理
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
# 使用ADASYN
oversampler = ADASYN(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42
)

X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
print("After ADASYN - Train set distribution:", Counter(y_train_res))

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
        max_iter=5000,
        # max_iter=500,  # 限制最大迭代次数
        tol=1e-4
    ))
])

# 优化超参数网格
param_grid = [
    {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-3, 2, 6),
        'clf__solver': ['saga'],
        # 'clf__class_weight': [class_weights, 'balanced']
        'clf__class_weight': [None, class_weights]
    },
    {
        'clf__penalty': ['elasticnet'],
        'clf__C': np.logspace(-3, 2, 6),
        'clf__solver': ['saga'],
        'clf__class_weight': [None, class_weights],
        # 'clf__class_weight': [class_weights, 'balanced'],
        'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        # 'clf__l1_ratio': [0.1, 0.3]
    }
]

# 网格搜索优化
print("\nStarting grid search...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    # scoring='f1_weighted',
    scoring='f1_macro', # 使用宏平均F1分数，更关注少数类
    # cv=3,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_res, y_train_res)
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

# 测试集评估
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== Test Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 可视化结果 --------------------------------------------------------
import os
from matplotlib.gridspec import GridSpec
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

# 创建目录保存所有图表
os.makedirs('model_visualizations_withres', exist_ok=True)

# 1. 训练集混淆矩阵
plt.figure(figsize=(10, 8))
cm_train = confusion_matrix(y_train, y_train_pred)
cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_train_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (Train Set)')
plt.savefig('model_visualizations_withres/confusion_matrix_train.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 测试集混淆矩阵
plt.figure(figsize=(10, 8))
cm_test = confusion_matrix(y_test, y_pred)
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_test_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (Test Set)')
plt.savefig('model_visualizations_withres/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 整个数据集混淆矩阵
# 在整个数据集上评估模型
X_full = preprocessor.transform(X)
y_full_pred = best_model.predict(X_full)
y_full = y
plt.figure(figsize=(10, 8))
cm_full = confusion_matrix(y_full, y_full_pred)
cm_full_normalized = cm_full.astype('float') / cm_full.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_full_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (Full Dataset)')
plt.savefig('model_visualizations_withres/confusion_matrix_full.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ROC曲线和PR曲线
plt.figure(figsize=(12, 10))

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.subplot(2, 1, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
plt.subplot(2, 1, 2)
plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig('model_visualizations_withres/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 特征重要性
plt.figure(figsize=(12, 8))
clf = best_model.named_steps['clf']

try:
    # 尝试获取预处理后的特征名称
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    # 如果预处理没有get_feature_names_out方法
    try:
        # 尝试从ColumnTransformer获取
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if transformer == 'passthrough':
                feature_names.extend(features)
            elif hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(features))
            else:
                feature_names.extend(features)
    except:
        # 最后手段：使用数字索引
        n_features = clf.coef_.shape[1]
        feature_names = [f"Feature_{i}" for i in range(n_features)]

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(clf.coef_[0])
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances (Absolute Coefficient Value)')
plt.xlabel('Absolute Coefficient Value')
plt.savefig('model_visualizations_withres/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 超参数调节曲线
if hasattr(grid_search, 'cv_results_'):
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # 提取重要参数和得分
    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    plot_data = cv_results[param_cols + ['mean_test_score']].copy()

    # 将不可哈希的 dict 转换为字符串
    for col in plot_data.columns:
        if plot_data[col].dtype == object:
            plot_data[col] = plot_data[col].astype(str)

    # 绘制不同参数组合的得分
    plt.figure(figsize=(15, 10))

    # 创建网格布局
    gs = GridSpec(2, 2, figure=plt.gcf())

    # 1. C参数与得分
    ax1 = plt.subplot(gs[0, 0])
    sns.boxplot(x='param_clf__C', y='mean_test_score', data=plot_data, ax=ax1)
    ax1.set_title('C Parameter vs F1 Score')
    ax1.set_xlabel('C Value (log scale)')
    ax1.set_ylabel('Mean F1 Score')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # 2. 惩罚类型与得分
    ax2 = plt.subplot(gs[0, 1])
    sns.boxplot(x='param_clf__penalty', y='mean_test_score', data=plot_data, ax=ax2)
    ax2.set_title('Penalty Type vs F1 Score')
    ax2.set_xlabel('Penalty Type')
    ax2.set_ylabel('Mean F1 Score')

    # 3. 类别权重与得分（现在可以安全绘图）
    ax3 = plt.subplot(gs[1, :])
    sns.boxplot(x='param_clf__class_weight', y='mean_test_score', data=plot_data, ax=ax3)
    ax3.set_title('Class Weight vs F1 Score')
    ax3.set_xlabel('Class Weight Strategy')
    ax3.set_ylabel('Mean F1 Score')

    plt.tight_layout()
    plt.savefig('model_visualizations_withres/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 评估指标表格
# 计算各项指标
train_metrics = {
    'Dataset': 'Train',
    'Accuracy': accuracy_score(y_train, y_train_pred),
    'Precision': precision_score(y_train, y_train_pred, average='macro'),
    'Recall': recall_score(y_train, y_train_pred, average='macro'),
    'F1': f1_score(y_train, y_train_pred, average='macro'),
    'AUC': roc_auc_score(y_train, y_train_proba)
}

test_metrics = {
    'Dataset': 'Test',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='macro'),
    'Recall': recall_score(y_test, y_pred, average='macro'),
    'F1': f1_score(y_test, y_pred, average='macro'),
    'AUC': roc_auc_score(y_test, y_proba)
}

# 创建表格
metrics_df = pd.DataFrame([train_metrics, test_metrics])
metrics_df = metrics_df.round(4)

# 保存为CSV
metrics_df.to_csv('model_visualizations_withres/evaluation_metrics.csv', index=False)

# 可视化表格
plt.figure(figsize=(10, 4))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table = pd.plotting.table(ax, metrics_df, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title('Model Evaluation Metrics', y=1.1)
plt.savefig('model_visualizations_withres/evaluation_metrics_table.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存结果
joblib.dump(best_model, 'best_logistic_model.pkl')
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nModel saved as 'best_logistic_model.pkl'")
print("Feature importance saved as 'feature_importance.csv'")
print("All visualizations saved in 'model_visualizations_withres' directory")