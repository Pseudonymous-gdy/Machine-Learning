import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings  # 添加警告过滤
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
import joblib

# 过滤特定警告
# warnings.filterwarnings('ignore', category=UserWarning, message="l1_ratio.*")  # 过滤l1_ratio警告
# warnings.filterwarnings('ignore', category=FutureWarning)  # 过滤未来警告
# warnings.filterwarnings('ignore', category=ConvergenceWarning)  # 过滤收敛警告

# 1. 安全读取数据
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("dataset read successfully! len of dataset:", len(df))
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

# 2. 定义标签列和特征列
label = 'sfdm2'

# 使用实际特征名称
# required_cols = [
#     'age', 'sex_1', 
#     'dzgroup_1', 'dzgroup_2', 'dzgroup_3', 'dzgroup_4', 'dzgroup_5', 'dzgroup_6', 'dzgroup_7',
#     'dzclass_1', 'dzclass_2', 'dzclass_3',
#     'num.co', 'edu', 'income', 
#     'scoma', 'sps', 'aps', 
#     'race_1', 'race_2', 'race_3', 'race_4',
#     'ca_1', 'ca_2', 
#     'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 
#     'crea', 'sod', 'ph', 'glucose', 'bun'
# ]
# 使用实际特征名称
required_cols = [
    'age', 
    'sex_0', 
    'sex_1', 
    'dzgroup_0',
    'dzgroup_1', 'dzgroup_2', 'dzgroup_3', 'dzgroup_4', 'dzgroup_5', 'dzgroup_6', 'dzgroup_7',
    'dzclass_0',
    'dzclass_1', 'dzclass_2', 'dzclass_3',
    'num.co', 'edu', 'income', 
    'scoma', 'sps', 'aps', 
    'race_0',
    'race_1', 'race_2', 'race_3', 'race_4',
    'ca_0',
    'ca_1', 'ca_2', 
    'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 
    'crea', 'sod', 'ph', 'glucose', 'bun'
]

# 3. 准备数据
X = df[required_cols]
y = df[label]

# 检查类别数量
n_classes = len(np.unique(y))
print(f"\ntarget '{label}' has {n_classes} classes")
print("distribution:", np.bincount(y))

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\ntraining size: {X_train.shape}, testing size: {X_test.shape}")
print(f"training class distribution: {np.bincount(y_train)}")
print(f"testing class distribution: {np.bincount(y_test)}")

# 5. 创建处理管道（增加max_iter和tol）
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        random_state=42, 
        max_iter=10000,  # 增加到10000
        tol=1e-4,        # 添加容差参数
        # 移除了multi_class参数（新版自动处理）
    ))
])

# 6. 优化超参数网格（拆分参数网格）
class_weights = {
    0.0: 1, 
    1.0: 50,  # 大幅提高权重
    2.0: 10,
    3.0: 10,
    4.0: 1
}

param_grid = [
    # 非elasticnet参数网格
    {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__solver': ['saga'],
        'clf__class_weight': [None, 'balanced', class_weights]
    },
    # elasticnet专用参数网格
    {
        'clf__penalty': ['elasticnet'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__solver': ['saga'],
        'clf__class_weight': [None, 'balanced', class_weights],
        'clf__l1_ratio': [0.3, 0.5, 0.7]
    }
]

# 7. 网格搜索优化
print("\nstart grid search...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    # scoring='roc_auc_ovr' if n_classes > 2 else 'roc_auc',
    scoring='f1_weighted'
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print("hyperparameter search completed!")

# 8. 评估最佳模型
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print("\n=== best hyperparameters ===")
print(best_params)
print(f"best cross-validation AUC: {best_score:.4f}")

# 9. 测试集评估
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n=== test set performance ===")
print(f"accuracy: {test_accuracy:.4f}")
print("\nclassification report:")
print(classification_report(y_test, y_pred))

# 多分类AUC计算
if n_classes > 2:
    y_proba = best_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"muti-classification AUC (One-vs-Rest): {test_auc:.4f}")
else:
    y_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {test_auc:.4f}")

# 10. 可视化结果
plt.figure(figsize=(16, 12))

# 10.1 混淆矩阵（标准化显示）
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 标准化

sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Normalized Confusion Matrix')

# 10.2 特征重要性（使用实际特征名称）
plt.subplot(2, 2, 2)
clf = best_model.named_steps['clf']

# 处理多分类系数
if n_classes > 2:
    # 计算特征平均重要性
    feature_importance = np.mean(np.abs(clf.coef_), axis=0)
else:
    feature_importance = np.abs(clf.coef_[0])

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': required_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# 取前15个重要特征
top_features = importance_df.head(15)

# 绘制水平条形图
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Mean absolute coefficient value')
plt.title('Top 15 feature importances')
plt.gca().invert_yaxis()  # 最重要特征在顶部

# 10.3 ROC曲线（多类别处理）
plt.subplot(2, 2, 3)
if n_classes > 2:
    # 绘制每个类别的ROC曲线
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_test == i,
            y_proba[:, i],
            name=f"Class {i}",
            ax=plt.gca()
        )
    plt.plot([0, 1], [0, 1], 'k--', label="random guessing")
    plt.legend()
    plt.title('Multi-class ROC Curve (One-vs-Rest)')
else:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'LogReg (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

# 10.4 正则化路径修正（使用相同预处理）
plt.subplot(2, 2, 4)
c_values = np.logspace(-3, 2, 20)
coef_paths = []

# 使用相同的预处理
scaler = best_model.named_steps['scaler']
X_train_scaled = scaler.transform(X_train)

# 提取最佳参数（安全获取l1_ratio）
best_penalty = best_params.get('clf__penalty', 'l2')
best_solver = best_params.get('clf__solver', 'saga')
best_class_weight = best_params.get('clf__class_weight', None)
best_l1_ratio = best_params.get('clf__l1_ratio', None)  # 可能不存在

for c in c_values:
    # 动态设置参数
    model_params = {
        'penalty': best_penalty,
        'C': c,
        'solver': best_solver,
        'class_weight': best_class_weight,
        'max_iter': 10000,
        'random_state': 42,
        'tol': 1e-4
    }
    
    # 仅在elasticnet时设置l1_ratio
    if best_penalty == 'elasticnet' and best_l1_ratio is not None:
        model_params['l1_ratio'] = best_l1_ratio
    
    model = LogisticRegression(**model_params)
    model.fit(X_train_scaled, y_train)
    
    # 存储特征重要性（多分类取平均）
    if n_classes > 2:
        coef_paths.append(np.mean(np.abs(model.coef_), axis=0))
    else:
        coef_paths.append(np.abs(model.coef_[0]))

coef_paths = np.array(coef_paths).T

# 绘制前5个最重要特征的路径
top_feature_indices = importance_df.index[:5]
for i in top_feature_indices:
    feature_name = required_cols[i]
    # 截断长特征名
    display_name = feature_name[:15] + '...' if len(feature_name) > 15 else feature_name
    plt.plot(np.log10(c_values), coef_paths[i], label=display_name)

plt.axvline(np.log10(best_params['clf__C']), color='k', linestyle='--', alpha=0.3, label='Best C value')
plt.xlabel('log10(C)')
plt.ylabel('Coefficient absolute value')
plt.title('Top 5 feature regularization paths')
plt.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=300)
plt.show()

# 11. 输出模型参数
print("\n=== 模型参数 ===")
print(f"截距 (bias): {clf.intercept_}")

print("\n系数 (weights):")
coef_df = pd.DataFrame(clf.coef_, columns=required_cols, index=[f"Class {i}" for i in range(n_classes)])
print(coef_df)

# 12. 保存最佳模型
joblib.dump(best_model, 'best_logistic_regression_model.pkl')
print("\n模型已保存为 'best_logistic_regression_model.pkl'")

# 13. 保存特征重要性
importance_df.to_csv('feature_importance.csv', index=False)
print("特征重要性已保存为 'feature_importance.csv'")