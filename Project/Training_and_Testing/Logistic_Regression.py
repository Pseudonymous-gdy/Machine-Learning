import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

# 读取数据preprocessed_data.csv
try:
    df = pd.read_csv('preprocessed_data.csv')
except Exception as e:
    print("Error reading the CSV file: ", e)
    exit()

# 定义标签列
label = 'sfdm2'

# 手动定义需要的列
required_cols = [
    'age', 
    # 'sex_0',
    'sex_1', 
    # 'dzgroup_0',
    'dzgroup_1','dzgroup_2','dzgroup_3','dzgroup_4','dzgroup_5','dzgroup_6','dzgroup_7', 
    'dzclass_0',
    'dzclass_1','dzclass_2','dzclass_3', 
    'num.co', 'edu', 'income', 
    'scoma', 'sps', 'aps', 
    # 'race_0',
    'race_1','race_2','race_3','race_4',
    # 'diabetes', 
    # 'dementia', 
    # 'ca_0',
    'ca_1','ca_2', 
    'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 
    'crea', 'sod', 'ph', 'glucose', 'bun'
]

# 创建特征矩阵和标签向量
X = df[required_cols]
y = df[label]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
import joblib

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
required_cols = [
    'age', 'sex_1', 
    'dzgroup_1', 'dzgroup_2', 'dzgroup_3', 'dzgroup_4', 'dzgroup_5', 'dzgroup_6', 'dzgroup_7',
    'dzclass_1', 'dzclass_2', 'dzclass_3',
    'num.co', 'edu', 'income', 
    'scoma', 'sps', 'aps', 
    'race_1', 'race_2', 'race_3', 'race_4',
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

print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")

# 5. 创建处理管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        random_state=42, 
        max_iter=5000,
        multi_class='multinomial' if n_classes > 2 else 'auto'
    ))
])

# 6. 定义超参数网格
param_grid = {
    'clf__penalty': ['l1', 'l2', 'elasticnet'],
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__solver': ['saga'],
    'clf__l1_ratio': [0.3, 0.5, 0.7],
    'clf__class_weight': [None, 'balanced']
}

# 7. 网格搜索优化
print("\n开始超参数搜索...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc_ovr' if n_classes > 2 else 'roc_auc',  # 多分类支持
    cv=5,
    n_jobs=-1,
    verbose=2  # 更详细的输出
)
grid_search.fit(X_train, y_train)
print("超参数搜索完成!")

# 8. 评估最佳模型
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print("\n=== 最佳超参数 ===")
print(best_params)
print(f"最佳交叉验证AUC: {best_score:.4f}")

# 9. 测试集评估
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n=== 测试集性能 ===")
print(f"准确率: {test_accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 多分类AUC计算
if n_classes > 2:
    y_proba = best_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"多分类AUC (One-vs-Rest): {test_auc:.4f}")
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
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('标准化混淆矩阵')

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
plt.xlabel('平均绝对系数值')
plt.title('Top 15 特征重要性')
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
    plt.plot([0, 1], [0, 1], 'k--', label="随机猜测")
    plt.legend()
    plt.title('多类别ROC曲线 (One-vs-Rest)')
else:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'LogReg (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')

# 10.4 正则化路径修正（使用相同预处理）
plt.subplot(2, 2, 4)
c_values = np.logspace(-3, 2, 20)
coef_paths = []

# 使用相同的预处理
scaler = best_model.named_steps['scaler']
X_train_scaled = scaler.transform(X_train)  # 使用训练好的scaler

for c in c_values:
    # 创建与最佳模型相同配置的模型
    model = LogisticRegression(
        penalty=best_params['clf__penalty'],
        C=c,
        solver=best_params['clf__solver'],
        l1_ratio=best_params['clf__l1_ratio'],
        multi_class='multinomial' if n_classes > 2 else 'auto',
        max_iter=5000,
        random_state=42
    )
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
    plt.plot(np.log10(c_values), coef_paths[i], 
             label=f"{required_cols[i][:15]}..." if len(required_cols[i]) > 15 else required_cols[i])

plt.axvline(np.log10(best_params['clf__C']), color='k', linestyle='--', alpha=0.3, label='最佳C值')
plt.xlabel('log10(C)')
plt.ylabel('系数绝对值')
plt.title('Top 5 特征的正则化路径')
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