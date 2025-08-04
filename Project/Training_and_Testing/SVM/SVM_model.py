import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # 添加SHAP库用于模型解释

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

# ===== SVM 实现部分 =====
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

print("\n=== SVM 模型训练 ===")

# 确保所有特征都是数值型
print("\n训练集数据类型:")
print(X_train.dtypes.value_counts())

# 创建预处理和SVM的管道
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM对特征尺度敏感，必须标准化
    ('svm', SVC(probability=True, random_state=42))  # 启用概率预测
])

# 定义参数网格
param_grid = {
    'svm__C': [0.1, 1, 10, 100],  # 正则化参数
    'svm__kernel': ['linear', 'rbf', 'poly'],  # 核函数类型
    'svm__gamma': ['scale', 'auto', 0.1, 1],  # RBF核的参数
    'svm__class_weight': ['balanced', None]  # 处理类别不平衡
}

# 使用网格搜索寻找最佳参数
print("\n开始SVM网格搜索...")
svm_grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',  # 使用宏平均F1分数，更关注少数类
    n_jobs=-1,
    verbose=2
)
svm_grid.fit(X_train, y_train)
print("SVM网格搜索完成!")

# 获取最佳模型
best_svm = svm_grid.best_estimator_
best_params = svm_grid.best_params_
best_score = svm_grid.best_score_

print("\n=== 最佳SVM模型 ===")
print(f"最佳参数: {best_params}")
print(f"最佳交叉验证F1分数: {best_score:.4f}")

# 在训练集上评估
y_train_pred = best_svm.predict(X_train)
y_train_proba = best_svm.predict_proba(X_train)[:, 1]

print("\n=== 训练集性能 ===")
print(f"准确率: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
print("分类报告:")
print(classification_report(y_train, y_train_pred))

# 在测试集上评估
y_test_pred = best_svm.predict(X_test)
y_test_proba = best_svm.predict_proba(X_test)[:, 1]

print("\n=== 测试集性能 ===")
print(f"准确率: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print("分类报告:")
print(classification_report(y_test, y_test_pred))

# 可视化结果
plt.figure(figsize=(15, 10))

# 1. 混淆矩阵
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_test_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('标准化混淆矩阵')

# 2. ROC曲线
plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend(loc='lower right')

# 3. 精确率-召回率曲线
plt.subplot(2, 2, 3)
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend(loc='upper right')

# 4. 特征重要性（仅适用于线性核）
if best_params['svm__kernel'] == 'linear':
    plt.subplot(2, 2, 4)
    coef = best_svm.named_steps['svm'].coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(coef)
    }).sort_values('Importance', ascending=False).head(15)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('SVM特征重要性（线性核）')
else:
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, '特征重要性仅适用于线性核', 
             ha='center', va='center', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.savefig('svm_results.png', dpi=300)
plt.show()

# SHAP解释（可选，计算量较大）
try:
    print("\n生成SHAP解释...")
    # 创建背景数据（减少样本量以加快计算）
    background = shap.sample(X_train, 100)
    
    # 创建解释器
    explainer = shap.KernelExplainer(best_svm.predict_proba, background)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test.iloc[:50])
    
    # 可视化摘要图
    plt.figure()
    shap.summary_plot(shap_values[1], X_test.iloc[:50], show=False)
    plt.title('SHAP值摘要')
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
    
    # 可视化单个预测示例
    plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0], show=False)
    plt.title('单个预测SHAP解释')
    plt.savefig('shap_force.png', bbox_inches='tight', dpi=300)
    
    print("SHAP解释生成完成!")
except Exception as e:
    print(f"SHAP解释失败: {e}")

# 保存模型
import joblib
joblib.dump(best_svm, 'best_svm_model.pkl')
print("\nSVM模型已保存为 'best_svm_model.pkl'")