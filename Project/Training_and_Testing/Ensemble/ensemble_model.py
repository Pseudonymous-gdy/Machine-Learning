import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from collections import Counter
import time
import shap  # 添加SHAP库用于模型解释

# 配置设置
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# 定义评估函数 --------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    try:
        # 尝试使用预测概率进行阈值调整
        y_proba = model.predict_proba(X_test)
        y_pred = adjust_threshold(y_proba, y_test)  # 修改为传入真实标签用于识别少数类
    except:
        # 如果模型不支持概率预测，使用默认预测
        y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"{model_name} evaluation:")
    print('-'*50)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("Weighted F1 score:", f1_score(y_test, y_pred, average='weighted'))
    print("Macro average F1 score:", f1_score(y_test, y_pred, average='macro'))
    
    # 添加少数类F1分数
    unique_classes = np.unique(y_test)
    for cls in unique_classes:
        cls_f1 = f1_score(y_test, y_pred, labels=[cls], average=None)[0]
        print(f"Class {cls} F1 score: {cls_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    # 绘制PR曲线
    plot_pr_curves(model, X_test, y_test, model_name)
    return y_pred

def plot_pr_curves(model, X_test, y_test, model_name):
    try:
        y_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            # 二值化当前类
            y_true_class = (y_test == i).astype(int)
            y_score_class = y_proba[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true_class, y_score_class)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, lw=2, 
                     label=f'Class {i} (AUC = {pr_auc:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'pr_curve_{model_name.replace(" ", "_")}.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not plot PR curves for {model_name}: {str(e)}")

def adjust_threshold(y_proba, y_true):
    """根据类别分布调整决策阈值"""
    # 从真实标签中识别少数类
    class_counts = np.bincount(y_true)
    minority_class = np.argmin(class_counts)
    n_classes = len(np.unique(y_true))
    
    # 设置阈值 - 对少数类使用更低的阈值
    thresholds = np.full(n_classes, 0.5)
    thresholds[minority_class] = 0.2  # 降低少数类的阈值
    
    # 应用阈值
    y_pred = np.argmax(y_proba, axis=1)
    max_proba = np.max(y_proba, axis=1)
    
    for i in range(n_classes):
        # 找到低置信度样本
        low_confidence_idx = np.where(max_proba < thresholds[i])[0]
        
        # 找到高置信度样本
        high_confidence_idx = np.where(
            (y_pred == i) & (y_proba[:, i] >= thresholds[i])
        )[0]
        
        # 只保留高置信度的预测
        mask = np.isin(np.arange(len(y_pred)), high_confidence_idx)
        y_pred[~mask] = -1  # 标记低置信度预测
    
    # 处理低置信度预测：分配到概率最高的类
    low_conf_idx = np.where(y_pred == -1)[0]
    if len(low_conf_idx) > 0:
        y_pred[low_conf_idx] = np.argmax(y_proba[low_conf_idx], axis=1)
    
    return y_pred

def optimize_model(model, param_grid, X_train, y_train, cv=3):
    """使用网格搜索优化模型超参数"""
    print(f"\nOptimizing {model.__class__.__name__}...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def weighted_ensemble_predict(models, weights, X):
    """加权融合多个模型的预测概率"""
    probas = []
    for (name, model), weight in zip(models.items(), weights):
        try:
            proba = model.predict_proba(X) * weight
            probas.append(proba)
        except:
            print(f"Model {name} does not support predict_proba, skipping.")
    
    if len(probas) == 0:
        raise ValueError("No models support probability prediction.")
    
    avg_proba = np.mean(probas, axis=0)
    return np.argmax(avg_proba, axis=1)

# 安全读取数据
try:
    df = pd.read_csv('data.csv')
    print("Dataset read successfully! Length of dataset:", len(df))
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

# 处理缺失值 --------------------------------------------------------
print("\n=== 缺失值处理 ===")

# 可视化缺失值分布
missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(12, 6))
missing_percent.plot(kind='bar')
plt.title('Missing Value Percentage in Each Column')
plt.ylabel('Missing Percentage (%)')
plt.savefig('missing_values.png', dpi=300)
# plt.show()

print("Columns with missing values:")
print(missing_percent)

# 删除高缺失率列
threshold = 0.3
missing_ratio = df.isnull().mean()
df_reduced = df.loc[:, missing_ratio <= threshold]
print(f"\nOriginal features: {df.shape[1]}, After removal: {df_reduced.shape[1]}")

# 数值列用中位数填充
numeric_cols = df_reduced.select_dtypes(include=['number']).columns
for col in numeric_cols:
    if df_reduced[col].isnull().any():
        median_val = df_reduced[col].median()
        df_reduced[col] = df_reduced[col].fillna(median_val)

# 非数值列用众数填充
non_numeric_cols = df_reduced.select_dtypes(include='object').columns
for col in non_numeric_cols:
    if df_reduced[col].isnull().any():
        mode_val = df_reduced[col].mode()[0]
        df_reduced[col] = df_reduced[col].fillna(mode_val)

# One-hot编码
df_processed = pd.get_dummies(df_reduced, columns=non_numeric_cols, drop_first=True)
print("\nProcessed dataframe shape:", df_processed.shape)

# 分割数据集 --------------------------------------------------------
target = 'death'
X = df_processed.drop(columns=[target])
y = df_processed[target]

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

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 特征选择 --------------------------------------------------------
print("\n=== 特征选择 ===")

# 相关性分析
df_corr = pd.concat([X_scaled_df, y], axis=1)
corr_with_target = df_corr.corr()[target].sort_values(ascending=False)
corr_with_target.drop(target, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title(f"Correlation with Target '{target}'")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300)
# plt.show()

corr_threshold = 0.1
selected_features_corr = corr_with_target[abs(corr_with_target) > corr_threshold].index.tolist()
print(f"\nCorrelation-selected features ({len(selected_features_corr)}):")

# 互信息
mi_scores = mutual_info_classif(X_scaled_df, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(20))
plt.title("Mutual Information Scores (Top 20)")
plt.tight_layout()
plt.savefig('feature_mi_scores.png', dpi=300)
# plt.show()

selected_features_mi = mi_df.head(20)['Feature'].tolist()
print(f"\nMI-selected features ({len(selected_features_mi)}):")

# 随机森林重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled_df, y)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title("Random Forest Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig('feature_rf_importance.png', dpi=300)
# plt.show()

selected_features_rf = feature_importances.head(20)['Feature'].tolist()
print(f"\nRF-selected features ({len(selected_features_rf)}):")

# 特征选择综合
all_methods = {
    'Correlation': selected_features_corr,
    'Mutual_Info': selected_features_mi,
    'Random_Forest': selected_features_rf
}

feature_selection_counts = pd.DataFrame(index=X.columns)
for method, features in all_methods.items():
    feature_selection_counts[method] = feature_selection_counts.index.isin(features).astype(int)

# # 计算加权分数
# feature_selection_counts['Weighted_Score'] = (
#     feature_selection_counts['Correlation'] * 0.4 +
#     feature_selection_counts['Mutual_Info'] * 0.3 +
#     feature_selection_counts['Random_Forest'] * 0.3
# )

# feature_selection_counts = feature_selection_counts.sort_values('Weighted_Score', ascending=False)

# print("\nFeature selection weighted scores:")
# print(feature_selection_counts.head(25))

# # 选择前25个特征
# consensus_features = feature_selection_counts.head(25).index.tolist()

feature_selection_counts['Selection_Count'] = feature_selection_counts.sum(axis=1)
feature_selection_counts = feature_selection_counts.sort_values('Selection_Count', ascending=False)
print("\nFeature selection consensus:")
print(feature_selection_counts.head(20))

# 选择被至少2种方法选中的特征
consensus_features = feature_selection_counts[feature_selection_counts['Selection_Count'] >= 2].index.tolist()
print(f"\nFinal selected features ({len(consensus_features)}): {consensus_features}")

X_final = X_scaled_df[consensus_features]

# 检查类别数量
n_classes = len(np.unique(y))
print(f"\nTarget '{target}' has {n_classes} classes after merging and encoding")
print("Class distribution:", Counter(y))

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set size: {len(X_train)}, Test set size: {len(X_test)}")
print("Train set distribution:", Counter(y_train))
print("Test set distribution:", Counter(y_test))

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# # 应用改进的过采样技术 - BorderlineSMOTE
# print("\nApplying BorderlineSMOTE to balance classes...")
# oversampler = BorderlineSMOTE(
#     sampling_strategy='auto',
#     k_neighbors=5,
#     random_state=42
# )
# X_train_res, y_train_res = oversampler.fit_resample(X_train_scaled, y_train)
# print("After BorderlineSMOTE - Train set distribution:", Counter(y_train_res)) 
# 使用ADASYN
oversampler = ADASYN(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42
)

X_train_res, y_train_res = oversampler.fit_resample(X_train_scaled, y_train)
print("After ADASYN - Train set distribution:", Counter(y_train_res))

# 计算样本权重
sample_weights = compute_sample_weight('balanced', y_train_res)

# 模型集合
models = {}
training_times = {}

# Balanced Random Forest
print("\nTraining Balanced Random Forest...")
start_time = time.time()
brf = BalancedRandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    sampling_strategy='auto',
    replacement=True,
    random_state=42,
    n_jobs=-1
)
brf.fit(X_train_res, y_train_res)
training_times['Balanced RF'] = time.time() - start_time
models['Balanced RF'] = brf
print(f"Balanced RF training completed in {training_times['Balanced RF']:.2f} seconds")

# RUSBoost
print("\nTraining RUSBoost...")
start_time = time.time()
rusboost = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=300,
    learning_rate=0.05,
    sampling_strategy='auto',
    random_state=42,
    algorithm='SAMME'  # 修复参数错误
)
rusboost.fit(X_train_res, y_train_res)
training_times['RUSBoost'] = time.time() - start_time
models['RUSBoost'] = rusboost
print(f"RUSBoost training completed in {training_times['RUSBoost']:.2f} seconds")

# XGBoost with hyperparameter optimization and class weights
print("\nOptimizing XGBoost with class weights...")
start_time = time.time()
# 计算少数类权重比例
scale_pos_weight = class_weights[1] / class_weights[0]  # 少数类权重 / 多数类权重

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic' if n_classes == 2 else 'multi:softmax',
    num_class=n_classes if n_classes > 2 else None,
    eval_metric='logloss' if n_classes == 2 else 'merror',
    scale_pos_weight=scale_pos_weight if n_classes == 2 else None,
    random_state=42,
    n_jobs=-1
)

# XGBoost参数网格 
xgb_param_grid = {
    'n_estimators': [150, 200, 300, 350],
    'learning_rate': [0.05, 0.1, 0.01],
    'max_depth': [4, 6],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0.5, 1]
}

# 优化XGBoost
best_xgb = optimize_model(xgb_model, xgb_param_grid, X_train_res, y_train_res)
models['Optimized XGBoost'] = best_xgb
training_times['Optimized XGBoost'] = time.time() - start_time

# Bagging with optimized base estimator
print("\nTraining Bagging Classifier...")
start_time = time.time()
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        class_weight='balanced', 
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5
    ),
    n_estimators=300,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train_res, y_train_res)
training_times['Bagging'] = time.time() - start_time
models['Bagging'] = bagging
print(f"Bagging training completed in {training_times['Bagging']:.2f} seconds")

# Gradient Boosting with class weights
print("\nTraining Gradient Boosting...")
start_time = time.time()
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_res, y_train_res, sample_weight=sample_weights)
training_times['Gradient Boosting'] = time.time() - start_time
models['Gradient Boosting'] = gb
print(f"Gradient Boosting training completed in {training_times['Gradient Boosting']:.2f} seconds")

# Stacking集成 - 使用表现最好的模型作为基学习器
print("\nTraining Stacking Ensemble...")
start_time = time.time()

# 选择表现最好的三个模型作为基学习器
base_learners = [
    ('brf', BalancedRandomForestClassifier(n_estimators=300, max_depth=10)),
    ('gb', GradientBoostingClassifier(n_estimators=250, learning_rate=0.05)),
    ('xgb', best_xgb)
]

# 添加元学习器
stacker = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=2000),
    cv=5,
    n_jobs=-1
)

stacker.fit(X_train_res, y_train_res)
training_times['Stacking'] = time.time() - start_time
models['Stacking'] = stacker
print(f"Stacking training completed in {training_times['Stacking']:.2f} seconds")

# 评估所有模型
print("\nEvaluating all models...")
print("=" * 60)
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    start_time = time.time()
    y_pred = evaluate_model(model, X_test_scaled, y_test, name)
    eval_time = time.time() - start_time
    
    # 收集评估指标
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'train_time': training_times.get(name, 'N/A'),
        'eval_time': eval_time
    }
    
    # 添加每个类别的F1分数
    for cls in np.unique(y_test):
        cls_f1 = f1_score(y_test, y_pred, labels=[cls], average=None)[0]
        results[name][f'f1_class_{cls}'] = cls_f1

# 模型比较
print("\nModel Comparison:")
print("=" * 60)
comparison = pd.DataFrame(results).T
comparison = comparison.sort_values(by='f1_weighted', ascending=False)
print(comparison)

# 特征重要性可视化
plt.figure(figsize=(15, 10))
model_names = list(models.keys())
for i, (name, model) in enumerate(models.items(), 1):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.subplot(3, 3, i)
            plt.barh(range(10), importances[indices][:10], align='center')
            plt.yticks(range(10), [X.columns[idx] for idx in indices[:10]])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Top 10 Features')
    except Exception as e:
        print(f"Could not get feature importances for {name}: {str(e)}")

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# 加权融合模型预测
print("\nEvaluating Weighted Ensemble Fusion...")
# 基于验证集性能设置权重
weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]  # 按模型性能分配权重

# 移除Stacking模型（因为它本身已经是集成模型）
fusion_models = {k: v for k, v in models.items() if k != 'Stacking'}
fusion_weights = weights[:len(fusion_models)]

# 评估融合模型
y_pred_fused = weighted_ensemble_predict(fusion_models, fusion_weights, X_test_scaled)

# 评估融合模型性能
print("\nFused Model Evaluation:")
print("=" * 50)
print("accuracy:", accuracy_score(y_test, y_pred_fused))
print("Weighted F1 score:", f1_score(y_test, y_pred_fused, average='weighted'))
print("Macro average F1 score:", f1_score(y_test, y_pred_fused, average='macro'))

# 添加每个类别的F1分数
for cls in np.unique(y_test):
    cls_f1 = f1_score(y_test, y_pred_fused, labels=[cls], average=None)[0]
    print(f"Class {cls} F1 score: {cls_f1:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred_fused, digits=3))

# 保存融合模型结果到比较表
results['Weighted Fusion'] = {
    'accuracy': accuracy_score(y_test, y_pred_fused),
    'f1_weighted': f1_score(y_test, y_pred_fused, average='weighted'),
    'f1_macro': f1_score(y_test, y_pred_fused, average='macro'),
    'train_time': 'N/A',
    'eval_time': 'N/A'
}

for cls in np.unique(y_test):
    cls_f1 = f1_score(y_test, y_pred_fused, labels=[cls], average=None)[0]
    results['Weighted Fusion'][f'f1_class_{cls}'] = cls_f1

# 更新模型比较表
print("\nUpdated Model Comparison with Fusion:")
print("=" * 60)
comparison = pd.DataFrame(results).T
comparison = comparison.sort_values(by='f1_weighted', ascending=False)
print(comparison)

# 保存最佳模型
best_model_name = comparison.index[0]
best_model = models.get(best_model_name, None) or fusion_models.get(best_model_name, None)
print(f"\nBest model identified: {best_model_name} with weighted F1: {comparison.loc[best_model_name, 'f1_weighted']:.4f}")

# 使用SHAP分析最佳模型对类别0的预测
if best_model_name != 'Weighted Fusion' and hasattr(best_model, 'predict_proba'):
    print("\nAnalyzing feature impact on minority class using SHAP...")
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(best_model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_test_scaled)
        
        # 分析类别0的预测
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[0], 
            X_test_scaled, 
            feature_names=consensus_features,
            plot_type='dot',
            show=False
        )
        plt.title('Feature Impact on Class 0 (Death)')
        plt.tight_layout()
        plt.savefig('shap_class0_impact.png', dpi=300)
        plt.show()
        
        # 特定类别0样本的SHAP解释
        minority_indices = np.where(y_test == 0)[0]
        if len(minority_indices) > 0:
            sample_idx = minority_indices[0]
            plt.figure(figsize=(10, 6))
            shap.force_plot(
                explainer.expected_value[0], 
                shap_values[0][sample_idx], 
                X_test_scaled[sample_idx],
                feature_names=consensus_features,
                show=False
            )
            plt.title(f'SHAP Explanation for Sample {sample_idx} (Class 0)')
            plt.tight_layout()
            plt.savefig('shap_class0_sample.png', dpi=300)
            plt.show()
            
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")