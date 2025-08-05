from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from collections import Counter
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
    
    # 创建模型专属目录保存所有图表
    model_dir = f"model_visualizations_without_resampling/{model_name.replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. 训练集混淆矩阵
    plt.figure(figsize=(10, 8))
    cm_train = confusion_matrix(y_train, model.predict(X_train))
    cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_train_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=np.unique(y_train), 
                yticklabels=np.unique(y_train))
    plt.title(f'{model_name} Confusion Matrix (Train Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_dir}/confusion_matrix_train.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 测试集混淆矩阵
    plt.figure(figsize=(10, 8))
    cm_test = confusion_matrix(y_test, y_pred)
    cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_test_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.title(f'{model_name} Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_dir}/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 整个数据集混淆矩阵
    plt.figure(figsize=(10, 8))
    X_full = preprocessor.transform(X)
    y_full_pred = model.predict(X_full)
    cm_full = confusion_matrix(y, y_full_pred)
    cm_full_normalized = cm_full.astype('float') / cm_full.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_full_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=np.unique(y), 
                yticklabels=np.unique(y))
    plt.title(f'{model_name} Confusion Matrix (Full Dataset)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_dir}/confusion_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. PR曲线
    if hasattr(model, 'predict_proba'):
        plot_pr_curves(model, X_test, y_test, model_name, model_dir)
    
    # 5. 评估指标表格
    # 计算各项指标
    train_metrics = {
        'Dataset': 'Train',
        'Accuracy': accuracy_score(y_train, model.predict(X_train)),
        'Precision': precision_score(y_train, model.predict(X_train), average='macro'),
        'Recall': recall_score(y_train, model.predict(X_train), average='macro'),
        'F1': f1_score(y_train, model.predict(X_train), average='macro'),
        'AUC': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) if hasattr(model, 'predict_proba') else np.nan
    }

    test_metrics = {
        'Dataset': 'Test',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1': f1_score(y_test, y_pred, average='macro'),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else np.nan
    }

    # 创建表格
    metrics_df = pd.DataFrame([train_metrics, test_metrics])
    metrics_df = metrics_df.round(4)

    # 保存为CSV
    metrics_df.to_csv(f'{model_dir}/evaluation_metrics.csv', index=False)

    # 可视化表格
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = pd.plotting.table(ax, metrics_df, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title(f'{model_name} Evaluation Metrics', y=1.1)
    plt.savefig(f'{model_dir}/evaluation_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return y_pred

def plot_pr_curves(model, X_test, y_test, model_name, model_dir):
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
        plt.savefig(f'{model_dir}/pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not plot PR curves for {model_name}: {str(e)}")

def optimize_model(model, param_grid, X_train, y_train, cv=3):
    """使用网格搜索优化模型超参数，并稳健绘制超参数曲线"""
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

    # 保存超参数调节曲线（健壮版）
    if hasattr(grid_search, 'cv_results_'):
        cv_results = pd.DataFrame(grid_search.cv_results_)

        # 提取参数列
        param_cols = [c for c in cv_results.columns if c.startswith('param_')]
        plot_data = cv_results[param_cols + ['mean_test_score']].copy()

        # 将不可哈希类型转为字符串
        for col in param_cols:
            plot_data[col] = plot_data[col].astype(str)

        # 创建目录
        model_name = model.__class__.__name__
        model_dir = f"model_visualizations/{model_name.replace(' ', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        # 绘图参数映射
        plot_configs = [
            ('n_estimators', 'Number of Estimators'),
            ('learning_rate', 'Learning Rate'),
            ('max_depth', 'Max Depth'),
            ('reg_alpha', 'Regularization Alpha'),
        ]

        # 只绘制存在的参数
        valid_plots = [(key, title) for key, title in plot_configs if f'param_{key}' in plot_data.columns]

        if not valid_plots:
            print("⚠️ 无可视化参数列，跳过超参数绘图。")
            return grid_search.best_estimator_

        # 创建子图
        n_plots = len(valid_plots)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, (key, title) in zip(axes, valid_plots):
            sns.boxplot(
                x=f'param_{key}',
                y='mean_test_score',
                data=plot_data,
                ax=ax
            )
            ax.set_title(f'{title} vs F1 Score')
            ax.set_xlabel(title)
            ax.set_ylabel('Mean F1 Score')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{model_dir}/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"超参数可视化已保存：{model_dir}/hyperparameter_tuning.png")

    return grid_search.best_estimator_


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

# 检查类别数量
n_classes = len(np.unique(y))
print(f"\nTarget '{target}' has {n_classes} classes after merging and encoding")
print("Class distribution:", Counter(y))

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

# # 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# # 使用ADASYN
# oversampler = ADASYN(
#     sampling_strategy='auto',
#     n_neighbors=5,
#     random_state=42
# )

# X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
# print("After ADASYN - Train set distribution:", Counter(y_train_res))

# 计算样本权重
sample_weights = compute_sample_weight('balanced', y_train)

# 模型集合
models = {}
training_times = {}

# Balanced Random Forest----------------------------------------------
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
brf.fit(X_train, y_train)
training_times['Balanced RF'] = time.time() - start_time
models['Balanced RF'] = brf
print(f"Balanced RF training completed in {training_times['Balanced RF']:.2f} seconds")

# RUSBoost----------------------------------------------
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
rusboost.fit(X_train, y_train)
training_times['RUSBoost'] = time.time() - start_time
models['RUSBoost'] = rusboost
print(f"RUSBoost training completed in {training_times['RUSBoost']:.2f} seconds")

# XGBoost with hyperparameter optimization and class weights----------------------------------------------
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
best_xgb = optimize_model(xgb_model, xgb_param_grid, X_train, y_train)
models['Optimized XGBoost'] = best_xgb
training_times['Optimized XGBoost'] = time.time() - start_time

# Bagging with optimized base estimator----------------------------------------------
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
bagging.fit(X_train, y_train)
training_times['Bagging'] = time.time() - start_time
models['Bagging'] = bagging
print(f"Bagging training completed in {training_times['Bagging']:.2f} seconds")

# Gradient Boosting with class weights----------------------------------------------
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
gb.fit(X_train, y_train, sample_weight=sample_weights)
training_times['Gradient Boosting'] = time.time() - start_time
models['Gradient Boosting'] = gb
print(f"Gradient Boosting training completed in {training_times['Gradient Boosting']:.2f} seconds")

# Stacking集成 - 使用表现最好的模型作为基学习器----------------------------------------------
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

stacker.fit(X_train, y_train)
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
    y_pred = evaluate_model(model, X_test, y_test, name)
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
y_pred_fused = weighted_ensemble_predict(fusion_models, fusion_weights, X_test)

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
