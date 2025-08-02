import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from collections import Counter
import time

# 配置设置
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def evaluate_model(model, X_test, y_test, model_name):
    try:
        # 尝试使用预测概率进行阈值调整
        y_proba = model.predict_proba(X_test)
        y_pred = adjust_threshold(y_proba, np.unique(y_test))
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
        plt.show()
    except Exception as e:
        print(f"Could not plot PR curves for {model_name}: {str(e)}")

def adjust_threshold(y_proba, classes):
    """根据类别分布调整决策阈值"""
    n_classes = len(classes)
    # 默认阈值
    thresholds = np.full(n_classes, 0.5)
    
    # 对少数类使用更低的阈值
    for i in range(n_classes):
        if i == 1:  # 假设类别1是少数类
            thresholds[i] = 0.3
        else:
            thresholds[i] = 0.4
    
    # 应用阈值
    y_pred = np.argmax(y_proba, axis=1)
    max_proba = np.max(y_proba, axis=1)
    
    for i in range(n_classes):
        low_confidence_idx = np.where(max_proba < thresholds[i])[0]
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

# 加载数据
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("Dataset read successfully! Length of dataset:", len(df))
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

label = 'sfdm2'

# 合并少数类别（1和2合并到3）
print("\nOriginal class distribution:", Counter(df[label]))
# df[label] = df[label].replace({
#     1.0: 3.0,  # 将类别1合并到类别3
#     2.0: 3.0   # 将类别2合并到类别3
# })
# # 重新映射少数类别:0->0 1,2,3->1, 4->2
# df[label] = df[label].replace({
#     0.0: 0.0,  # 类别0保持不变
#     1.0: 1.0,  # 类别1保持不变
#     2.0: 1.0,  # 类别2合并到类别1
#     3.0: 1.0,  # 类别3合并到类别1
#     4.0: 2.0   # 类别4合并到类别2
# })
# print("After merging class 1 and 2 into 3:", Counter(df[label]))

# 重新映射标签为连续整数
# le = LabelEncoder()
# df[label] = le.fit_transform(df[label])
# print("After label encoding:", Counter(df[label]))

required_cols = [
    'age', 
    'sex_0', 'sex_1', 
    'dzgroup_0', 'dzgroup_1', 'dzgroup_2', 'dzgroup_3', 
    'dzgroup_4', 'dzgroup_5', 'dzgroup_6', 'dzgroup_7',
    'dzclass_0', 'dzclass_1', 'dzclass_2', 'dzclass_3',
    'num.co', 'edu', 'income', 
    'scoma', 'sps', 'aps', 
    'race_0', 'race_1', 'race_2', 'race_3', 'race_4',
    'ca_0', 'ca_1', 'ca_2', 
    'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 
    'crea', 'sod', 'ph', 'glucose', 'bun'
]

X = df[required_cols]
y = df[label]

# 检查类别数量
n_classes = len(np.unique(y))
print(f"\nTarget '{label}' has {n_classes} classes after merging and encoding")
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

# 应用SMOTE过采样
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42, k_neighbors=min(3, min(Counter(y_train).values()) - 1))
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE - Train set distribution:", Counter(y_train_res))

# 计算类别权重
class_weights = compute_sample_weight('balanced', y_train_res)

# 模型集合
models = {}
training_times = {}

# 1. Balanced Random Forest
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

# 2. RUSBoost
print("\nTraining RUSBoost...")
start_time = time.time()
rusboost = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=150,
    learning_rate=0.05,
    sampling_strategy='auto',
    random_state=42,
    algorithm='SAMME'  # 修复参数错误
)
rusboost.fit(X_train_res, y_train_res)
training_times['RUSBoost'] = time.time() - start_time
models['RUSBoost'] = rusboost
print(f"RUSBoost training completed in {training_times['RUSBoost']:.2f} seconds")

# 3. XGBoost with hyperparameter optimization
print("\nOptimizing XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=n_classes,
    eval_metric='merror',
    random_state=42,
    n_jobs=-1
)

# XGBoost参数网格 - 精简版
xgb_param_grid = {
    'n_estimators': [150, 200, 300],
    'learning_rate': [0.05, 0.1],
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

# 4. Bagging with optimized base estimator
print("\nTraining Bagging Classifier...")
start_time = time.time()
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        class_weight='balanced', 
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5
    ),
    n_estimators=150,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train_res, y_train_res)
training_times['Bagging'] = time.time() - start_time
models['Bagging'] = bagging
print(f"Bagging training completed in {training_times['Bagging']:.2f} seconds")

# 5. Gradient Boosting with class weights
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
gb.fit(X_train_res, y_train_res, sample_weight=class_weights)
training_times['Gradient Boosting'] = time.time() - start_time
models['Gradient Boosting'] = gb
print(f"Gradient Boosting training completed in {training_times['Gradient Boosting']:.2f} seconds")

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
            
            plt.subplot(2, 3, i)
            plt.barh(range(10), importances[indices][:10], align='center')
            plt.yticks(range(10), [required_cols[idx] for idx in indices[:10]])
            plt.xlabel('Feature Importance')
            plt.title(f'{name} - Top 10 Features')
    except Exception as e:
        print(f"Could not get feature importances for {name}: {str(e)}")

plt.tight_layout()
plt.show()

# 保存最佳模型
best_model_name = comparison.index[0]
best_model = models[best_model_name]
print(f"\nBest model identified: {best_model_name} with weighted F1: {comparison.loc[best_model_name, 'f1_weighted']:.4f}")

# 这里可以添加模型保存代码，例如：
# import joblib
# joblib.dump(best_model, 'best_classifier.pkl')
# joblib.dump(scaler, 'scaler.pkl')