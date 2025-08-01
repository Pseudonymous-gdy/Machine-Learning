import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"{model_name} evaluation:")
    print('-'*50)
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("Weighted F1 score:", f1_score(y_test, y_pred, average='weighted'))
    print("Macro average F1 score:", f1_score(y_test, y_pred, average='macro'))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), 
                yticklabels=np.unique(y))
    plt.title(f'{model_name} confusion matrix')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()
    
    return y_pred

try:
    df = pd.read_csv('preprocessed_data.csv')
    print("dataset read successfully! len of dataset:", len(df))
except Exception as e:
    print("Error reading the CSV file:", e)
    exit()

label = 'sfdm2'

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

df['sfdm2'] = df['sfdm2'].replace({
    1.0: 3.0,  # 将类别1合并到类别3
    2.0: 3.0   # 将类别2合并到类别3
})

X = df[required_cols]
y = df[label]

# 检查类别数量
n_classes = len(np.unique(y))
print(f"\ntarget '{label}' has {n_classes} classes")
print("distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 添加SMOTE过采样（处理不平衡数据）
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE - train set distribution:", np.bincount(y_train))
print(f"\ntrain set size: {len(X_train)}, test set size: {len(X_test)}")
print("train set distribution:", np.bincount(y_train))
print("test set distribution:", np.bincount(y_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

models = {}

# Bagging

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(class_weight='balanced', max_depth=10),  # 增加树深度
    n_estimators=300, # 基学习器数量
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
models['Bagging'] = bagging

# Random Forest
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=15,  # 增加深度
    class_weight='balanced_subsample',
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train, sample_weight=class_weights)
models['Gradient Boosting'] = gb

# XGBoost (移除过时参数)
# 添加自定义权重函数
def calculate_class_weights(y):
    class_counts = np.bincount(y)
    total_samples = len(y)
    return total_samples / (len(class_counts) * class_counts)

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),  # 动态设置类别数
    n_estimators=300,  # 增加树的数量
    learning_rate=0.05,  # 降低学习率
    max_depth=4,        # 减小深度防止过拟合
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,      # 添加L1正则化
    reg_lambda=1,       # 添加L2正则化
    scale_pos_weight=calculate_class_weights(y_train),  # 自定义权重
    eval_metric='merror',
    random_state=42
)
xgb_model.fit(X_train, y_train, sample_weight=class_weights)
models['XGBoost'] = xgb_model

# 评估所有模型
results = {}
for name, model in models.items():
    y_pred = evaluate_model(model, X_test, y_test, name)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro')
    }

# 模型比较
print("\nevaluation:")
print("=" * 50)
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
            plt.subplot(2, 2, i)
            indices = np.argsort(importances)[::-1]
            plt.title(f'{name} - feature importances')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), indices)
            plt.tight_layout()
    except Exception as e:
        print(f"无法获取 {name} 的特征重要性: {str(e)}")

plt.show()