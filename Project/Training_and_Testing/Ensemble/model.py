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

# 过滤未来警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. 创建不平衡的多分类数据集
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_classes=5,
    weights=[0.40, 0.25, 0.15, 0.12, 0.08],
    flip_y=0.1,
    random_state=42
)

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 定义评估函数
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"{model_name} 性能评估:")
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

# 3. 计算类别权重
class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 5. 定义并训练集成模型
models = {}

# 5.1 Bagging

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(class_weight='balanced', max_depth=10),  # 增加树深度
    n_estimators=200,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
models['Bagging'] = bagging

# 5.2 Random Forest
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,  # 增加深度
    class_weight='balanced_subsample',
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

# 5.3 Gradient Boosting
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

# 5.4 XGBoost (移除过时参数)
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    # 移除 use_label_encoder 参数
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train, y_train, sample_weight=class_weights)
models['XGBoost'] = xgb_model

# 6. 评估所有模型
results = {}
for name, model in models.items():
    y_pred = evaluate_model(model, X_test, y_test, name)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro')
    }

# 7. 模型比较
print("\nevaluation:")
print("=" * 50)
comparison = pd.DataFrame(results).T
comparison = comparison.sort_values(by='f1_weighted', ascending=False)
print(comparison)

# 8. 特征重要性可视化
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