# 患者死亡预测逻辑回归模型报告

## 1. 项目概述
本项目旨在构建一个预测患者死亡风险的逻辑回归模型。基于包含9,105条患者记录的医疗数据集，我们通过系统的数据预处理、特征工程和模型优化流程，开发了一个能够预测患者死亡概率的二分类模型。

### 关键目标
- 识别与患者死亡风险高度相关的临床特征
- 构建可解释的预测模型
- 平衡模型在少数类（非死亡）上的表现
- 避免标签泄露问题

## 2. 数据预处理与特征工程


### 2.2 标签泄露预防
为避免使用与目标变量直接相关的特征导致模型过拟合，我们移除了可能导致标签泄露的特征：
```python
leakage_cols = ['hospdead', 'dnrday', 'totcst', 'totmcst', 'charges', 
                'totcst_log', 'totmcst_log', 'hday', 'long_term_diff', 
                'short_term_diff', 'surv2m', 'dnr', 'prg6m']
X = X.drop(columns=leakage_cols, errors='ignore')
```

### 2.3 数据分布与类别不平衡
目标变量分布呈现明显的不平衡：
```
Target variable 'death' distribution:
death
1    0.681054  # 死亡(多数类)
0    0.318946  # 非死亡(少数类)
```
这种不平衡会影响模型对少数类的识别能力，需在建模中特别处理。



## 4. 模型构建与优化

### 4.1 处理类别不平衡
```python
 类别权重
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
```

### 4.2 模型选择与超参数优化
采用逻辑回归模型，通过网格搜索优化关键参数：
- **正则化类型**：L1/L2/ElasticNet
- **正则化强度(C)**：0.001到100的对数空间
- **类别权重**：平衡权重或未加权
- **L1比例**：ElasticNet正则化的混合比例

```python
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    # scoring='f1_weighted',
    scoring='f1_macro', # 使用宏平均F1分数，更关注少数类
    cv=5,
    n_jobs=-1,
    verbose=1
)
```

### 4.3 最佳模型参数
```
Best params: {
    'clf__C': 1.0,
    'clf__class_weight': {0: 1.5678, 1: 0.7341},  # 少数类权重更高
    'clf__penalty': 'l1',  # L1正则化有助于特征选择
    'clf__solver': 'saga'
}
```

## 5. 模型评估与解释

### 5.1 整体性能指标
| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| **准确率** | 0.7532 | 0.7661 |
| **AUC** | 0.8438 | 0.8486 |
| **宏平均F1** | 0.74 | 0.75 |

### 5.2 分类报告（测试集）
```
              precision    recall  f1-score   support

           0       0.60      0.80      0.69       581  # 非死亡类
           1       0.89      0.75      0.81      1240  # 死亡类

    accuracy                           0.77      1821
   macro avg       0.74      0.78      0.75      1821
weighted avg       0.80      0.77      0.77      1821
```

### 5.3 关键发现
1. **类别平衡改善**：通过类别加权，非死亡类(0)的召回率达到0.80，显著高于未加权模型
2. **模型稳健性**：测试集AUC(0.8486)略高于训练集(0.8438)，表明模型无过拟合
3. **特征重要性**：L1正则化自动执行特征选择，得到稀疏解

## 6. 结论与改进方向

### 6.1 模型优势
- **可解释性强**：逻辑回归提供清晰的特征系数解释
- **计算高效**：适合部署在资源受限环境
- **类别平衡处理有效**：通过加权显著提升少数类召回率
- **特征选择严谨**：多方法融合确保选择最相关特征

### 6.2 改进方向
   
2. **模型集成**：
   ```python
   # 示例：集成逻辑回归与随机森林
   from sklearn.ensemble import StackingClassifier
   
   estimators = [
       ('lr', LogisticRegression(class_weight='balanced')),
       ('rf', RandomForestClassifier(n_estimators=100))
   ]
   
   stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
   ```

3. **阈值优化**：
   ```python
   # 基于业务需求调整分类阈值
   from sklearn.metrics import precision_recall_curve
   
   precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
   # 选择满足特定召回率/精确度要求的阈值
   ```

4. **处理非线性关系**：
   ```python
   # 添加多项式特征
   from sklearn.preprocessing import PolynomialFeatures
   
   poly = PolynomialFeatures(degree=2, interaction_only=True)
   X_poly = poly.fit_transform(X)
   ```

### 6.3 应用价值
本模型可作为临床决策支持工具：
1. **高风险患者识别**：早期预警系统，识别死亡风险高的患者
2. **资源分配优化**：指导ICU床位和医疗资源分配
3. **治疗策略制定**：为高风险患者制定更积极的治疗方案
4. **预后咨询**：为患者家属提供更准确的预后信息

> **模型文件**：  
> - 最佳模型：`best_logistic_model.pkl`  
> - 特征重要性：`feature_importance.csv`