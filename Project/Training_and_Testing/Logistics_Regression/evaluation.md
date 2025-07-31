### 模型表现分析

注：模型分析时，将目标类别1-5改为了0-4

从分类报告中可以看出以下关键问题：

1. **严重的类别不平衡问题**：
   - 类别分布极不均衡：类别0.0（936个样本）和4.0（918个样本）占主导
   - 少数类别严重不足：类别1.0（仅11个样本）、2.0（169个样本）、3.0（274个样本）

2. **模型失效的类别**：
   - **类别1.0、2.0、3.0完全未被识别**：
     - 精确率(precision)和召回率(recall)均为0
     - F1-score=0，说明模型完全无法预测这些类别
   - **类别3.0识别能力极弱**：
     - 精确率0.14但召回率0.00，说明预测结果都是误报

3. **评价指标矛盾**：
   - 宏观平均(macro avg) F1=0.28 显示整体表现差
   - 加权平均(weighted avg) F1=0.55 被多数类拉高
   - AUC=0.71 与分类指标不一致，表明模型具有排序能力但缺乏分类能力

### 根本原因诊断

1. **数据层面**：
   - **样本量不足**：类别1.0仅11个样本，无法训练有效模型
   - **特征区分度不足**：当前特征无法区分少数类别
   - **类别定义问题**：可能存在需要合并的小类别

2. **模型选择问题**：
   - 逻辑回归本质是线性模型，难以处理复杂决策边界
   - 当前参数设置（正则化强度C）可能不适合少数类

3. **评估指标误导**：
   - 使用AUC作为网格搜索指标，但实际需要关注recall/F1

### 改进方案

#### 紧急措施（立即见效）
```python
# 修改网格搜索评估指标 -> 关注少数类的F1-score
grid_search = GridSearchCV(
    scoring='f1_weighted',  # 改用加权F1-score
    # 其他参数保持不变...
)

# 强制关注少数类 - 自定义类别权重
class_weights = {
    0.0: 1, 
    1.0: 50,  # 大幅提高权重
    2.0: 10,
    3.0: 10,
    4.0: 1
}

param_grid = [
    {
        'clf__class_weight': [None, 'balanced', class_weights]  # 添加自定义权重
    },
    # 其他参数网格...
]
```

#### 中期解决方案
1. **数据重构**：
   ```python
   # 合并小类别（根据医学意义）
   df['sfdm2'] = df['sfdm2'].replace({
       1.0: 3.0,  # 将类别1合并到类别3
       2.0: 3.0   # 将类别2合并到类别3
   })
   
   # 检查合并后分布
   print(np.bincount(df['sfdm2']))
   ```

2. **采样策略优化**：
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.pipeline import Pipeline  # 改用imbalanced-learn的pipeline
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('sampler', SMOTE()),  # 添加过采样
       ('clf', LogisticRegression(...))
   ])
   ```

#### 长期根本性改进
1. **模型替换方案**：
   ```python
   # 方案1：改用树形模型（自动处理不平衡数据）
   from sklearn.ensemble import RandomForestClassifier
   pipeline.steps[1] = ('clf', RandomForestClassifier(class_weight='balanced'))
   
   # 方案2：梯度提升树（GBDT）
   from xgboost import XGBClassifier
   pipeline.steps[1] = ('clf', XGBClassifier(scale_pos_weight=compute_weights()))
   ```

2. **重新定义问题**：
   ```python
   # 将多分类转为二分类（生存/非生存）
   df['survival'] = df['sfdm2'].apply(lambda x: 1 if x in [0.0, 4.0] else 0)
   ```
   理由：原始数据中0.0和4.0占83%，可能代表核心业务场景

### 实施路线图

| 阶段 | 行动项 | 预期效果 | 实施难度 |
|------|--------|----------|----------|
| 紧急修复 | 调整类别权重 + 修改评估指标 | 少数类F1提升20% | 低（1小时） |
| 数据重构 | 合并医学意义相近的类别 | 减少类别数，增加样本 | 中（2小时） |
| 采样优化 | 实现SMOTE采样管道 | 平衡各类别样本量 | 中（3小时） |
| 模型升级 | 切换到XGBoost/RF | 综合性能提升30% | 高（1天） |

> **关键建议**：优先执行类别合并（dzgroup/dzclass可能提供合并依据），同时切换为加权F1评估。当前AUC=0.71证明特征信息量足够，问题核心在于模型未能利用这些信息区分少数类。