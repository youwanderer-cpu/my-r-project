# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 17:11:59 2025
@author: 多纳太岚
"""
#https://www.kaggle.com/competitions/home-credit-default-risk/submissions
#数据集可以从上面地址下载

import pandas as pd
import numpy as np
import os
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns

# 手动指定一个纯英文的路径作为缓存目录，否则最后gridresearch会报错
# 必须先在 C 盘根目录新建一个叫 tmp 的文件夹！
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:\\tmp'

# 定义所有文件的名称列表
path_prefix = "C://Users//多纳太岚//Downloads//home-credit-default-risk" 
file_names = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv"
]

# 创建一个字典来存储所有的 DataFrame
dfs = {}

print("开始读取数据...")

for file in file_names:
    # 构造完整路径
    file_path = os.path.join(path_prefix, file)
    
    # 提取不带后缀的文件名作为 key (例如 'bureau')
    key_name = file.split('.')[0]
    
    try:
        # 读取 CSV 文件
        dfs[key_name] = pd.read_csv(file_path, encoding='utf-8')
        
        # 打印读取成功信息及数据形状
        print(f"成功读取: {key_name:<25} 形状: {dfs[key_name].shape}")
        
    except FileNotFoundError:
        print(f"警告: 文件未找到 - {file}")

print("\n所有文件读取完成。")

# 为了构造特征先把训练集与测试集合并
df = pd.concat([dfs['application_train'], dfs['application_test']], ignore_index=True)
print(f"主表初始形状 (Train + Test): {df.shape}")

print("正在对主表 (Application) 进行 One-Hot 编码...")

# 识别二值特征 (只有两个值的类别，比如 'Y'/'N')
# 通常我们用 factorize 把它们转成 0/1，比 One-Hot 少产生一列
for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    df[bin_feature], uniques = pd.factorize(df[bin_feature])

# 对剩余的类别特征进行 One-Hot 编码
# 这里的 dummy_na=False 通常主表里不需要把 NaN 单独作为一列（除非缺失本身有强业务含义）
# 但为了保险起见，可以设为 True
df = pd.get_dummies(df, dummy_na=True)

print(f"主表编码完成。当前形状: {df.shape}")

# 为了演示清晰，定义一个通用的聚合合并函数
# 定义通用处理函数：自动 One-Hot 编码并聚合
def process_and_merge(main_df, aux_df, group_col, prefix):
    """
    main_df: 主表
    aux_df: 辅表 (如 bureau, prev)
    group_col: 连接键 (如 SK_ID_CURR)
    prefix: 列名前缀 (如 BUREAU, PREV)
    """
    # 对辅表进行 One-Hot 编码 (保留文本信息)
    # dummy_na=True 会把 NaN 也当成一种类别处理，防止信息丢失
    aux_df = pd.get_dummies(aux_df, dummy_na=True)
    
    # 区分两类列：原始数值列 vs One-Hot生成的列
    # 逻辑：One-Hot 生成的列只有 0 和 1，适合算 mean(占比) 和 sum(次数)
    # 原始数值列(如金额) 适合算 max, min, mean 等
    
    # 简单的区分方法：列名里如果包含原来的类别值，就是 One-Hot 列
    
    agg_dict = {}
    for col in aux_df.columns:
        if col == group_col:
            continue 
        # 如果列名在原始列里且是数字，应用全套统计
        # 这里为了简化和覆盖全面，对所有列应用常用统计
        agg_dict[col] = ['min', 'max', 'mean', 'sum', 'var']
    
    # 聚合
    aux_agg = aux_df.groupby(group_col).agg(agg_dict)
    
    # 扁平化列名
    aux_agg.columns = [prefix + '_' + col[0] + '_' + col[1].upper() for col in aux_agg.columns]
    aux_agg.reset_index(inplace=True)
    
    # 合并
    main_df = main_df.merge(aux_agg, on=group_col, how='left')
    
    print(f"合并 {prefix} 完成。当前特征数: {main_df.shape[1]}")
    return main_df

# 开始批量处理所有表格

# Bureau & Bureau Balance (特殊处理二级关系)
if 'bureau' in dfs and 'bureau_balance' in dfs:
    bureau = dfs['bureau'].copy()
    bb = dfs['bureau_balance'].copy()
    
    # 先处理 Bureau Balance (One-Hot + Agg)
    bb = pd.get_dummies(bb, dummy_na=True)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(['min', 'max', 'mean', 'size'])
    bb_agg.columns = ['BB_' + col[0] + '_' + col[1].upper() for col in bb_agg.columns]
    
    # 合并到 Bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # 再处理 Bureau (这里会自动处理 CREDIT_ACTIVE, CREDIT_TYPE 等文本)
    df = process_and_merge(df, bureau, 'SK_ID_CURR', 'BUREAU')
    
    del bureau, bb, bb_agg
    gc.collect()

# Previous Applications (之前的申请记录)
if 'previous_application' in dfs:
    prev = dfs['previous_application'].copy()
    # 简单的特征衍生
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    # 自动处理 NAME_CONTRACT_STATUS, CODE_REJECT_REASON 等所有文本
    df = process_and_merge(df, prev, 'SK_ID_CURR', 'PREV')
    
    del prev
    gc.collect()

# POS_CASH_balance
if 'POS_CASH_balance' in dfs:
    pos = dfs['POS_CASH_balance'].copy()
    # 自动处理 NAME_CONTRACT_STATUS 等
    df = process_and_merge(df, pos, 'SK_ID_CURR', 'POS')
    del pos
    gc.collect()

# Installments Payments
if 'installments_payments' in dfs:
    ins = dfs['installments_payments'].copy()
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    
    df = process_and_merge(df, ins, 'SK_ID_CURR', 'INSTAL')
    del ins
    gc.collect()

# Credit Card Balance
if 'credit_card_balance' in dfs:
    cc = dfs['credit_card_balance'].copy()
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    
    df = process_and_merge(df, cc, 'SK_ID_CURR', 'CC')
    del cc
    gc.collect()

print(f"\n最终处理完成！DataFrame 形状: {df.shape}")

pd.DataFrame(df.columns, columns=['feature_name']).to_csv('features.csv', index=False)

# 找到所有类型为 'object' 的列，防止xgboost报错，xgboost无法处理object
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"发现 {len(object_cols)} 个 object 类型的列，正在尝试转换...")

# 批量转换为数值型
# errors='ignore' 表示如果真的遇到纯文本（无法转数字），就跳过
for col in object_cols:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass # 如果真的转不了，就保持原样（后续再处理）

# 再次检查是否还有遗留的 object 列
remaining_objects = df.select_dtypes(include=['object']).columns.tolist()
if len(remaining_objects) > 0:
    print(f"警报：仍有 {len(remaining_objects)} 列是 object 类型 (可能是未编码的字符串):")
    print(remaining_objects[:5])
    
    # 如果这些列是类别（如 'CODE_GENDER'），需要做 One-Hot 编码
    # df.drop(columns=remaining_objects, inplace=True) 
else:
    print("所有 object 列都已修复为数值型。")


# 筛选出数值类型的列（只有数字才会有 inf）并处理xgboost无法处理的inf情况
numeric_df = df.select_dtypes(include=[np.number])

# 检查正无穷 (inf) 和负无穷 (-inf)
# np.isinf(df) 会返回一个全是 True/False 的表格
# .sum() 会统计每列有多少个 True
inf_counts = np.isinf(numeric_df).sum()

# 只保留那些 inf 数量大于 0 的列
cols_with_inf = inf_counts[inf_counts > 0]

# 打印结果
if len(cols_with_inf) > 0:
    print(f"发现 {len(cols_with_inf)} 个包含无穷大的列：\n")
    # 打印列名和对应的 inf 数量
    print(cols_with_inf.sort_values(ascending=False)) 
else:
    print("数据集中没有发现无穷大 (inf) 值。")

# 构造筛选条件：只要这几列里任意一列是 inf，就选出来
# np.isinf(df[problem_cols]) 会返回 True/False 矩阵
# .any(axis=1) 表示只要这一行里有一个 True，整行就为 True

target_cols = cols_with_inf.index.tolist()
condition = np.isinf(df[target_cols]).any(axis=1)

# 打印出问题的样本
# 展示 ID 和那几列的值
inf_rows = df.loc[condition, ['SK_ID_CURR'] + target_cols]
print(f"找到 {len(inf_rows)} 个包含 inf 的样本：")
print(inf_rows)
# 将问题样本的inf更换成NaN
df.loc[condition, target_cols] = df.loc[condition, target_cols].replace([np.inf, -np.inf], np.nan)
inf_rows = df.loc[condition, ['SK_ID_CURR'] + target_cols]
print(f"修复 {len(inf_rows)} inf 的样本：")
print(inf_rows)

# 拆分训练集和测试集
# 逻辑：Test 集在合并时 TARGET 会变成 NaN，利用这一点拆分
print("正在拆分训练集与测试集...")
train_df = df[df['TARGET'].notnull()].copy()
train_df['DAYS_EMPLOYED'] = train_df['DAYS_EMPLOYED'].replace(365243, np.nan)
test_df = df[df['TARGET'].isnull()].copy()

df[df['TARGET'].notnull()]['DAYS_EMPLOYED'].describe()

#train_df.to_csv('train_df.csv')

# Test 集不需要 TARGET 列 
test_df.drop(columns=['TARGET'], inplace=True)

print(f"拆分完成 -> Train: {train_df.shape}, Test: {test_df.shape}")

# 确定特征列 (排除 ID 和 Label)
# 我们只在这些列上进行计算
feature_cols = [col for col in train_df.columns if col not in ['SK_ID_CURR', 'TARGET']]

# 用一个列表来记录所有要删除的列名
cols_to_drop = []

# 筛选规则 A: 剔除缺失率过高的列 (基于 Train)
print("\n[规则 A] 正在检查高缺失值特征...")

# 只在 train_df 上计算缺失率
train_missing = train_df[feature_cols].isnull().mean()
threshold = 0.70 # 阈值 70%

high_missing_cols = train_missing[train_missing > threshold].index.tolist()
cols_to_drop.extend(high_missing_cols)

print(f"  - 发现 {len(high_missing_cols)} 个特征缺失率 > {threshold}")

# 筛选规则 B: 剔除单值/零方差列 (基于 Train)
print("\n[规则 B] 正在检查单值(无区分度)特征...")

# 只在 train_df 上计算唯一值数量
# nunique <= 1 意味着所有样本在这个特征上值都一样，或者全是 NaN
nunique = train_df[feature_cols].nunique(dropna=False) 
single_value_cols = nunique[nunique <= 1].index.tolist()
cols_to_drop.extend(single_value_cols)

print(f"  - 发现 {len(single_value_cols)} 个特征只有一个数值")

# ==========================================
# 筛选规则 C: 剔除高相关特征 (基于 Train)
# ==========================================
print("\n[规则 C] 正在检查高共线性特征 (Collinear)...")

# 注意：计算整个 Train 的相关性矩阵非常慢且耗内存
# 技巧：随机采样 20,000 个 Train 样本来估算相关性，足够准确且速度快
sample_size = 20000
if len(train_df) > sample_size:
    corr_sample = train_df[feature_cols].sample(sample_size, random_state=42)
else:
    corr_sample = train_df[feature_cols]

# 计算相关性矩阵
corr_matrix = corr_sample.corr().abs()

# 选取上三角矩阵 (避免重复计算 A-B 和 B-A)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 找出相关性 > 0.80 的列
high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.80)]
cols_to_drop.extend(high_corr_cols)

print(f"  - 发现 {len(high_corr_cols)} 个特征相关性 > 0.80")

# 执行删除 (同时应用到 Train 和 Test)
# 去重 (防止某一列同时满足多个删除条件)
final_drop_list = list(set(cols_to_drop))

print(f"\n准备删除总计 {len(final_drop_list)} 个特征...")

# 在 Train 上删除
train_df.drop(columns=final_drop_list, inplace=True, errors='ignore')
# 在 Test 上删除 (必须删掉完全一样的列)
test_df.drop(columns=final_drop_list, inplace=True, errors='ignore')

print("特征筛选完成！")
print(f"最终 Train 形状: {train_df.shape}")
print(f"最终 Test 形状: {test_df.shape}")

# 内存清理
del corr_matrix, upper, corr_sample
gc.collect()

# 准备数据
X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train_df['TARGET']


X_test = test_df.drop(columns=['SK_ID_CURR']) # 测试集数据

# 定义 KFold
# n_splits=5 表示 5 折 CV
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储结果
oof_preds = np.zeros(X.shape[0])       # 训练集的 OOF 预测 (用于算本地 CV 分数)
sub_preds = np.zeros(X_test.shape[0])  # 测试集的预测 (用于提交)
feature_importance_df = pd.DataFrame() # 存储特征重要性

# 开始循环训练
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"--- Fold {n_fold + 1} / 5 ---")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=1000,   # 树的数量：给一个很大的值，靠 early_stopping 自动停
        learning_rate=0.1,  # 学习率：越小越好，但训练越慢。0.01~0.1 之间，gridresearch 0.1 最好
        max_depth=3,     # 树深：默认6。过深容易过拟合，gridresearch 3 最好
        subsample=0.8,   # 行采样：每棵树只用 80% 的样本
        min_child_weight= 1,  # gridresearch 1 最好
        colsample_bytree=0.8,  # 列采样：每棵树只用 80% 的特征 (这个对多特征数据集极其重要！)
        scale_pos_weight=11,    # 核心参数！正负样本的权重比。计算公式：(负样本数 / 正样本数)。Home Credit 大概是 11 倍。
        reg_alpha=0.5,    #正则化L1参数
        objective='binary:logistic',
        n_jobs=-1,
        tree_method='hist',
        device='cuda',  # 如果有GPU
        random_state=42,
        early_stopping_rounds=100,
        eval_metric='auc', 
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    
    oof_preds[valid_idx] = model.predict_proba(X_val)[:, 1]
    sub_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
    
    print(f"Fold {n_fold + 1} AUC: {roc_auc_score(y_val, oof_preds[valid_idx]):.5f}")


    # 把这一折的特征重要性存起来
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X.columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = n_fold + 1

    useless_features = fold_importance[fold_importance['importance'] == 0]
    useful_features = fold_importance[fold_importance['importance'] > 0]

    print(f"总特征数: {len(fold_importance)}")
    print(f"被模型弃用的特征数 (Importance=0): {len(useless_features)}")
    print(f"保留的有效特征数: {len(useful_features)}")
    
    # 拼接到总表中
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# 循环结束，统一查看结果
cv_auc = roc_auc_score(y, oof_preds)
print(f"\n整体 CV AUC: {cv_auc:.5f}")

# 计算 5 折的平均重要性并显示
print("\n--- Top 20 特征重要性 (5折平均) ---")
best_features = feature_importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
print(best_features.head(20))
best_features.to_csv(f'best_features_5fold_cv_{cv_auc:.4f}.csv')

# 可视化 Top 30
plt.figure(figsize=(10, 8))
# 选取前 30 个最重要的特征
cols = best_features.head(30).index
best_features_data = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

sns.barplot(x="importance", y="feature", data=best_features_data, order=cols)
plt.title('XGBoost Features (avg over folds)')
plt.tight_layout()
plt.show()

# 生成提交文件
submission = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'],
    'TARGET': sub_preds  # 这里已经是 5 个模型的平均值了
})
submission.to_csv(f'submission_5fold_cv_{cv_auc:.4f}.csv', index=False)
print("提交文件已生成！")



# 以下是GridSearchCV的代码示例，当前未启用
# # 划分验证集 (必须要有，用于 Early Stopping)

# model = xgb.XGBClassifier(
    
#     n_estimators=500,         
#     subsample=0.8,              
#     colsample_bytree=0.8,       
#     scale_pos_weight=11,        
    
    
#     objective='binary:logistic',
#     n_jobs=-1,                  
#     tree_method='hist',         
#     random_state=42,
#     device='cuda', 
# )


# param_grid = { 
#     'max_depth': [3, 5, 7], 
#     'min_child_weight': [1, 3, 5], 
#     'learning_rate': [0.01, 0.05, 0.1]
# }

# # 设置交叉验证
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # 配置搜索
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=cv,
#     verbose=3, 
#     n_jobs=1  
# )

# # 开始搜索
# print("Starting Grid Search...")  

# grid_search.fit(X,y)

# # 结果
# print("\n--- 最佳结果 ---")
# print(f"最佳 AUC: {grid_search.best_score_:.4f}")
# print("最佳动态参数:", grid_search.best_params_)
# print("所有参数 (含固定):", grid_search.best_estimator_.get_params())
