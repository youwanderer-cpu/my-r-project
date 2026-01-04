import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.ticker as mtick
import seaborn as sns
import itertools
# ==========================================
# 1. 数据预处理与权重计算
# ==========================================
def load_and_preprocess_local(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=9)
        if 'Date' not in df.columns:
            df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 提取 1-30 年零息收益率
    yield_cols = [f'SVENY{i:02d}' for i in range(1, 31)]
    valid_cols = [c for c in yield_cols if c in df.columns]
    df_yields = df[valid_cols] / 100.0  
    
    # 截取最近 8 年样本
    end_date = df_yields.index.max()
    start_date = end_date - pd.DateOffset(years=8)
    sample = df_yields.loc[start_date:end_date].copy()
    
    # 应用 0.8 衰减因子权重
    T = len(sample)
    days_from_end = np.arange(T)[::-1]
    sample['Weight'] = 0.8 ** (days_from_end / 252.0)
    
    return sample
# ==========================================
# 1. 核心载荷与映射矩阵定义 (Cascade Form)
# ==========================================
def B_tau(tau, alpha):
    if tau < 1e-6: return 1.0
    return (1 - np.exp(-alpha * tau)) / (alpha * tau)

def get_A_matrix(a_vec):
    """级联模型特征矩阵 A: 确保因子 m 和 l 也是利率量级"""
    as_, am, al = a_vec
    # 级联结构映射：r_t = x1 + x2 + x3
    A = np.array([
        [1.0, as_/(as_-am), (as_*am)/((as_-al)*(am-al))],
        [0.0, 1.0,          am/(am-al)],
        [0.0, 0.0,          1.0]
    ])
    return A

def get_Upsilon_all(tau, a_vec):
    """计算总载荷向量 Upsilon = B(tau) * inv(A)"""
    A_inv = np.linalg.inv(get_A_matrix(a_vec))
    B_vec = np.array([B_tau(tau, a) for a in a_vec])
    return B_vec @ A_inv

def get_C_term(tau, a_vec, s_vec, rho):
    """
    计算模型中的凸性修正项 C(tau, alpha, sigma)
   
    """
    as_, am, al = a_vec
    sm, sl = s_vec 
    
    # 1. 构造映射矩阵 A 及其逆矩阵
    A_inv = np.linalg.inv(get_A_matrix(a_vec))
    
    # 2. 构造简化形式的波动率协方差矩阵 (针对中期 m 和长期 l 因子)
    # 根据级联模型定义，短期因子的随机项通常合并在 m 中或忽略
    cov_reduced = np.zeros((3, 3))
    cov_reduced[1, 1] = sm**2
    cov_reduced[2, 2] = sl**2
    cov_reduced[1, 2] = cov_reduced[2, 1] = rho * sm * sl
    
    # 3. 变换为因子空间的协方差矩阵 Sigma_matrix
    # Sigma = A_inv @ Cov_reduced @ A_inv.T
    sigma_matrix = A_inv @ cov_reduced @ A_inv.T 
    
    # 4. 根据公式 A9.11 计算双重求和项
    c_val = 0
    alphas = [as_, am, al]
    for i in range(3):
        for j in range(3):
            bi = B_tau(tau, alphas[i])
            bj = B_tau(tau, alphas[j])
            # bij = (1 - exp(-(ai + aj) * tau)) / ((ai + aj) * tau)
            bij = (1 - np.exp(-(alphas[i] + alphas[j]) * tau)) / ((alphas[i] + alphas[j]) * tau)
            
            # 凸性项分量
            term = (sigma_matrix[i, j] / (2 * alphas[i] * alphas[j])) * (1 - bi - bj + bij)
            c_val += term
            
    return c_val
# ==========================================
# 2. 估计 Alpha (强制 0.4 的硬间距防止爆炸)
# ==========================================
def estimate_alpha_robust(df_sample):
    maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    dy_full = df_sample[[f'SVENY{m:02d}' for m in maturities]].diff()
    temp_df = dy_full.copy()
    temp_df['w'] = df_sample['Weight']
    temp_df = temp_df.dropna()
    dy = temp_df[[f'SVENY{m:02d}' for m in maturities]]
    weights = temp_df['w'].values
    dy_bench = dy[['SVENY02', 'SVENY10']]
    
    beta_hat = []
    for col in dy.columns:
        model = sm.WLS(dy[col], dy_bench, weights=weights).fit()
        beta_hat.append(model.params.values)
    beta_hat = np.array(beta_hat)

    def objective(a):
        if not (a[0] > a[1]  and a[1] > a[2] + 0.05 and a[2] > 0.001): 
            return 1e10
        try:
            ups_ml_all = np.array([get_Upsilon_all(tau, a)[1:] for tau in maturities])
            ups_b = ups_ml_all[[1, 5], :] 
            if np.linalg.cond(ups_b) > 500: return 1e10
            model_slopes = ups_ml_all @ np.linalg.inv(ups_b)
            return np.sum((model_slopes - beta_hat)**2)
        except:
            return 1e10

    res = minimize(objective, x0=[1.2, 0.4, 0.02], method='Nelder-Mead')
    return res.x

# ==========================================
# 3. 稳健波动率估计 (匹配加权方差)
# ==========================================
def estimate_volatility_robust(df_sample, opt_alpha):
    maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    dy = df_sample[[f'SVENY{m:02d}' for m in maturities]].diff()
    temp_df = dy.copy()
    temp_df['w'] = df_sample['Weight']
    temp_df = temp_df.dropna()
    
    realized_vars = []
    for m in maturities:
        col = f'SVENY{m:02d}'
        mean = np.average(temp_df[col], weights=temp_df['w'])
        var = np.average((temp_df[col] - mean)**2, weights=temp_df['w']) * 252
        realized_vars.append(var)
    realized_vars = np.array(realized_vars)

    def objective_sigma(params):
        sm, sl, rho = params
        # 强制波动率在合理区间 (50bps - 250bps)，防止其漂移到 1500bps
        if not (0.005 < sm < 0.025 and 0.001 < sl < 0.015 and 0.2 < rho < 0.95):
            return 1e10
        cov = np.array([[sm**2, rho*sm*sl], [rho*sm*sl, sl**2]])
        ups_ml_all = np.array([get_Upsilon_all(tau, opt_alpha)[1:] for tau in maturities])
        model_vars = np.diag(ups_ml_all @ cov @ ups_ml_all.T)
        return np.sum((model_vars - realized_vars)**2)

    res = minimize(objective_sigma, x0=[0.012, 0.004, 0.7], method='Nelder-Mead')
    return res.x

# ==========================================
# 4. 因子提取 (修复 NameError: data)
# ==========================================
def extract_final_fig98(df_sample, a_vec, s_vec, rho, mu, gsw_file, fred_data):
    """增加了 fred_data 参数，解决 name 'data' is not defined 错误"""
    # 获取 2y 和 10y 远期利率
    df_gsw = pd.read_csv(gsw_file, skiprows=9)
    df_gsw['Date'] = pd.to_datetime(df_gsw['Date'])
    df_gsw.set_index('Date', inplace=True)
    f_mkt = df_gsw[['SVENF02', 'SVENF10']] / 100.0
    
    # 使用传入的 fred_data 中的短期利率
    combined = f_mkt.join(fred_data['short_rate'], how='inner').dropna()
    
    # 远期载荷 L = exp(-a*tau)
    def get_fwd_L(tau):
        e_vec = np.array([np.exp(-a_vec[i]*tau) for i in range(3)])
        return e_vec @ np.linalg.inv(get_A_matrix(a_vec))

    L2, L10 = get_fwd_L(2), get_fwd_L(10)
    # 求解 m 和 l
    inv_mat = np.linalg.inv(np.array([L2[1:], L10[1:]]))
    
    results = []
    for idx, row in combined.iterrows():
        rt = row['short_rate']
        # f_obs = Lr*r + Lm*m + Ll*l (mu 包含在因子水平中)
        target = np.array([row['SVENF02'] - L2[0]*rt, row['SVENF10'] - L10[0]*rt])
        m_l = inv_mat @ target
        results.append([rt, m_l[0], m_l[1], row['SVENF02']])
        
    return pd.DataFrame(results, index=combined.index, columns=['Short', 'Medium', 'Long', '2yr Fwd'])


def estimate_mu_step_proper_weighted(df_sample, a_vec, s_vec, rho):
    """
    根据讲义 A9.2.2 第三步：匹配加权平均收益率水平
    使用 df_sample['Weight'] 确保 mu 反映的是近期加权后的市场中轴
    """
    maturities = [1, 2, 5, 7, 10, 20, 30]
    weight_vec = df_sample['Weight'].values
    
    # 核心修改：计算加权平均收益率 (Weighted Average Yields)
    avg_market_yields = []
    for m in maturities:
        col = f'SVENY{m:02d}'
        # 计算该期限在 8 年样本中的加权平均值
        w_avg = np.average(df_sample[col], weights=weight_vec)
        avg_market_yields.append(w_avg)
    avg_market_yields = np.array(avg_market_yields)
    
    def objective_mu(mu_candidate):
        # 计算凸性修正项 C(tau)
        c_terms = np.array([get_C_term(tau, a_vec, s_vec, rho) for tau in maturities])
        
        # 模型预测的加权中轴：y_base = mu - C(tau)
        model_avg_yields = mu_candidate - c_terms
        
        # 最小化加权误差平方和
        return np.sum((avg_market_yields - model_avg_yields)**2)

    # 初始猜测：以最近一天的 10 年期利率作为初始参考值
    initial_guess = df_sample['SVENY10'].iloc[-1]
    res = minimize(objective_mu, x0=[initial_guess], bounds=[(0.01, 0.08)])
    return res.x[0]
# ==========================================
# 5. 执行流程
# ==========================================
# 加载 FRED 数据 (用于提取 short_rate)
data = pd.read_excel('GaussPlus_FRED_Data.xlsx', index_col=0, parse_dates=True)

# 加载并处理 GSW 数据 (用于估计参数)
sample_data = load_and_preprocess_local('feds200628.csv')

# 估计参数
alpha_params = estimate_alpha_robust(sample_data)
opt_sigma = estimate_volatility_robust(sample_data, alpha_params)
final_mu = estimate_mu_step_proper_weighted(sample_data, alpha_params, opt_sigma[:2], opt_sigma[2])

print(f"\n--- 参数估计成功 ---")
print(f"回归速度 Alpha: {alpha_params}")
print(f"波动率 Sigma (bps): {opt_sigma[:2]*10000}")
print(f"长期均值 Mu: {final_mu*100:.2f}%")

# 4. 提取因子 (传入估计出的 final_mu)
# 注意：在提取因子函数中，mu 作为基准水平参与计算
plot_df = extract_final_fig98(sample_data, alpha_params, opt_sigma[:2], opt_sigma[2], final_mu, 'feds200628.csv', data)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(plot_df['Long'], label=f'Long Factor (Mean={final_mu*100:.2f}%)', color='blue', linewidth=2)
plt.plot(plot_df['Medium'], label='Medium Factor', color='orange', alpha=0.7)
plt.plot(plot_df['Short'], label='Short Rate (r_t)', color='red', alpha=0.4)
plt.plot(plot_df['2yr Fwd'], label='2yr Fwd', color='black', linestyle='--', alpha=0.6)
# --- 核心修改部分 ---
ax = plt.gca()  # 获取当前的坐标轴
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # 将 1.0 映射为 100%
# --------------------

plt.title("The Two-Year Forward Rate and Gauss+ Factors Extracted from Daily Market Data.")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

plot_df.to_excel('GaussPlus_Factors_Estimated.xlsx', index=True)



# ==========================================
# 1. 准备数据与期限 (修复对齐问题)
# ==========================================
plot_maturities = np.arange(1, 31)
yield_cols = [f'SVENY{m:02d}' for m in plot_maturities]

# 创建差分数据
dy_all = sample_data[yield_cols].diff()

# 核心修复：将权重合并到 DataFrame 中一起 dropna，确保行数完美匹配
temp_df = dy_all.copy()
temp_df['Weight_Aligned'] = sample_data['Weight']
temp_df = temp_df.dropna()  # 这一步会同时过滤掉第一行和中间的任何空值行

# 重新分离 dy 和 weights
dy = temp_df[yield_cols]
weights = temp_df['Weight_Aligned'].values
dy_bench = dy[['SVENY02', 'SVENY10']]

# ==========================================
# 2. 计算市场观察值 (Estimated Betas)
# ==========================================
est_betas = []
for m in plot_maturities:
    col = f'SVENY{m:02d}'
    # 现在 dy[col], dy_bench, weights 的行数都是 1910，不会再报错
    res = sm.WLS(dy[col], dy_bench, weights=weights).fit()
    est_betas.append(res.params.values)
est_betas = np.array(est_betas)

# ==========================================
# 3. 计算模型预测值 (Model Betas)
# ==========================================
def calculate_model_betas(a_vec, mats):
    # 计算全期限载荷矩阵 (只取 m 和 l 因子，对应 A9.2.2 第一步)
    ups_all = np.array([get_Upsilon_all(tau, a_vec) for tau in mats])
    ups_ml = ups_all[:, 1:] # 提取 Medium 和 Long 载荷 (index 1 and 2)
    
    # 提取基准期限 (2y 和 10y) 的载荷矩阵
    ups_bench = np.array([
        get_Upsilon_all(2, a_vec)[1:], 
        get_Upsilon_all(10, a_vec)[1:]
    ])
    
    # 模型 Beta = Ups_all @ inv(Ups_bench)
    return ups_ml @ np.linalg.inv(ups_bench)

model_betas = calculate_model_betas(alpha_params, plot_maturities)


# ==========================================
# 1. 设置绘图参数
# ==========================================
x = np.arange(1, 31)  # 期限 1-30 年
width = 0.4          # 柱子的宽度

plt.figure(figsize=(15, 7))

# ==========================================
# 2. 绘制分组柱状图 (市场估计值)
# ==========================================
# 2年期基准的 Beta 柱子（向左偏移 width/2）
plt.bar(x - width/2, est_betas[:, 0], width, label='Empirical Beta (2-Year)', 
        color='skyblue', edgecolor='navy', alpha=0.7)

# 10年期基准的 Beta 柱子（向右偏移 width/2）
plt.bar(x + width/2, est_betas[:, 1], width, label='Empirical Beta (10-Year)', 
        color='salmon', edgecolor='darkred', alpha=0.7)

# ==========================================
# 3. 叠加模型理论值 (保持折线以显示拟合效果)
# ==========================================
# 如果您希望全都是柱状图，可以注释掉下面两行
plt.plot(x, model_betas[:, 0], color='blue', marker='o', markersize=3, 
         linestyle='-', linewidth=1.5, label='Model Beta (2-Year)')
plt.plot(x, model_betas[:, 1], color='red', marker='s', markersize=3, 
         linestyle='-', linewidth=1.5, label='Model Beta (10-Year)')

# ==========================================
# 4. 图表修饰
# ==========================================
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xlabel('Maturity (Years)', fontsize=12)
plt.ylabel('Beta Value', fontsize=12)
plt.title("Coefficients of Regressing Zero Coupon Bond Yields of Various Terms on Two- and 10-Year Zero Coupon Bond Yields,\n"
          "from Empirical Analysis and as Implied by the Estimated Gauss+ Model", 
          fontsize=14, pad=20)
plt.xticks(np.arange(1, 31)) # 确保 X 轴显示所有年份
plt.legend(loc='best', fontsize=10)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()



# ==========================================
# 1. 准备参数 (使用您 guass2.py 运行出的结果)
# ==========================================
# 假设已经运行了 estimate_alpha_robust 和 estimate_volatility_robust
# alpha_params = [as, am, al]
# opt_sigma = [sm, sl, rho]
a_vec = alpha_params 
sm, sl, rho = opt_sigma 

plot_mats = np.arange(1, 31)

def get_model_volatility(tau, a_vec, sm, sl, rho):
    """计算期限 tau 的模型理论波动率 (bps/year)"""
    # 1. 获取物理因子载荷 Upsilon(tau)
    ups = get_Upsilon_all(tau, a_vec)
    
    # 2. 构造物理因子协方差矩阵 (针对 r, m, l)
    # 记住：Gauss+ 中短期利率 r 没有扰动项
    cov_phys = np.zeros((3, 3))
    cov_phys[1, 1] = sm**2
    cov_phys[2, 2] = sl**2
    cov_phys[1, 2] = cov_phys[2, 1] = rho * sm * sl
    
    # 3. 变换为独立因子空间的协方差矩阵 Sigma
    A_inv = np.linalg.inv(get_A_matrix(a_vec))
    sigma_matrix = A_inv @ cov_phys @ A_inv.T
    
    # 4. 计算总波动率: Vol^2 = Upsilon @ Sigma @ Upsilon.T
    # 这里的 Upsilon 在代码定义中已经是 B @ A_inv
    # 但根据公式，直接用物理载荷对应独立因子的 B 向量即可
    # 为了保持一致性，我们直接用 Upsilon @ Cov_phys @ Upsilon.T
    # (注意：Upsilon 是物理载荷，而我们要算的是总体的变动标准差)
    
    # 更直接的做法：
    # 模型预测 y_tau 变动为 ups[0]*dr + ups[1]*dm + ups[2]*dl
    # 只有 dm 和 dl 有随机项，所以：
    var = (ups[1]**2 * sm**2 + 
           ups[2]**2 * sl**2 + 
           2 * ups[1] * ups[2] * rho * sm * sl)
    
    return np.sqrt(var)

# ==========================================
# 2. 计算实际波动率 (Realized Volatility)
# ==========================================
# 准备差分数据和权重 (利用之前修复的对齐逻辑)
dy_all = sample_data[[f'SVENY{m:02d}' for m in plot_mats]].diff()
temp_df = dy_all.copy()
temp_df['w'] = sample_data['Weight']
temp_df = temp_df.dropna()
weights = temp_df['w'].values / temp_df['w'].mean() # 归一化权重

realized_vols = []
for m in plot_mats:
    col = f'SVENY{m:02d}'
    # 计算加权标准差
    daily_diff = temp_df[col].values
    weighted_var = np.sum(weights * (daily_diff**2)) / np.sum(weights)
    # 年化：日方差 * 252，再开方
    ann_vol = np.sqrt(weighted_var * 252)
    realized_vols.append(ann_vol)

# ==========================================
# 3. 计算模型预测值
# ==========================================
model_vols = [get_model_volatility(tau, a_vec, sm, sl, rho) for tau in plot_mats]

# ==========================================
# 4. 绘图 (Figure 9.7 风格)
# ==========================================
plt.figure(figsize=(10, 6))

# 实际波动率（散点/柱状）
plt.bar(plot_mats, np.array(realized_vols)*10000, color='skyblue', 
        alpha=0.6, label='Empirical Volatility')

# 模型波动率（实线）
plt.plot(plot_mats, np.array(model_vols)*10000, color='blue', 
         linewidth=2, label='Model Volatility')


# 装饰
plt.title("Yield Volatility in Annual Basis Points,\n" 
"from Empirical Analysis and as Implied by the Estimated Gauss+ Model",fontsize=14, pad=20)
plt.xlabel("Maturity (Years)")
plt.ylabel("Volatility (Basis Points)")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.show()

# ==========================================
# 1. 核心模型函数 (基于您的 alpha_params)
# ==========================================
def calculate_residuals(sample_data, factors_df, alpha_params, sigma_vec, rho):
    """
    计算所有期限的残差，并确保日期对齐
    """
    maturities = np.arange(1, 31)
    sm, sl = sigma_vec
    as_, am, al = alpha_params
    
    # 1. 核心修复：确保 sample_data 和 factors_df 的日期完全一致
    # 使用 join 按照索引（日期）取交集
    aligned_data = sample_data.join(factors_df[['Short', 'Medium', 'Long']], how='inner')
    
    # 重新提取对齐后的数据
    aligned_factors = aligned_data[['Short', 'Medium', 'Long']].values
    
    # 2. 预计算模型组件
    c_terms = np.array([get_C_term(tau, alpha_params, [sm, sl], rho) for tau in maturities])
    ups_all = np.array([get_Upsilon_all(tau, alpha_params) for tau in maturities])
    
    # 创建结果 DataFrame，使用对齐后的索引
    residuals = pd.DataFrame(index=aligned_data.index)
    
    for i, tau in enumerate(maturities):
        actual_col = f'SVENY{tau:02d}'
        if actual_col not in aligned_data.columns: 
            continue
        
        # 实际收益率 (来自对齐后的数据)
        actual_yield = aligned_data[actual_col]
        
        # 模型收益率: y(tau) = Upsilon @ Factors - C(tau)
        # 此时 aligned_factors 和 actual_yield 长度必然相等 (均为 1993)
        model_yield = aligned_factors @ ups_all[i] - c_terms[i]
        
        # 计算残差 (bps)
        residuals[tau] = (actual_yield - model_yield) * 10000
        
    return residuals

# ==========================================
# 2. 绘制残差热力图
# ==========================================
def plot_residual_heatmap(residuals_df):
    plt.figure(figsize=(14, 8))
    
    # 为了清晰，我们可以对日期进行重采样（如月度平均），防止热力图过于拥挤
    res_monthly = residuals_df.resample('M').mean().T
    
    # 绘制热力图
    sns.heatmap(res_monthly, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Residual (bps)'})
    
    plt.title("Gauss+ Model Yield Residuals: Actual - Model (Basis Points)\n"
              "Note: Residuals at 2y and 10y are zero by definition", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Maturity (Years)")
    
    # 格式化 X 轴日期显示
    ax = plt.gca()
    labels = [item.get_text()[:7] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()

# 执行计算 (假设您已经有了之前步骤的参数和 DataFrame)
residuals = calculate_residuals(sample_data, plot_df, alpha_params, opt_sigma[:2], opt_sigma[2])
residuals.to_excel('GaussPlus_Yield_Residuals.xlsx', index=True)
plot_residual_heatmap(residuals)


# 1. 加载残差数据

# 2. 设定您估计出的 Alpha 参数 (用于计算精确对冲比例)

# --- 核心计算：获取 5y 相对于 2y 和 10y 的 Model Betas ---
def get_weights(a_vec):
    # 获取 2y, 5y, 10y 的因子载荷 (Medium & Long 因子)
    u5 = get_Upsilon_all(5, a_vec)[1:]
    u2 = get_Upsilon_all(2, a_vec)[1:]
    u10 = get_Upsilon_all(10, a_vec)[1:]
    
    # 解方程: u5 = beta2*u2 + beta10*u10
    # 这保证了当 2y 和 10y 变动时，组合的对冲效果是模型中性的
    weights = u5 @ np.linalg.inv(np.array([u2, u10]))
    return weights

# ==========================================
# 1. 策略参数与实战成本设定
# ==========================================
notional_5y = 100_000_000  # 1亿美元 5y 做空 (Belly)
cost_bps = 0.2             # 单边交易成本 (买卖价差 + 佣金) 0.2 bp
reb_threshold = 0.05       # 调仓带：权重偏离超过 5% 才会触发调仓交易

# 假设 alpha_params 已经定义，获取静态 Yield Beta
b2y, b10y = get_weights(alpha_params)

# ==========================================
# 2. 预计算动态 DV01 与 信号
# ==========================================
def calc_dv01(yield_series, tau):
    # 将百分比收益率转为小数 (例如 4.0 -> 0.04)
    y = yield_series / 100 if yield_series.max() > 1 else yield_series
    price = 100 / ((1 + y) ** tau)
    # DV01 = P * Duration_mod * 0.0001
    return (price * tau * 0.0001) / (1 + y)

# 计算每日市场真实 DV01 (每 100 元面值)
dv01_2y_ts = calc_dv01(sample_data['SVENY02'], 2)
dv01_5y_ts = calc_dv01(sample_data['SVENY05'], 5)
dv01_10y_ts = calc_dv01(sample_data['SVENY10'], 10)

# 信号生成 (基于残差组合)
res_fly = residuals[5] - (b2y * residuals[2] + b10y * residuals[10])
z_score = (res_fly - res_fly.rolling(60).mean()) / res_fly.rolling(60).std()
valid_z = z_score.dropna()

# ==========================================
# 3. 核心函数重构：增加信号追踪功能
# ==========================================
def run_pro_backtest(entry_thresh, exit_targ, max_hold):
    in_trade = False
    equity_curve = [0.0]
    total_fees = 0
    hold_n2, hold_n10 = 0.0, 0.0
    
    # 新增：详细的交易清单和信号日志
    trade_list = []  # 记录每一笔完整交易
    signals_log = [] # 记录每一个信号点点用于绘图
    
    entry_equity_pre_cost = 0.0
    entry_date = None
    entry_z = 0.0
    
    common_dates = valid_z.index
    
    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i-1]
        z = valid_z.loc[date]
        
        # 理论最优权重计算
        target_n2 = notional_5y * b2y * (dv01_5y_ts.loc[date] / dv01_2y_ts.loc[date])
        target_n10 = notional_5y * b10y * (dv01_5y_ts.loc[date] / dv01_10y_ts.loc[date])
        
        # 计算今日持仓损益
        d_spread = res_fly.loc[date] - res_fly.loc[prev_date]
        cash_dv01_5y = notional_5y * (dv01_5y_ts.loc[date] / 100)
        pnl_today = d_spread * cash_dv01_5y if in_trade else 0
        current_equity = equity_curve[-1] + pnl_today
        
        if not in_trade:
            if z < entry_thresh:
                in_trade = True
                entry_idx = i
                entry_date = date
                entry_z = z
                entry_equity_pre_cost = equity_curve[-1] # 记录建仓前的总净值
                
                # --- 建立头寸并支付成本 ---
                hold_n2, hold_n10 = target_n2, target_n10
                total_volume = notional_5y + hold_n2 + hold_n10
                fee = total_volume * (cost_bps / 10000)
                total_fees += fee
                current_equity -= fee
                
                signals_log.append({'Date': date, 'Type': 'Entry', 'Equity': current_equity, 'Z': z})
        else:
            days_held = i - entry_idx
            # 检查出场条件
            is_target_exit = z > exit_targ
            is_time_exit = days_held >= max_hold
            
            if is_target_exit or is_time_exit:
                in_trade = False
                # --- 平仓并支付成本 ---
                total_volume = notional_5y + hold_n2 + hold_n10
                fee = total_volume * (cost_bps / 10000)
                total_fees += fee
                current_equity -= fee
                
                # 计算这笔交易的纯利润 (Exit Equity - Pre-Entry Equity)
                trade_net_pnl = current_equity - entry_equity_pre_cost
                
                # 存入交易清单
                trade_list.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': date,
                    'Holding_Days': days_held,
                    'PnL_USD': trade_net_pnl,
                    'Entry_Z': entry_z,
                    'Exit_Z': z,
                    'Exit_Reason': 'Target' if is_target_exit else 'TimeLimit'
                })
                
                signals_log.append({'Date': date, 'Type': 'Exit', 'Equity': current_equity, 'Z': z})
                hold_n2, hold_n10 = 0.0, 0.0
            else:
                # 调仓检查
                diff_2 = abs(hold_n2 - target_n2) / target_n2
                diff_10 = abs(hold_n10 - target_n10) / target_n10
                if diff_2 > reb_threshold or diff_10 > reb_threshold:
                    reb_vol = abs(hold_n2 - target_n2) + abs(hold_n10 - target_n10)
                    fee = reb_vol * (cost_bps / 10000)
                    total_fees += fee
                    current_equity -= fee
                    hold_n2, hold_n10 = target_n2, target_n10
                    
        equity_curve.append(current_equity)
        
    return equity_curve, total_fees, pd.DataFrame(trade_list), pd.DataFrame(signals_log)

# ==========================================
# 4. 参数寻优循环 (Grid Search) - 增加胜率统计
# ==========================================
entry_range = [-1.0, -1.2, -1.5, -1.8, -2.0]
exit_range  = [0.0, 0.3, 0.5, 0.8, 1.0]
hold_range  = [20, 30, 45, 60, 90]

# ==========================================
# 4. 参数寻优循环 (修改后的修复版本)
# ==========================================
search_results = []

for et, xt, mh in itertools.product(entry_range, exit_range, hold_range):
    # run_pro_backtest 返回的第 3 个值是 trades_df (DataFrame)
    equity, fees, trades_df, _ = run_pro_backtest(et, xt, mh)
    
    if len(trades_df) >= 5:
        # --- 修复位置：统计 PnL_USD 列中大于 0 的个数 ---
        # trades_df['PnL_USD'] > 0 会返回一个布尔序列，.sum() 会统计 True 的个数
        win_count = (trades_df['PnL_USD'] > 0).sum()
        win_rate = win_count / len(trades_df)
        
        search_results.append({
            'entry_thresh': et, 
            'exit_targ': xt, 
            'max_hold': mh,
            'total_net_pnl': equity[-1], 
            'num_trades': len(trades_df),
            'win_rate': win_rate
        })
# 转换为 DataFrame 并排序
optimization_df = pd.DataFrame(search_results)
best_row = optimization_df.sort_values('total_net_pnl', ascending=False).iloc[0]

# 保存完整的网格搜索结果
optimization_df.to_excel('pro_backtest_grid_search_results.xlsx', index=False)

best_entry = best_row['entry_thresh']
best_exit = best_row['exit_targ']
best_hold = int(best_row['max_hold'])

print(f"\n寻优完成！最佳参数组合：")
print(f"Entry: {best_entry} | Exit: {best_exit} | Hold: {best_hold}")
print(f"总净利润: ${best_row['total_net_pnl']:,.2f} | 胜率: {best_row['win_rate']:.2%}")

# ==========================================
# 5. 使用最佳参数运行最终回测并绘图
# ==========================================
net_equity, total_cost, trades_final, df_signals = run_pro_backtest(best_entry, best_exit, best_hold)

# --- 修复位置：使用 Pandas 的方式计算胜率 ---
if not trades_final.empty:
    final_win_rate = (trades_final['PnL_USD'] > 0).mean()
else:
    final_win_rate = 0

    
plt.figure(figsize=(14, 12))

# --- 子图 1: 权益曲线 + 信号标记 ---
plt.subplot(2, 1, 1)
plt.plot(valid_z.index, net_equity, label='Net Equity (After Costs)', color='darkgreen', linewidth=1.5, alpha=0.8)

if not df_signals.empty:
    entries = df_signals[df_signals['Type'] == 'Entry']
    exits = df_signals[df_signals['Type'] == 'Exit']
    plt.scatter(entries['Date'], entries['Equity'], marker='^', color='red', s=100, label='Short 5Y (Entry)', zorder=5)
    plt.scatter(exits['Date'], exits['Equity'], marker='v', color='lime', s=100, label='Cover (Exit)', zorder=5)

plt.title(f"Optimized Equity Curve (Net: ${net_equity[-1]:,.0f} | Win Rate: {final_win_rate:.1%})", fontsize=14)
plt.ylabel("Cumulative USD PnL")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 子图 2: Z-Score + 信号标记 ---
plt.subplot(2, 1, 2)
plt.plot(z_score.index, z_score, label='Rolling Z-Score', color='royalblue', alpha=0.5)
plt.fill_between(z_score.index, best_entry, z_score, where=(z_score < best_entry), color='red', alpha=0.15)

if not df_signals.empty:
    plt.scatter(entries['Date'], entries['Z'], marker='^', color='red', s=100, zorder=5)
    plt.scatter(exits['Date'], exits['Z'], marker='v', color='lime', s=100, zorder=5)

plt.axhline(best_entry, color='red', linestyle='--', label=f'Entry: {best_entry}')
plt.axhline(best_exit, color='green', linestyle=':', label=f'Exit: {best_exit}')
plt.title(f"Signal Analysis: {len(trades_final)} Trades Executed", fontsize=14)
plt.ylabel("Z-Score")
plt.legend()

plt.tight_layout()
plt.show()

# 保存流水
trades_final.to_csv('trade_execution_log.csv', index=False)