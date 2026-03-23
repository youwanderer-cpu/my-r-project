import streamlit as st
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io
import json
import os
import hashlib
import csv

st.set_page_config(page_title="租赁资产管理系统-安全审计增强版", layout="wide")

# --- A. 系统常量与安全配置 ---
DB_FILE = "leases_db.json"
USERS_FILE = "users.json"
LOG_FILE = "operation_logs.csv"

def hash_password(password, salt="Lease_Project_2026"):
    """加盐哈希计算"""
    return hashlib.sha256((password + salt).encode()).hexdigest()

# --- 主程序中的安全逻辑 ---
def load_users():
    """仅负责读取，不再包含任何明文密码创建逻辑"""
    if not os.path.exists(USERS_FILE):
        st.error("🚨 系统未初始化！找不到 users.json 文件，请联系管理员。")
        st.stop()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def log_operation(username, role, action, details):
    """记录操作审计日志至 CSV"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["时间", "用户名", "角色", "操作类型", "操作详情"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username, role, action, details
        ])

# --- C. 数据持久化逻辑 ---

def save_to_local(data):
    serializable = []
    for item in data:
        new_item = item.copy()
        new_item['start'] = item['start'].isoformat()
        new_item['end'] = item['end'].isoformat()
        if 'mod_history' in new_item:
            serialized_history = []
            for record in new_item['mod_history']:
                rec_copy = record.copy()
                for k, v in rec_copy.items():
                    if isinstance(v, (date, pd.Timestamp)):
                        rec_copy[k] = v.isoformat()
                serialized_history.append(rec_copy)
            new_item['mod_history'] = serialized_history
        serializable.append(new_item)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=4)

def load_from_local():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    item['start'] = date.fromisoformat(item['start'])
                    item['end'] = date.fromisoformat(item['end'])
                    if 'mod_history' in item:
                        for record in item['mod_history']:
                            for k in ['变更生效日期', '原到期日', '新到期日']:
                                if k in record and record[k]:
                                    record[k] = date.fromisoformat(record[k])
                    else:
                        item['mod_history'] = []
                return data
        except: return []
    return []

# --- 1. 安全登录系统 (居中优化版) ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        # 1. 标题居中显示 (使用 HTML)
        st.markdown("<h1 style='text-align: center;'>🔐 租赁资产管理系统 - 安全登录</h1>", unsafe_allow_html=True)
        
        # 2. 创建三列布局：[1, 2, 1] 比例
        # 左右两列为空白占位，中间列放置登录框，实现水平居中
        left_space, col_mid, right_space = st.columns([1, 2, 1])
        
        with col_mid:
            st.write("") # 增加顶部空隙
            # 使用容器包裹，视觉上更聚合
            with st.container(border=True):
                username = st.text_input("用户名")
                password = st.text_input("密码", type="password")
                
                st.info("💡 **权限说明**：本系统仅限财务部使用。")
                st.warning("📞 **获取密码**：请联系财务部资产管理组 (分机: XXXX-XXXX)")
                
                # 按钮使用 use_container_width 使其占满中间列宽度，更美观
                if st.button("🚀 进入系统", use_container_width=True):
                    users = load_users()
                    if username in users:
                        if hash_password(password) == users[username]["password_hash"]:
                            st.session_state["authenticated"] = True
                            st.session_state["user_role"] = users[username]["role"]
                            st.session_state["user_name"] = users[username]["name"]
                            st.session_state["username_id"] = username
                            log_operation(username, users[username]["role"], "登录", "用户成功进入系统")
                            st.rerun()
                    st.error("❌ 用户名或密码错误")
        return False
    return True

if check_password():
    role = st.session_state["user_role"]
    uname = st.session_state["username_id"]
    
    if 'lease_db' not in st.session_state:
        st.session_state['lease_db'] = load_from_local()

    def get_total_months(start_date, end_date):
        m_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        return m_diff + 1 if end_date.day >= start_date.day else (m_diff if m_diff > 0 else 1)

    # --- 2. 核心计算引擎 (保持原逻辑) ---
    def calculate_lease_schedule(name, start_date, end_date, cycle_rent_net, rent_free_months, prepaid, annual_rate, tax_rate, vat_rate, payment_freq=1, is_prepaid=False, mod_history=None):
        """
        核心计算引擎 - V5.2 终极专业版
        支持：周期租金、期初/期末切换、现金流精准对齐、负债科目拆解
        
        参数:
        cycle_rent_net: 每一期实际支付的金额（不含税）
        payment_freq: 付款频率（月数），如季付=3
        is_prepaid: True 为期初支付 (Advance)，False 为期末支付 (Arrears)


        """
    # 1. 预处理：在进入循环前，将同月的多重变更合并，确保只保留最后一次结果
        effective_mods = {}
        if mod_history:
            for m in sorted(mod_history, key=lambda x: x['变更生效日期']):
                m_key = f"{m['变更生效日期'].year}-{m['变更生效日期'].month}"
                effective_mods[m_key] = m

        # 1. 基础时间参数计算
        total_months = get_total_months(start_date, end_date)
        date_range = [(start_date + relativedelta(months=i, day=31)) for i in range(total_months)]
        
        # 初始运行参数
        curr_cycle_rent = cycle_rent_net
        curr_rate = annual_rate
        curr_end = end_date
        periodic_rate = (1 + curr_rate)**(1/12) - 1
        
        # 2. 构造初始现金流序列 (修复定位逻辑)
        cash_flows_initial = [0] * total_months
        for i in range(total_months):
            if i >= int(rent_free_months):
                # 【关键修复】：判定当前月份是否为付款月
                # 期初支付：相对月份 0, 3, 6... 付款；期末支付：相对月份 2, 5, 8... 付款
                check_val = i - int(rent_free_months) if is_prepaid else (i + 1 - int(rent_free_months))
                if check_val % payment_freq == 0:
                    cash_flows_initial[i] = curr_cycle_rent
        
        # 【会计准则要求】：计算未折现的租赁付款额总额
        total_undiscounted_payments = sum(cash_flows_initial)
        
        # 3. 计算初始租赁负债 (现值/PV)
        if is_prepaid:
            # 期初支付模式：第一期租金不折现 (k=0)，幂次从 0 开始
            liability_initial = sum([cf / (1 + periodic_rate)**k for k, cf in enumerate(cash_flows_initial)])
        else:
            # 期末支付模式：第一期租金折现一次 (k=1)，幂次从 1 开始
            liability_initial = sum([cf / (1 + periodic_rate)**(k+1) for k, cf in enumerate(cash_flows_initial)])
            
        rou_asset_initial = liability_initial + prepaid
        
        # 动态跟踪余额
        liab_bal = liability_initial
        rou_nbv = rou_asset_initial
        
        schedule = []
        # 4. 月度摊销循环
        for i in range(total_months):
            curr_date = date_range[i]
            m_key = f"{curr_date.year}-{curr_date.month}"
        
        # 修改：从合并后的字典中获取当前月份的最终变更
            active_mod = effective_mods.get(m_key)
            if active_mod:
                rem_months = get_total_months(curr_date, curr_end)
                # 优先读取新周期租金
                new_cycle_rent = active_mod.get('新周期租金', active_mod.get('新月租金', 0) * payment_freq)
                new_rate_val = float(active_mod['修订后折现率'].replace('%','')) / 100
                new_p_rate = (1 + new_rate_val)**(1/12) - 1
                
                # 构造重估后的剩余现金流 (同步修复定位)
                rem_cash_flows = [0] * rem_months
                for k in range(rem_months):
                    abs_month_idx = i + k 
                    if abs_month_idx >= int(rent_free_months):
                        check_val = abs_month_idx - int(rent_free_months) if is_prepaid else (abs_month_idx + 1 - int(rent_free_months))
                        if check_val % payment_freq == 0:
                            rem_cash_flows[k] = new_cycle_rent
                
                # 现值重估 (区分时点)
                if is_prepaid:
                    new_liab = sum([cf / (1 + new_p_rate)**k for k, cf in enumerate(rem_cash_flows)])
                else:
                    new_liab = sum([cf / (1 + new_p_rate)**(k+1) for k, cf in enumerate(rem_cash_flows)])
                
                rou_nbv += (new_liab - liab_bal)
                liab_bal = new_liab
                curr_cycle_rent, curr_rate, periodic_rate = new_cycle_rent, new_rate_val, new_p_rate

            # --- 月度财务计算 ---
            
            # A. 确定当月实际支付金额 (修复明细表定位)
            payment = 0
            if i >= int(rent_free_months):
                check_val = i - int(rent_free_months) if is_prepaid else (i + 1 - int(rent_free_months))
                if check_val % payment_freq == 0:
                    payment = curr_cycle_rent

            # B. 计算利息费用：根据支付时点确定计息基数
            if is_prepaid:
                # 【期初支付模式】：月初付款，基数 = 期初余额 - 当月支付
                interest = max(0, (liab_bal - payment)) * periodic_rate
                liab_bal_next = (liab_bal - payment) + interest
            else:
                # 【期末支付模式】：月初余额即基数，月末付款
                interest = liab_bal * periodic_rate
                liab_bal_next = liab_bal + interest - payment
            
            # C. 折旧计算
            rem_periods = total_months - i
            monthly_depr = rou_nbv / rem_periods if rem_periods > 0 else 0
            
            # D. 更新期末余额
            liab_bal = liab_bal_next
            rou_nbv -= monthly_depr
            
            # E. 税务及进项
            vat_amount = payment * vat_rate
            temp_diff = rou_nbv - liab_bal
            
            schedule.append({
                "日期": curr_date.strftime("%Y-%m-%d"),
                "资产名称": name,
                "租赁负债_期末": max(0, liab_bal),
                "使用权资产_期末": max(0, rou_nbv),
                "利息费用": interest,
                "折旧费用": monthly_depr,
                "现金支出(不含税)": payment,
                "当月增值税进项": vat_amount,
                "现金总支出(含税)": payment + vat_amount,
                "递延所得税资产(DTA)": abs(temp_diff * tax_rate) if temp_diff < 0 else 0,
                "递延所得税负债(DTL)": abs(temp_diff * tax_rate) if temp_diff > 0 else 0
            })
            
        return pd.DataFrame(schedule), liability_initial, rou_asset_initial, total_undiscounted_payments
# --- 3. 侧边栏 (支持周期支付金额输入) ---
# --- 3. 侧边栏 (支持周期支付与时点切换增强版) ---
    with st.sidebar:
        st.header(f"👤 {st.session_state['user_name']}")
        st.caption(f"权限角色: {role.upper()}")
        if st.button("安全退出"):
            log_operation(uname, role, "登出", "用户主动退出系统")
            st.session_state["authenticated"] = False
            st.rerun()
        st.divider()
        
        if role == "manager":
            st.header("➕ 新增租赁资产")
            with st.form("lease_form", clear_on_submit=False):
                a_name = st.text_input("资产名称", placeholder="例如：浦东办公楼A座")
                
                # A. 付款周期与金额
                col_f1, col_f2 = st.columns(2)
                freq_map = {"月付": 1, "季付": 3, "半年付": 6, "年付": 12}
                a_freq_label = col_f1.selectbox("付款周期", list(freq_map.keys()))
                a_freq = freq_map[a_freq_label]
                a_cycle_rent = col_f2.number_input(f"{a_freq_label}金额 (不含税)", value=30000.0, step=1000.0)
                
                # B. 新增：付款时点选择 (期初 vs 期末)
                # 影响折现幂次与利息计提基数
                a_timing = st.selectbox(
                    "付款时点 (影响折现与计息)", 
                    ["期末支付 (后付)", "期初支付 (先付)"],
                    index=0  # 默认为期末支付
                )
                is_prepaid = (a_timing == "期初支付 (先付)")
                
                # 自动计算月均参考
                #st.caption(f"💡 自动折算：相当于月租金 ¥{a_cycle_rent/a_freq:,.2f} (不含税)")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                a_start = col1.date_input("起始日期", value=date.today())
                a_end = col2.date_input("终止日期", value=date.today() + relativedelta(years=2) - relativedelta(days=1))
                
                a_vat = st.number_input("增值税率 (%)", value=9.0) / 100
                a_free = st.number_input("免租期 (月)", value=0)
                a_prepaid = st.number_input("预付租金 (不含税)", value=0.0)
                a_rate = st.number_input("折现率 (%)", value=4.35, help="通常使用增量借款利率") / 100
                a_tax = st.number_input("企业所得税率 (%)", value=25.0) / 100
                
                if st.form_submit_button("🚀 确认添加资产"):
                    if a_name and a_end > a_start:
                        # 1. 准备数据对象 (新增 is_prepaid 标志)
                        new_asset = {
                            "name": a_name, 
                            "start": a_start, 
                            "end": a_end, 
                            "cycle_rent": a_cycle_rent,      
                            "rent": a_cycle_rent / a_freq,   
                            "payment_freq": a_freq,          
                            "is_prepaid": is_prepaid,        # <--- 新增：支付时点持久化
                            "free": a_free, 
                            "prepaid": a_prepaid, 
                            "rate": a_rate, 
                            "tax": a_tax, 
                            "vat": a_vat, 
                            "mod_history": []
                        }
                        
                        # 2. 保存到 Session 数据库
                        st.session_state['lease_db'].append(new_asset)
                        save_to_local(st.session_state['lease_db'])
                        
                        # 3. 构造极其详细的审计日志 (增加支付时点详情)
                        log_details = (
                            f"资产名称: {a_name} | "
                            f"付款安排: {a_freq_label} ({a_timing}) | "
                            f"周期金额: ¥{a_cycle_rent:,.2f} | "
                            f"月均等值: ¥{a_cycle_rent/a_freq:,.2f} | "
                            f"折现率: {a_rate*100:.2f}% | "
                            f"期间: {a_start} 至 {a_end}"
                        )
                        log_operation(uname, role, "新增资产", log_details)
                        
                        st.success(f"资产 {a_name} 已录入。系统已按{a_timing}逻辑生成摊销表。")
                        st.rerun()
        else:
            st.info("ℹ️ 只读权限：您的账号无法新增租赁资产。")


    # --- 4. 主界面渲染 (修正调用逻辑) ---
    st.title("🏢 租赁资产全生命周期财务管理中心")
    if not st.session_state['lease_db']:
        st.info("💡 暂无数据。")
    else:
        # 【关键修正】：传入 cycle_rent 和 payment_freq
        # --- 4. 主界面渲染 (修正后的调用逻辑) ---
        full_results = []
        for i in st.session_state['lease_db']:
            res = calculate_lease_schedule(
                i['name'], i['start'], i['end'], 
                i.get('cycle_rent', i['rent']), 
                i['free'], i['prepaid'], i['rate'], i['tax'], i['vat'], 
                payment_freq=i.get('payment_freq', 1), 
                is_prepaid=i.get('is_prepaid', False),  # <--- 必须补上这一行
                mod_history=i.get('mod_history', [])
            )
            full_results.append(res)
            
        master_df = pd.concat([r[0] for r in full_results])
        master_df['日期'] = pd.to_datetime(master_df['日期'])

        # --- A. 初始入账价值预览 (专业会计拆解版) ---
        st.subheader("📑 初始入账价值预览")
        
        # 建立频率标签映射
        inv_freq_map = {1: "月付", 3: "季付", 6: "半年付", 12: "年付"}
        
        init_summary = []
        # 注意：这里解包 calculate_lease_schedule 返回的 4 个值
        for i, r in enumerate(full_results):
            info = st.session_state['lease_db'][i]
            # 解包：结果表, 负债现值, 资产原值, 租赁付款额总额
            df_res, liab_pv, rou_nbv, total_pay = r 
            
            # 计算未确认融资费用 (租赁付款额 - 租赁负债现值)
            unrecognized_finance_cost = total_pay - liab_pv
            
            # 获取付款周期标签
            p_freq = info.get('payment_freq', 1)
            freq_label = inv_freq_map.get(p_freq, "月付")
            
            # 确定当前周期金额
            c_rent = info.get('cycle_rent', info['rent'] * p_freq)

            init_summary.append({
                "资产名称": info['name'],
                "起始日期": info['start'],
                "终止日期": info['end'],
                "付款周期": freq_label,
                "每期租金(不含税)": c_rent,
                "月均等值": info['rent'],
                "租赁付款额(总额)": total_pay,
                "未确认融资费用": unrecognized_finance_cost,
                "租赁负债(现值)": liab_pv,
                "初始使用权资产": rou_nbv,
                "支付时点": "期初支付" if info.get('is_prepaid') else "期末支付"
            })

        init_df = pd.DataFrame(init_summary)
        
        # 使用 Styler 进行财务格式化（千分位、保留两位小数）
        st.dataframe(init_df.style.format({
            "每期租金(不含税)": "¥{:,.2f}",
            "月均等值": "¥{:,.2f}",
            "租赁付款额(总额)": "¥{:,.2f}",
            "未确认融资费用": "¥{:,.2f}",
            "租赁负债(现值)": "¥{:,.2f}",
            "初始使用权资产": "¥{:,.2f}"
        }), use_container_width=True)

        # --- B. 报表下载 (数据同步强化版) ---
        st.subheader("📥 导出财务明细报表")
        
        # 1. 准备月末快照汇总数据 (基于最新的 master_df)
        snapshot_summary = master_df.groupby('日期').agg({
            '使用权资产_期末': 'sum', '租赁负债_期末': 'sum',
            '利息费用': 'sum', '折旧费用': 'sum', '现金支出(不含税)': 'sum',
            '当月增值税进项': 'sum', '现金总支出(含税)': 'sum',
            '递延所得税资产(DTA)': 'sum', '递延所得税负债(DTL)': 'sum'
        }).reset_index()

        snapshot_summary['递延所得税净额'] = snapshot_summary['递延所得税资产(DTA)'] - snapshot_summary['递延所得税负债(DTL)']
        
        # 2. 准备租赁变更记录数据
        all_mod_records = []
        for asset in st.session_state['lease_db']:
            for rec in asset.get('mod_history', []):
                all_mod_records.append({"资产名称": asset['name'], **rec})
        mod_history_df = pd.DataFrame(all_mod_records)

        # 3. 准备初始合同清单数据 (核心修改：从计算结果 df_res 提取实付租金)
        initial_contracts_list = []
        inv_freq_map = {1: "月付", 3: "季付", 6: "半年付", 12: "年付"}

        for i, r in enumerate(full_results):
            info = st.session_state['lease_db'][i]
            # 解包计算引擎返回的 4 个值
            df_res, liab_pv, rou_nbv, total_pay = r 
            
            p_freq = info.get('payment_freq', 1)
            freq_label = inv_freq_map.get(p_freq, "月付")
            timing_label = "期初支付" if info.get('is_prepaid') else "期末支付"

            # 【关键修正点】：不再从 info 读取租金，而是从计算出的明细表 df_res 中提取第一笔真实支付额
            # 这能确保如果起始日发生了变更，Excel 清单中显示的是变更后的金额
            valid_payments = df_res[df_res['现金支出(不含税)'] > 0]['现金支出(不含税)']
            actual_cycle_rent = valid_payments.iloc[0] if not valid_payments.empty else info.get('cycle_rent', 0)

            unrecognized_finance_cost = total_pay - liab_pv

            initial_contracts_list.append({
                "资产名称": info['name'],
                "起始日期": info['start'],
                "终止日期": info['end'],
                "付款周期": freq_label,
                "支付时点": timing_label,
                "周期租金(最新生效)": actual_cycle_rent, # <--- 修正为提取计算后的实付款
                "月均等值": actual_cycle_rent / p_freq,   # <--- 同步更新月均参考
                "租赁付款额(总额)": total_pay,
                "未确认融资费用": unrecognized_finance_cost,
                "初始确认-租赁负债(现值)": liab_pv,
                "初始确认-使用权资产": rou_nbv,
                "增值税率": f"{info['vat']*100:.1f}%",
                "免租期(月)": info['free'],
                "预付租金(不含税)": info['prepaid'],
                "折现率": f"{info['rate']*100:.2f}%",
                "所得税率": f"{info['tax']*100:.2f}%"
            })
            
        initial_contracts_df = pd.DataFrame(initial_contracts_list)

        # 4. 执行 Excel 导出 (保持 xlsxwriter 逻辑不变)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
            master_df.to_excel(writer, index=False, sheet_name='全周期明细')
            snapshot_summary.to_excel(writer, index=False, sheet_name='月末财务快照汇总')
            initial_contracts_df.to_excel(writer, index=False, sheet_name='初始合同清单') 
            mod_history_df.to_excel(writer, index=False, sheet_name='租赁变更记录')
            
            workbook = writer.book
            num_fmt = workbook.add_format({'num_format': '#,##0.00', 'align': 'right'})
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:P', 18, num_fmt) # 稍微加宽一点列宽
        
        st.download_button(
            label="💾 下载 Excel 完整报表", 
            data=output.getvalue(), 
            file_name=f"Lease_Financial_Report_{date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        # --- C. 图表展示 ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 资产与负债趋势")
            st.line_chart(master_df.groupby('日期')[['租赁负债_期末', '使用权资产_期末']].sum())
        with c2:
            st.subheader("📉 递延所得税净额趋势")
            tax_trend = master_df.groupby('日期')[['递延所得税资产(DTA)', '递延所得税负债(DTL)']].sum()
            tax_trend['DTA_Net'] = (tax_trend['递延所得税资产(DTA)'] - tax_trend['递延所得税负债(DTL)']).clip(lower=0)
            tax_trend['DTL_Net'] = (tax_trend['递延所得税负债(DTL)'] - tax_trend['递延所得税资产(DTA)']).clip(lower=0)
            st.line_chart(tax_trend[['DTA_Net', 'DTL_Net']])

        # --- D. 月度财务快照 (增加增值税展示) ---
        st.divider()
        st.subheader("🔍 月度存量财务快照")
        target_month = st.date_input("选择月末时点", value=date.today())
        target_str = target_month.strftime("%Y-%m")
        view = master_df[master_df['日期'].dt.strftime("%Y-%m") == target_str]
        
        if not view.empty:
            sum_rou = view['使用权资产_期末'].sum()
            sum_liab = view['租赁负债_期末'].sum()
            net_tax = view['递延所得税资产(DTA)'].sum() - view['递延所得税负债(DTL)'].sum()
            
            st.markdown("##### 🏛️ 资产负债表项目 (余额/净额列示)")
            r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
            r1_c1.metric("总使用权资产", f"¥{sum_rou:,.2f}")
            r1_c2.metric("总租赁负债", f"¥{sum_liab:,.2f}")
            r1_c3.metric("递延所得税资产 (净额)", f"¥{max(0, net_tax):,.2f}")
            r1_c4.metric("递延所得税负债 (净额)", f"¥{abs(min(0, net_tax)):,.2f}")

            st.markdown("##### 🧾 利润表与现金流项目 (当月发生)")
            r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
            r2_c1.metric("当月利息总额", f"¥{view['利息费用'].sum():,.2f}")
            r2_c2.metric("当月折旧总额", f"¥{view['折旧费用'].sum():,.2f}")
            r2_c3.metric("当月增值税进项", f"¥{view['当月增值税进项'].sum():,.2f}")
            r2_c4.metric("含税现金支出", f"¥{view['现金总支出(含税)'].sum():,.2f}")
            
            st.dataframe(view.style.format({
                "日期": lambda t: t.strftime("%Y-%m-%d"),
                "租赁负债_期末": "{:,.2f}", "使用权资产_期末": "{:,.2f}", "利息费用": "{:,.2f}", 
                "折旧费用": "{:,.2f}", "现金支出(不含税)": "{:,.2f}", "当月增值税进项": "{:,.2f}",
                "现金总支出(含税)": "{:,.2f}", "递延所得税资产(DTA)": "{:,.2f}", "递延所得税负债(DTL)": "{:,.2f}"
            }), use_container_width=True)
        else:
            st.warning("所选时点无活跃合同记录。")

        # --- E. 资产清单管理 (同步周期付款与全审计版) ---
        st.divider()
        with st.expander("📋 资产清单管理及变更预览", expanded=True):
            # 获取当前用户信息用于日志
            current_user = st.session_state.get("username_id", "未知用户")
            current_role = st.session_state.get("user_role", "viewer")

            # 建立频率标签映射
            inv_freq_map = {1: "月付", 3: "季付", 6: "半年付", 12: "年付"}

            for i, item in enumerate(st.session_state['lease_db']):
                m_count = get_total_months(item['start'], item['end'])
                asset_df = full_results[i][0].copy()
                asset_df['日期'] = pd.to_datetime(asset_df['日期'])
                
                # 获取当前付款频率和周期金额
                p_freq = item.get('payment_freq', 1)
                freq_label = inv_freq_map.get(p_freq, "月付")
                current_cycle_rent = item.get('cycle_rent', item['rent'] * p_freq)

                c_info, c_action = st.columns([6, 2])

                with c_info:
                    st.markdown(f"#### **{item['name']}**")
                    
                    # 获取计算引擎返回的初始价值 (liab_pv, rou_asset)
                    init_liab = full_results[i][1]
                    init_asset = full_results[i][2]
                    
                    # --- 新增：付款时点标签 ---
                    timing_label = "期初支付 (先付)" if item.get('is_prepaid') else "期末支付 (后付)"
                    
                    # 1. 基础信息展示
                    st.write(f"📅 **期间**: {item['start']} 至 {item['end']} (共 {m_count} 个月)")
                    st.write(f"💎 **入账价值**: 负债 ¥{init_liab:,.2f} / 资产 ¥{init_asset:,.2f}")
                    
                    # 修改：在付款安排中明确标注时点
                    st.write(f"💳 **付款安排**: {freq_label} ({timing_label}) | ¥{current_cycle_rent:,.2f} / 期")
                    
                    # 补充：在 caption 中也可以增加标注，确保关键参数一目了然
                    st.caption(f"💰 折算月均: ¥{item['rent']:,.2f} | ⚖️ 折现率: {item['rate']*100:.2f}% | 🎁 免租: {item['free']}月 | 💳 预付: ¥{item['prepaid']:,.2f} | 🧾 增值税率: {item['vat']*100:.1f}% | 🧾 所得税率: {item['tax']*100:.2f}%")
                             
                    # 2. 变更预览 (代码保持兼容)
                    if item.get('mod_history'):
                        latest_mod = item['mod_history'][-1]
                        mod_month_str = latest_mod['变更生效日期'].strftime("%Y-%m")
                        mod_view = asset_df[asset_df['日期'].dt.strftime("%Y-%m") == mod_month_str]
                        
                        if not mod_view.empty:
                            display_new_end = latest_mod.get('新到期日', item['end']).strftime("%Y-%m-%d")
                            # 优先显示周期金额
                            disp_mod_rent = latest_mod.get('新周期租金', latest_mod['新月租金'] * p_freq)

                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 12px; border-left: 5px solid #ff4b4b; margin-top: 8px; border-radius: 5px; color: #31333f;">
                                <b>🔄 最近变更预览 ({latest_mod['变更生效日期']})</b> | 📅 新到期日: {display_new_end}<br>
                                变更后{freq_label}: <b>¥{disp_mod_rent:,.2f}</b> | 修订折现率: <b>{latest_mod['修订后折现率']}</b><br>
                                ⚖️ <b>变更月末负债: ¥{mod_view.iloc[0]['租赁负债_期末']:,.2f}</b> | 🏗️ <b>资产: ¥{mod_view.iloc[0]['使用权资产_期末']:,.2f}</b>
                            </div>
                            """, unsafe_allow_html=True)

                with c_action:
                    if role == "manager":
                        c1, c2 = st.columns(2)
                        if c1.button("🔧 变更", key=f"mod_{i}"):
                            st.session_state[f"show_mod_{i}"] = not st.session_state.get(f"show_mod_{i}", False)
                            log_operation(current_user, current_role, "进入变更页面", f"准备调整资产 [{item['name']}]")
                            
                        if c2.button("❌ 删除", key=f"del_{i}"):
                            del_name = item['name']
                            st.session_state['lease_db'].pop(i)
                            save_to_local(st.session_state['lease_db'])
                            log_operation(current_user, current_role, "删除资产", f"删除了资产: {del_name}")
                            st.rerun()
                    else:
                        st.write("🔒 编辑受限")

                # --- 租赁变更动态输入模块 (同步周期付款逻辑) ---
                if role == "manager" and st.session_state.get(f"show_mod_{i}", False):
                    with st.container(border=True):
                        st.markdown(f"🖋️ **{item['name']} - 发起租赁变更 (重估每期支付)**")
                        mod_type = st.selectbox("变更类型", ["扩大租赁范围", "缩小租赁范围", "延长或缩短租赁期限", "租赁对价的变更（租金调整）", "提前终止租赁"], key=f"mod_type_{i}")
                        
                        m_col1, m_col2 = st.columns(2)
                        eff_date = m_col1.date_input("变更生效日期", value=date.today(), key=f"eff_date_{i}")
                        new_rate = m_col2.number_input("修订后折现率 (%)", value=item['rate']*100, key=f"new_rate_{i}") / 100
                        
                        # 初始化变更参数
                        m_new_cycle_rent = current_cycle_rent
                        m_new_end = item['end']

                        # 根据变更类型显示不同的输入框
                        if mod_type in ["扩大租赁范围", "缩小租赁范围", "租赁对价的变更（租金调整）"]:
                            m_new_cycle_rent = st.number_input(f"新{freq_label}金额 (不含税)", value=current_cycle_rent, key=f"m_rent_cycle_{i}")
                            st.caption(f"💡 自动折算：相当于新月租金 ¥{m_new_cycle_rent/p_freq:,.2f}")
                        
                        if mod_type in ["延长或缩短租赁期限"]:
                            m_new_end = st.date_input("新终止日期", value=item['end'], key=f"m_end_new_{i}")
                        
                        if mod_type == "提前终止租赁":
                            m_new_end = eff_date

                        if st.button("💾 保存变更并重算", key=f"save_mod_{i}"):
                            timing_str = "期初支付" if item.get('is_prepaid') else "期末支付"

                            # 1. 详细审计日志
                            log_parts = [
                                f"资产: {item['name']}",
                                f"支付模式: {timing_str}",
                                f"类型: {mod_type}",
                                f"生效日: {eff_date}",
                                f"{freq_label}: ¥{current_cycle_rent:,.2f} -> ¥{m_new_cycle_rent:,.2f}",
                                f"到期日: {item['end']} -> {m_new_end}",
                                f"折现率: {item['rate']*100:.2f}% -> {new_rate*100:.2f}%"
                            ]
                            log_operation(current_user, current_role, "提交租赁变更", " | ".join(log_parts))

                            # 2. 构造历史记录
                            record = {
                                "变更类型": mod_type,
                                "变更生效日期": eff_date,
                                "修订后折现率": f"{new_rate*100:.2f}%",
                                "原周期租金": current_cycle_rent,
                                "新周期租金": m_new_cycle_rent,
                                "新月租金": m_new_cycle_rent / p_freq, # 保持与计算引擎兼容
                                "原到期日": item['end'],
                                "新到期日": m_new_end
                            }
                            
                            item['mod_history'].append(record)
                            item['cycle_rent'] = m_new_cycle_rent # 更新当前周期金额
                            item['rent'] = m_new_cycle_rent / p_freq # 更新当前月均租金
                            item['rate'] = new_rate
                            
                            if mod_type in ["提前终止租赁", "延长或缩短租赁期限"]:
                                item['end'] = m_new_end
                            
                            save_to_local(st.session_state['lease_db'])
                            st.session_state[f"show_mod_{i}"] = False
                            st.success(f"变更已记录，系统将按 {freq_label} ¥{m_new_cycle_rent:,.2f} 重估。")
                            st.rerun()

        # --- F. 经理专属：审计日志查看器 ---
        if role == "manager":
            st.divider()
            with st.expander("🛡️ 系统审计日志 (仅经理可见)"):
                if os.path.exists(LOG_FILE):
                    logs_df = pd.read_csv(LOG_FILE)
                    st.dataframe(logs_df.sort_values(by="时间", ascending=False), use_container_width=True)
                    with open(LOG_FILE, "rb") as f:
                        st.download_button("📥 下载审计日志 (CSV)", data=f, file_name=f"Audit_Logs_{date.today()}.csv", mime="text/csv")
                else: st.info("暂无操作日志记录。")

            if st.button("🔥 危险：清空所有资产数据", type="primary"):
                st.session_state['lease_db'] = []; save_to_local([])
                log_operation(uname, role, "清空数据库", "执行了全量数据清空操作")
                st.rerun()