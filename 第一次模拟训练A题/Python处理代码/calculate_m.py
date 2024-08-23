# =============================================================================
# 计算 M 相关逻辑
# =============================================================================
# %% 定义年还款函数
def calculate_annual_payment(M, annual_rate, years):
    # 计算年利率
    r = annual_rate
    
    # 计算还款期数
    n = years
    
    # 计算每年还款额
    annual_payment = (M * r) / (1 - (1 + r) ** -n)
    
    return annual_payment

# %% 主程序

# 贷款金额
M = 100000

# 年利率
annual_rate = 0.15

# 贷款年限
years = 10

# 计算每年还款额
annual_payment = calculate_annual_payment(M, annual_rate, years)

# 打印每年还款额，保留两位小数
print(f"每年还款额为: {annual_payment:.2f}")