# =============================================================================
# 计算 w 相关逻辑
# =============================================================================
# %% 导入库
import numpy as np

# %% 莱斯利雌雄总矩
# 初始雌性种群数量分布向量
x0 = np.ones(12) * 10

# 雌雄总出生率向量 a
a = np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

# 各年龄段存活率向量 b
b = np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98])

# 创建空矩阵：莱斯利雌雄总矩阵 L_pp
n = len(x0)
L_pp = np.zeros((n, n))

# 填充空矩阵：生育率
L_pp[0, :] = a

# 填充空矩阵：存活率
for i in range(1, n):
    L_pp[i, i-1] = b[i-1]
    
# %% 莱斯利雌性矩 或 莱斯利雄性矩
# 莱斯利雌性/雄性矩阵 L_p
L_p = np.vstack([L_pp[0, :] * 0.5, L_pp[1:, :]]) # 雌性/雄性出生率为总出生率 a 的一半

# %% 考虑出售r比例的小母牛后的，莱斯利雌性矩
# 出售刚出生 r 比例的小母牛，即出售第一个年龄段小母牛的比例
r = 0.5 # 暂设为0.5

# 考虑出售刚出生 r 比例的小母牛后的莱斯利雌性矩阵
L_r = np.vstack([L_p[0, :] * (1 - r), L_p[1:, :]])

# %% 选择向量
y_1 = np.zeros(12)
y_1[0] = 1
y_12 = np.zeros(12)
y_12[-1] = 1
y_3_12 = np.zeros(12)
y_3_12[2:] = 1

# %% 主循环求关键年龄组的个数
# 模拟的年数
years = 5

population = x0.copy()      # 复制初始雌性种群
populations = []            # 记录每年的种群数量
populations.append(population.copy()) # 0 年的先写入populations中

num_xiaogongniu_sales = []  # 每年出售小公牛的数量列表
num_xiaogongniu_sales.append(0)

num_xiaomuniu_sales = []    # 每年出售小母牛的数量列表
num_xiaomuniu_sales.append(0)

num_damuniu_sales = []      # 计算每年可产牛奶的大母牛数量
num_damuniu_sales.append(0)

num_laomuniu_sales = []     # 每年出售老母牛的数量列表
num_laomuniu_sales.append(0)

for year in range(years):
    num_xiaogongniu_sales.append(L_p @ population @ y_1)
    num_xiaomuniu_sales.append(L_p @ population @ y_1 * r)
    num_damuniu_sales.append(L_r @ population @ y_3_12)
    num_laomuniu_sales.append(L_r @ population @ y_12)
    
    # %% 牧草需求
    alpha = (2/3) * (population[0] + population[1]) + 1 * population[2:12].sum()
    
    # %% 粮食需求
    beta1 = 20 
    beta2 = 30
    beta3 = 30
    beta4 = 10
    beta = beta1 + beta2 + beta3 + beta4
    
    q_beta = 1.1 * beta1 + 0.9 * beta2 + 0.8 * beta3 + 0.6 * beta4
    
    l_beta = q_beta - 0.6 * population[1:12].sum()
    
    if l_beta > 0:
        w_beta = l_beta * 75
    else:
        w_beta = l_beta * 90

    # %% 甜菜需求 
    gamma = 10
    
    q_gamma = 1.5 * gamma
    
    l_gamma = q_gamma - 0.7 * population[1:12].sum()
    
    if l_gamma > 0:
        w_gamma = l_gamma * 58
    else:
        w_gamma = l_gamma * 70

    # %% 土地限制
    total_land = alpha + beta + gamma
    total_land_limit = 200
    
    # %% 迭代逻辑
    population = L_r @ population
    populations.append(population)

populations = np.array(populations)
num_xiaogongniu_sales = np.array(num_xiaogongniu_sales)
num_xiaomuniu_sales = np.array(num_xiaomuniu_sales)
num_damuniu_sales = np.array(num_damuniu_sales)
num_laomuniu_sales = np.array(num_laomuniu_sales)

# %% 求关键年龄组的年毛利
w_xiaogongniu = 30 * num_xiaogongniu_sales
w_xiaomuniu = 40 * num_laomuniu_sales
w_damuniu = 370 * num_damuniu_sales
w_laomuniu = 120 * num_laomuniu_sales

w_nian = w_xiaogongniu + w_xiaomuniu + w_damuniu + w_laomuniu + w_beta + w_gamma