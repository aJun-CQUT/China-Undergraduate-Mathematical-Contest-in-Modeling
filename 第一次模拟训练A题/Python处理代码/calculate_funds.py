# =============================================================================
# 资金：年工人管理资金成本相关逻辑
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
y_1_2 = np.zeros(12)
y_1_2[0:2] = 1
y_3_12 = np.zeros(12)
y_3_12[2:12] = 1

# %% 主循环求关键年龄组的个数
# 模拟的年数
years = 5

population = x0.copy()                # 复制初始雌性种群
populations = []                      # 记录每年的种群数量
populations.append(population.copy()) # 0 年的先写入populations中
c_xiaomunius = []
c_xiaomunius.append(0)
c_damunius = []
c_damunius.append(0)
c_betas = []
c_betas.append(0)
c_gammas = []
c_gammas.append(0)

for year in range(years):
    c_xiaomunius.append(500 * population @ y_1_2)
    c_damunius.append(100 * population @ y_3_12)
    
    # %% 粮食需求
    beta1 = 20 
    beta2 = 30
    beta3 = 30
    beta4 = 10
    beta = beta1 + beta2 + beta3 + beta4
    
    c_betas.append(15 * beta)
    
    # %% 甜菜需求 
    gamma = 10
    c_gammas.append(10 * gamma)
    
    # %% 迭代逻辑
    population = L_r @ population
    populations.append(population)
    
# %% 格式化变量
# 转换列表为数组
populations = np.array(populations)
c_xiaomunius = np.array(c_xiaomunius)
c_damunius = np.array(c_damunius)
c_betas = np.array(c_betas)
c_gammas = np.array(c_gammas)
