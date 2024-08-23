# =============================================================================
# 时间：年工人成本 c_gongren 相关逻辑
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
t_xiaomunius = []
t_xiaomunius.append(0)
t_damunius = []
t_damunius.append(0)
t_betas = []
t_betas.append(0)
t_gammas = []
t_gammas.append(0)

for year in range(years):
    t_xiaomunius.append(population @ y_1_2)
    t_damunius.append(population @ y_3_12)
    
    # %% 粮食需求
    beta1 = 20 
    beta2 = 30
    beta3 = 30
    beta4 = 10
    beta = beta1 + beta2 + beta3 + beta4
    
    t_betas.append(4 * beta)
    
    # %% 甜菜需求 
    gamma = 10
    t_gammas.append(gamma)
    
    # %% 迭代逻辑
    population = L_r @ population
    populations.append(population)
    
# %% 格式化变量
# 转换列表为数组
populations = np.array(populations)
t_xiaomunius = np.array(t_xiaomunius)
t_damunius = np.array(t_damunius)
t_betas = np.array(t_betas)
t_gammas = np.array(t_gammas)