# =============================================================================
# 土地分配：种植 α，β，γ 相关逻辑
# =============================================================================
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
L_p = np.vstack([L_pp[0, :] * 0.5, L_pp[1:, :]])  # 雌性/雄性出生率为总出生率 a 的一半

# %% 考虑出售r比例的小母牛后的，莱斯利雌性矩
# 出售刚出生 r 比例的小母牛，即出售第一个年龄段小母牛的比例
r = 0.5  # 暂设为0.5

# 考虑出售刚出生 r 比例的小母牛后的莱斯利雌性矩阵
L_r = np.vstack([L_p[0, :] * (1 - r), L_p[1:, :]])

# %% 主循环求关键年龄组的个数
# 模拟的年数
years = 5
population = x0.copy()      # 复制初始雌性种群
populations = []            # 记录每年的种群数量
populations.append(population.copy())

# 存储每年的 alphas, betas, gammas, q_betas, q_gammas, l_betas, l_gammas, w_betas, w_gammas
alphas = []
betas = []
gammas = []
q_betas = []
q_gammas = []
l_betas = []
l_gammas = []
w_betas = []
w_gammas = []

for year in range(years):
    
    # %% 牧草需求
    alpha = (2/3) * np.sum(population[0:2]) + 1 * np.sum(population[2:12])
    
    # %% 粮食需求
    beta1 = 20 
    beta2 = 30
    beta3 = 30
    beta4 = 10
    beta = beta1 + beta2 + beta3 + beta4
    
    q_beta = 1.1 * beta1 + 0.9 * beta2 + 0.8 * beta3 + 0.6 * beta4
    
    l_beta = q_beta - 0.6 * np.sum(population[2:12])
    
    if l_beta > 0:
        w_beta = l_beta * 75
    else:
        w_beta = l_beta * 90

    # %% 甜菜需求 
    gamma = 10
    
    q_gamma = 1.5 * gamma
    
    l_gamma = q_gamma - 0.7 * np.sum(population[2:12])
    
    if l_gamma > 0:
        w_gamma = l_gamma * 58 
    else:
        w_gamma = l_gamma * 70

    # 记录到 alphas, betas, gammas, q_betas, q_gammas, l_betas, l_gammas, w_betas, w_gammas 中
    alphas.append(alpha)
    betas.append(beta)
    gammas.append(gamma)
    q_betas.append(q_beta)
    q_gammas.append(q_gamma)
    l_betas.append(l_beta)
    l_gammas.append(l_gamma)
    w_betas.append(w_beta)
    w_gammas.append(w_gamma)

    # %% 土地限制
    total_land = alpha + beta + gamma
    total_land_limit = 200
    # %% 种群迭代
    population = L_r @ population
    populations.append(population)
    
# %% 格式化变量
# 转换列表为数组
populations = np.array(populations)
alphas = np.array(alphas)
betas = np.array(betas)
gammas = np.array(gammas)
q_betas = np.array(q_betas)
q_gammas = np.array(q_gammas)
l_betas = np.array(l_betas)
l_gammas = np.array(l_gammas)
w_betas = np.array(w_betas)
w_gammas = np.array(w_gammas)
