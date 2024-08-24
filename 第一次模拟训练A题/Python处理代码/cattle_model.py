import numpy as np
from deap import base, creator, tools, algorithms

# 牛群模型类定义
class CattleModel:
    def __init__(self, initial_population, birth_rates, survival_rates, alphas, betas, gamma, r, M, years=5):
        self.x0 = initial_population
        self.birth_rates = birth_rates
        self.survival_rates = survival_rates
        self.alphas = alphas
        self.betas = betas
        self.gamma = gamma
        self.r = r
        self.M = M
        self.years = years
        self.n = len(initial_population)
        self.L_pp = self.create_L_pp()
        self.L_p = np.vstack([self.L_pp[0, :] * 0.5, self.L_pp[1:, :]])
        self.L_r = np.vstack([self.L_p[0, :] * (1 - r), self.L_p[1:, :]])
        self.y_1 = np.zeros(self.n); self.y_1[0] = 1
        self.y_12 = np.zeros(self.n); self.y_12[-1] = 1
        self.y_1_2 = np.zeros(self.n); self.y_1_2[0:2] = 1
        self.y_3_12 = np.zeros(self.n); self.y_3_12[2:] = 1
        self.populations = [self.x0.copy()]
        self.reset_metrics()

    def create_L_pp(self):
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        self.alphas_metrics = [0]
        self.betas_metrics = [[0, 0, 0, 0]]
        self.gammas = [0]
        self.q_betas = [0]
        self.q_gammas = [0]
        self.l_betas = [0]
        self.l_gammas = [0]
        self.w_betas = [0]
        self.w_gammas = [0]
        self.c_xiaomunius = [0]
        self.c_damunius = [0]
        self.c_betas = [0]
        self.c_gammas = [0]
        self.t_xiaomunius = [0]
        self.t_damunius = [0]
        self.t_betas = [0]
        self.t_gammas = [0]
        self.t_totals = [0]
        self.c_workers = [0]
        self.num_xiaogongniu_sales = [0]
        self.num_xiaomuniu_sales = [0]
        self.num_damuniu_sales = [0]
        self.num_laomuniu_sales = [0]
        self.w_xiaogongniu = [0]
        self.w_xiaomuniu = [0]
        self.w_damuniu = [0]
        self.w_laomuniu = [0]
        self.w_years = [0]
        self.c_years = [0]
        self.E_years = [0]

    def simulate(self):
        for year in range(self.years):
            population = self.populations[-1]
            self.update_metrics(population)
            self.update_population()
        return -self.calculate_total_profit()

    def update_metrics(self, population):
        alpha = self.alphas
        beta1, beta2, beta3, beta4 = self.betas
        gamma = self.gamma
        
        q_beta = beta1 * 1.1 + beta2 * 0.9 + beta3 * 0.8 + beta4 * 0.6
        q_gamma = 1.5 * gamma
        l_beta = q_beta - 0.6 * np.sum(population[2:12])
        l_gamma = q_gamma - 0.7 * np.sum(population[2:12])
        
        w_beta = l_beta * 75 if l_beta > 0 else l_beta * 90
        w_gamma = l_gamma * 58 if l_gamma > 0 else l_gamma * 70

        self.alphas_metrics.append(alpha)
        self.betas_metrics.append([beta1, beta2, beta3, beta4])
        self.gammas.append(gamma)
        self.q_betas.append(q_beta)
        self.q_gammas.append(q_gamma)
        self.l_betas.append(l_beta)
        self.l_gammas.append(l_gamma)
        self.w_betas.append(w_beta)
        self.w_gammas.append(w_gamma)

        num_xiaogongniu_sales = self.L_p @ population @ self.y_1
        num_xiaomuniu_sales = self.L_p @ population @ self.y_1 * self.r
        num_damuniu_sales = self.L_r @ population @ self.y_3_12
        num_laomuniu_sales = self.L_r @ population @ self.y_12

        self.num_xiaogongniu_sales.append(num_xiaogongniu_sales)
        self.num_xiaomuniu_sales.append(num_xiaomuniu_sales)
        self.num_damuniu_sales.append(num_damuniu_sales)
        self.num_laomuniu_sales.append(num_laomuniu_sales)

        self.w_xiaogongniu.append(30 * num_xiaogongniu_sales)
        self.w_xiaomuniu.append(40 * num_laomuniu_sales)
        self.w_damuniu.append(370 * num_damuniu_sales)
        self.w_laomuniu.append(120 * num_laomuniu_sales)

        self.t_xiaomunius.append(population @ self.y_1_2)
        self.t_damunius.append(population @ self.y_3_12)
        self.t_betas.append(4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas.append(14 * gamma)
        self.t_totals.append(population @ self.y_1_2 + population @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)

        self.c_xiaomunius.append(500 * population @ self.y_1_2)
        self.c_damunius.append(100 * population @ self.y_3_12)
        self.c_betas.append(15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas.append(10 * gamma)
        self.c_workers.append(4000 if self.t_totals[-1] <= 5500 else 4000 + self.t_totals[-1] * 1.2)

        w_year = (self.w_xiaogongniu[-1] + self.w_xiaomuniu[-1] + self.w_damuniu[-1] + self.w_laomuniu[-1] + self.w_betas[-1] + self.w_gammas[-1])
        self.w_years.append(w_year)

        c_year = (self.c_betas[-1] + self.c_gammas[-1] + self.c_xiaomunius[-1] + self.c_damunius[-1] + self.c_workers[-1] + (self.M * 0.15) / (1 - (1 + 0.15) ** -10))
        self.c_years.append(c_year)

        E_year = w_year - c_year
        self.E_years.append(E_year)

    def update_population(self):
        population = self.L_r @ self.populations[-1]
        self.populations.append(population)

    def calculate_total_profit(self):
        return sum(self.E_years)

def create_cattle_model(params):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = params
    return CattleModel(
        initial_population=np.ones(12) * 10,
        birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
        survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
        alphas=alpha,
        betas=[beta1, beta2, beta3, beta4],
        gamma=gamma,
        r=r,
        M=M
    )

def constraint_factory(cattle_model_func, constraint_value_func):
    def constraint(params):
        model = cattle_model_func(params)
        try:
            for _ in range(5):  # Simulate for 5 years
                model.simulate()
            populations = model.populations
            return constraint_value_func(populations, params)
        except Exception as e:
            print(f"Error in constraint calculation: {e}")
            return 1e6  # Return a large number on error
    return constraint

def constraint1_value(populations, params):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = params
    return np.sum(populations[-1]) - (M / 200 + 130)

def constraint2_value(populations, params):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = params
    return M - 200 * (45 + np.sum(populations[-1][:2]))

def constraint3_value(populations, params):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = params
    return alpha - (2 / 3 * np.sum(populations[-1][:2]) + np.sum(populations[-1][2:]))

def constraint4_value(populations, params):
    return np.sum(populations[-1][2:]) - 50

def constraint5_value(populations, params):
    return 175 - np.sum(populations[-1][2:])

def evaluate(individual):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = individual
    if alpha + beta1 + beta2 + beta3 + beta4 + gamma > 200:
        return 10000000,  # 返回一个大的数字以表示不合格
    
    model = create_cattle_model(individual)
    
    try:
        profit = model.simulate()
    except Exception as e:
        print(f"Error in simulation: {e}")
        profit = 10000000  # 返回一个大的数字以表示错误
    
    return profit,

def mutate(individual):
    mut_choice = np.random.randint(0, len(individual))
    individual[mut_choice] += np.random.uniform(-1, 1)  # 随机变异
    return individual,

def crossover(ind1, ind2):
    if np.random.rand() < 0.5:  # 50% 概率进行交叉
        cxpoint = np.random.randint(1, len(ind1)-1)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2

# 创建适应度和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 设置遗传算法参数
population_size = 100
num_generations = 300
mutation_rate = 0.2
crossover_rate = 0.5
elite_size = 5  # 精英个体数量

toolbox = base.Toolbox()
toolbox.register("alpha", np.random.uniform, 0, 100)
toolbox.register("beta1", np.random.uniform, 0, 20)
toolbox.register("beta2", np.random.uniform, 0, 30)
toolbox.register("beta3", np.random.uniform, 0, 30)
toolbox.register("beta4", np.random.uniform, 0, 10)
toolbox.register("gamma", np.random.uniform, 0, 100)
toolbox.register("r", np.random.uniform, 0, 1)
toolbox.register("M", np.random.uniform, 0, 1000000)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.alpha, toolbox.beta1, toolbox.beta2, toolbox.beta3, toolbox.beta4, toolbox.gamma, toolbox.r, toolbox.M), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义约束条件
constraints = [
    {'type': 'ineq', 'fun': constraint_factory(create_cattle_model, constraint1_value)},
    {'type': 'ineq', 'fun': constraint_factory(create_cattle_model, constraint2_value)},
    {'type': 'ineq', 'fun': constraint_factory(create_cattle_model, constraint3_value)},
    {'type': 'ineq', 'fun': constraint_factory(create_cattle_model, constraint4_value)},
    {'type': 'ineq', 'fun': constraint_factory(create_cattle_model, constraint5_value)}
]

# 主程序
population = toolbox.population(n=population_size)
best_fitness = float('inf')
stagnation_counter = 0

for gen in range(num_generations):
    # 评估所有个体
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 记录当前最佳适应度
    current_best = min(ind.fitness.values[0] for ind in population if ind.fitness.valid)
    if current_best < best_fitness:
        best_fitness = current_best
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    # 动态调整变异率和交叉率
    if stagnation_counter > 10:
        mutation_rate = min(0.5, mutation_rate * 1.1)
        crossover_rate = max(0.3, crossover_rate * 0.9)
        stagnation_counter = 0

    # 精英保留
    elites = tools.selBest(population, elite_size)
    
    # 选择
    offspring = toolbox.select(population, len(population) - elite_size)
    offspring = list(map(toolbox.clone, offspring))

    # 交叉和变异
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < crossover_rate:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < mutation_rate:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 更新种群
    population[:] = elites + offspring

    # 打印当前代数和最佳适应度
    print(f"Generation {gen}: Best Fitness = {best_fitness}")

# 最终评估
fits = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
if not fits:
    print("警告：没有有效的适应度值")
else:
    best_idx = np.argmin(fits)
    best_ind = population[best_idx]

    print(f"最优个体: {best_ind}")
    print(f"最优利润: {-fits[best_idx]}")  # 取负值以得到利润


print(f"最优个体: {best_ind}")
print(f"最优利润: {-fits[best_idx]}")  # 取负值以得到利润

# 验证最优解是否满足约束条件
best_model = create_cattle_model(best_ind)
best_model.simulate()
for i, constraint in enumerate(constraints):
    constraint_value = constraint['fun'](best_ind)
    print(f"约束条件 {i+1} 的值: {constraint_value}")
    if constraint_value < 0:
        print(f"警告: 约束条件 {i+1} 未满足")

# 输出最优解的详细信息
print("\n最优解详细信息:")
print(f"alpha: {best_ind[0]}")
print(f"beta1: {best_ind[1]}")
print(f"beta2: {best_ind[2]}")
print(f"beta3: {best_ind[3]}")
print(f"beta4: {best_ind[4]}")
print(f"gamma: {best_ind[5]}")
print(f"r: {best_ind[6]}")
print(f"M: {best_ind[7]}")

# 输出最优解的牛群数量变化
print("\n牛群数量变化:")
for year, population in enumerate(best_model.populations):
    print(f"年份 {year}: {population}")

# 输出最优解的各项指标
print("\n各项指标:")
print(f"总利润: {best_model.calculate_total_profit()}")
print(f"年度利润: {best_model.E_years}")
print(f"年度收入: {best_model.w_years}")
print(f"年度成本: {best_model.c_years}")