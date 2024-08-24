import numpy as np
from deap import base, creator, tools
from cattle import Cattle

# 创建牛群模型的函数
def create_cattle_model(params):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = params
    return Cattle(
        x0=np.ones(12) * 10,
        birth_rates=np.array([0., 0., 1.1] * 4),
        survival_rates=np.array([0.95] * 11),
        alphas=alpha,
        betas=[beta1, beta2, beta3, beta4],
        gamma=gamma,
        r=r,
        M=M
    )

# 约束条件计算函数
def calculate_constraints(model, params):
    try:
        xs = model.simulate()
        return [
            np.sum(xs[-1]) - (params[-1] / 200 + 130),
            params[-1] - 200 * (45 + np.sum(xs[-1][:2])),
            params[0] - (2 / 3 * np.sum(xs[-1][:2]) + np.sum(xs[-1][2:])),
            np.sum(xs[-1][2:]) - 50,
            175 - np.sum(xs[-1][2:])
        ]
    except Exception as e:
        print(f"Error in simulation: {e}")
        return [1e6] * 5

# 适应度评估函数
def evaluate(individual):
    alpha, beta1, beta2, beta3, beta4, gamma, r, M = individual
    
    # 确保 `r` 和参数总和的合法性
    if r < 0 or r > 1 or alpha + beta1 + beta2 + beta3 + beta4 + gamma > 200:
        return 1000000,

    model = create_cattle_model(individual)
    constraints = calculate_constraints(model, individual)

    if any(c < 0 for c in constraints[:3]):
        return 1000000,  # 违反约束条件1、2、3

    try:
        profit = model.calculate_total_profit()
    except Exception as e:
        print(f"Error in profit calculation: {e}")
        profit = 1000000

    return profit,

# 变异函数
def mutate(individual):
    mut_choice = np.random.randint(len(individual))
    mutation = np.random.uniform(-1, 1)
    individual[mut_choice] += mutation
    individual[mut_choice] = max(0.1, individual[mut_choice])
    
    if mut_choice == 6:
        individual[mut_choice] = np.clip(individual[mut_choice], 0, 1)
        
    return individual,

# 交叉函数
def crossover(ind1, ind2):
    if np.random.rand() < 0.5:
        cxpoint = np.random.randint(1, len(ind1) - 1)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    
    ind1 = np.maximum(ind1, 0.1)
    ind2 = np.maximum(ind2, 0.1)
    
    return ind1, ind2

# 设置遗传算法参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

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

# 主程序
def main():
    population_size = 30
    num_generations = 5
    mutation_rate = 0.2
    crossover_rate = 0.5
    elite_size = 5  # 精英个体数量
    
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
            best_ind = min((ind for ind in population if ind.fitness.valid), key=lambda ind: ind.fitness.values[0])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # 动态调整变异和交叉率
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
        print(f"Generation {gen}: Best Fitness = {-best_fitness}")

        # 检查约束条件5，只在第五次迭代时
        if gen == 4:
            if 'best_ind' in locals():
                best_model = create_cattle_model(best_ind)
                constraints = calculate_constraints(best_model, best_ind)
                print(f"约束条件 4: {'满足' if constraints[3] >= 0 else '不满足'} (值: {constraints[3]})")
                print(f"约束条件 5: {'满足' if constraints[4] >= 0 else '不满足'} (值: {constraints[4]})")

    # 最终模型评估
    fits = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
    if not fits:
        print("警告：没有有效的适应度值")
    else:
        best_idx = np.argmin(fits)
        best_ind = population[best_idx]

        print("\n最优结果:")
        print(f"\n最优个体: {best_ind}")
        print(f"\n最优适应度: {-fits[best_idx]}")

        # 使用最优参数创建并模拟牛群模型
        best_model = create_cattle_model(best_ind)
        best_model.simulate()

        print(f"总利润: {best_model.calculate_total_profit()}")
        print(f"最终种群: {best_model.xs[-1]}")
        print(f"年度收入: {best_model.w_years}")
        print(f"年度成本: {best_model.c_years}")
        print(f"年度利润: {best_model.E_years}")

        # 检查约束条件
        for i, constraint in enumerate(calculate_constraints(best_model, best_ind)):
            print(f"约束条件 {i+1}: {'满足' if constraint >= 0 else '不满足'} (值: {constraint})")

if __name__ == "__main__":
    main()
