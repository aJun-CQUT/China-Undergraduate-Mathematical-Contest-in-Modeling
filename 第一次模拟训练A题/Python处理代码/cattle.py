import numpy as np
from scipy.optimize import minimize, Bounds


class Cattle:
    def __init__(self, x0, birth_rates, survival_rates, alpha, betas, gamma, r, M, years=5):
        # Leslie 矩阵
        self.x0 = np.array(x0)                         # 初始种群数量分布向量
        self.xs = [self.x0.copy()]                     # 种群数量分布向量列表
        self.birth_rates = np.array(birth_rates)       # 出生率
        self.survival_rates = np.array(survival_rates) # 存活率
        self.alpha = alpha                             # alpha参数
        self.betas = np.array(betas)                   # beta参数
        self.gamma = gamma                             # gamma参数
        self.r = r                                     # r参数
        self.M = M                                     # M参数
        self.years = years                             # 模拟年数
        self.n = len(x0)                               # 种群中年龄组的个数
        self.L_pp = self.create_L_pp()                 # 创建L_pp矩阵
        self.L_p = np.vstack([self.L_pp[0, :] * 0.5, self.L_pp[1:, :]])    # 创建L_p矩阵
        self.L_r = np.vstack([self.L_p[0, :] * (1 - r), self.L_p[1:, :]])  # 创建L_r矩阵
        # 选择向量
        self.y_1 = np.zeros(self.n); self.y_1[0] = 1                       # y_1向量
        self.y_12 = np.zeros(self.n); self.y_12[-1] = 1                    # y_12向量
        self.y_1_2 = np.zeros(self.n); self.y_1_2[0:2] = 1                 # y_1_2向量
        self.y_3_12 = np.zeros(self.n); self.y_3_12[2:] = 1                # y_3_12向量
        self.reset_metrics()                                               # 重置指标

    def create_L_pp(self):
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        # 初始化指标
        self.alpha_metrics = np.array([0])             # alpha指标
        self.betas_metrics = np.array([[0, 0, 0, 0]])  # beta指标
        self.gammas = np.array([0])                    # gamma指标
        self.q_betas = np.array([0])                   # q_beta指标
        self.q_gammas = np.array([0])                  # q_gamma指标
        self.l_betas = np.array([0])                   # l_beta指标
        self.l_gammas = np.array([0])                  # l_gamma指标
        self.w_betas = np.array([0])                   # w_beta指标
        self.w_gammas = np.array([0])                  # w_gamma指标
        self.c_xiaomunius = np.array([0])              # 小母牛成本
        self.c_damunius = np.array([0])                # 大母牛成本
        self.c_betas = np.array([0])                   # beta成本
        self.c_gammas = np.array([0])                  # gamma成本
        self.t_xiaomunius = np.array([0])              # 小母牛总数
        self.t_damunius = np.array([0])                # 大母牛总数
        self.t_betas = np.array([0])                   # beta总数
        self.t_gammas = np.array([0])                  # gamma总数
        self.t_totals = np.array([0])                  # 总数
        self.c_workers = np.array([0])                 # 工人成本
        self.num_xiaogongniu_sales = np.array([0])     # 小公牛销售数量
        self.num_xiaomuniu_sales = np.array([0])       # 小母牛销售数量
        self.num_damuniu_sales = np.array([0])         # 大母牛销售数量
        self.num_laomuniu_sales = np.array([0])        # 老母牛销售数量
        self.w_xiaogongniu = np.array([0])             # 小公牛收入
        self.w_xiaomuniu = np.array([0])               # 小母牛收入
        self.w_damuniu = np.array([0])                 # 大母牛收入
        self.w_laomuniu = np.array([0])                # 老母牛收入
        self.w_years = np.array([0])                   # 年收入
        self.c_years = np.array([0])                   # 年成本
        self.E_years = np.array([0])                   # 年利润

    def simulate(self):
        for year in range(self.years):
            x = self.xs[-1]                            # 当前年份的种群数量分布向量
            self.update_metrics(x)                     # 更新指标
            self.update_x()                            # 更新种群数量分布向量
        profit = self.calculate_total_profit()         # 计算总利润
        return profit, np.array(self.xs)               # 返回总利润和种群数量分布向量数组

    def update_metrics(self, x):
        alpha = self.alpha
        beta1, beta2, beta3, beta4 = self.betas
        gamma = self.gamma
        
        q_beta = beta1 * 1.1 + beta2 * 0.9 + beta3 * 0.8 + beta4 * 0.6  # q_beta计算
        q_gamma = 1.5 * gamma                                           # q_gamma计算
        l_beta = q_beta - 0.6 * np.sum(x[2:12])                         # l_beta计算
        l_gamma = q_gamma - 0.7 * np.sum(x[2:12])                       # l_gamma计算
        
        w_beta = l_beta * 75 if l_beta > 0 else l_beta * 90             # w_beta计算
        w_gamma = l_gamma * 58 if l_gamma > 0 else l_gamma * 70         # w_gamma计算

        # 更新指标
        self.alpha_metrics = np.append(self.alpha_metrics, alpha)
        self.betas_metrics = np.vstack([self.betas_metrics, [beta1, beta2, beta3, beta4]])
        self.gammas = np.append(self.gammas, gamma)
        self.q_betas = np.append(self.q_betas, q_beta)
        self.q_gammas = np.append(self.q_gammas, q_gamma)
        self.l_betas = np.append(self.l_betas, l_beta)
        self.l_gammas = np.append(self.l_gammas, l_gamma)
        self.w_betas = np.append(self.w_betas, w_beta)
        self.w_gammas = np.append(self.w_gammas, w_gamma)

        # 销售数量计算
        num_xiaogongniu_sales = self.L_p @ x @ self.y_1
        num_xiaomuniu_sales = self.L_p @ x @ self.y_1 * self.r
        num_damuniu_sales = self.L_r @ x @ self.y_3_12
        num_laomuniu_sales = self.L_r @ x @ self.y_12

        # 更新销售数量
        self.num_xiaogongniu_sales = np.append(self.num_xiaogongniu_sales, num_xiaogongniu_sales)
        self.num_xiaomuniu_sales = np.append(self.num_xiaomuniu_sales, num_xiaomuniu_sales)
        self.num_damuniu_sales = np.append(self.num_damuniu_sales, num_damuniu_sales)
        self.num_laomuniu_sales = np.append(self.num_laomuniu_sales, num_laomuniu_sales)

        # 收入计算
        self.w_xiaogongniu = np.append(self.w_xiaogongniu, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu = np.append(self.w_xiaomuniu, 40 * num_laomuniu_sales)
        self.w_damuniu = np.append(self.w_damuniu, 370 * num_damuniu_sales)
        self.w_laomuniu = np.append(self.w_laomuniu, 120 * num_laomuniu_sales)

        # 总数和成本计算
        self.t_xiaomunius = np.append(self.t_xiaomunius, x @ self.y_1_2)
        self.t_damunius = np.append(self.t_damunius, x @ self.y_3_12)
        self.t_betas = np.append(self.t_betas, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas = np.append(self.t_gammas, 14 * gamma)
        self.t_totals = np.append(self.t_totals, x @ self.y_1_2 + x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)

        self.c_xiaomunius = np.append(self.c_xiaomunius, 500 * x @ self.y_1_2)
        self.c_damunius = np.append(self.c_damunius, 100 * x @ self.y_3_12)
        self.c_betas = np.append(self.c_betas, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas = np.append(self.c_gammas, 10 * gamma)
        self.c_workers = np.append(self.c_workers, 4000 if self.t_totals[-1] <= 5500 else 4000 + self.t_totals[-1] * 1.2)

        # 年收入和年成本计算
        w_year = (self.w_xiaogongniu[-1] + self.w_xiaomuniu[-1] + self.w_damuniu[-1] + self.w_laomuniu[-1] + self.w_betas[-1] + self.w_gammas[-1])
        self.w_years = np.append(self.w_years, w_year)

        c_year = (self.c_betas[-1] + self.c_gammas[-1] + self.c_xiaomunius[-1] + self.c_damunius[-1] + self.c_workers[-1] + (self.M * 0.15) / (1 - (1 + 0.15) ** -10))
        self.c_years = np.append(self.c_years, c_year)

        E_year = w_year - c_year
        self.E_years = np.append(self.E_years, E_year)

    def update_x(self):
        x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum(np.floor(x), 0))  # 取整并确保非负

    def calculate_total_profit(self):
        return np.sum(self.E_years)


if __name__ == '__main__':
    # 定义目标函数
    def objective_function(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M
        )
        profit, xs = cattle.simulate()
        return -profit  # 最优化问题通常最小化目标函数，因此我们取负值来最大化利润
    
    # 约束条件
    def constraint1(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M
        )
        _, xs = cattle.simulate()
        return [(M / 200 + 130) - np.sum(x) for x in xs]
    
    def constraint2(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M
        )
        _, xs = cattle.simulate()
        return [alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:])) for x in xs]
    
    def constraint3(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        return 200 - np.sum([alpha, beta1, beta2, beta3, beta4, gamma])
    
    def constraint4(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M
        )
        _, xs = cattle.simulate()
        return 50 - np.sum(xs[-1][2:])
    
    def constraint5(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M
        )
        _, xs = cattle.simulate()
        return np.sum(xs[-1][2:]) - 175

    # 初值猜想
    initial_guess = [0.5, 500000, 20, 10, 10, 10, 10, 5]
    
    # 参数的边界
    bounds = Bounds([0, 0, 50, 0, 0, 0, 0, 0], [1, 1000000, 200, 20, 30, 30, 10, 200])

    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4},
        {'type': 'ineq', 'fun': constraint5}
    ]

    # 调用优化函数
    result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

    # 输出结果
    print("优化结果:")
    print(result)
