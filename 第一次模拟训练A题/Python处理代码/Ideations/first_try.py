import numpy as np
from scipy.optimize import minimize


class Cattle:
    def __init__(self, x0, birth_rates, survival_rates, alpha, betas, gamma, r, M, years):
        self.x0 = np.array(x0)
        self.xs = [self.x0.copy()]
        self.birth_rates = np.array(birth_rates)
        self.survival_rates = np.array(survival_rates)
        self.alpha = alpha
        self.betas = np.array(betas)
        self.gamma = gamma
        self.r = r
        self.M = M
        self.m = self.calculate_m(M)  # 调用新的方法计算 m
        self.years = years
        self.n = len(x0)
        self.L_pp = self.create_L_pp()
        self.L_p = np.vstack([self.L_pp[0, :] * 0.5, self.L_pp[1:, :]])
        self.L_r = np.vstack([self.L_p[0, :] * (1 - r), self.L_p[1:, :]])
        self.y_1 = np.zeros(self.n); self.y_1[0] = 1
        self.y_12 = np.zeros(self.n); self.y_12[-1] = 1
        self.y_1_2 = np.zeros(self.n); self.y_1_2[0:2] = 1
        self.y_3_12 = np.zeros(self.n); self.y_3_12[2:] = 1
        self.reset_metrics()

    def calculate_m(self, M):
        # 计算 m 的逻辑
        return (M * 0.15 * (1 + 0.15)**10) / ((1 + 0.15)**10 - 1)

    def create_L_pp(self):
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        # 初始化指标，避免与初始变量重名
        self.alpha_values = np.array([0])
        self.betas_values = np.array([[0, 0, 0, 0]])
        self.gamma_values = np.array([0])
        self.q_betas_values = np.array([0])
        self.q_gammas_values = np.array([0])
        self.l_betas_values = np.array([0])
        self.l_gammas_values = np.array([0])
        self.w_betas_values = np.array([0])
        self.w_gammas_values = np.array([0])
        self.c_xiaomunius_values = np.array([0])
        self.c_damunius_values = np.array([0])
        self.c_betas_values = np.array([0])
        self.c_gammas_values = np.array([0])
        self.t_xiaomunius_values = np.array([0])
        self.t_damunius_values = np.array([0])
        self.t_betas_values = np.array([0])
        self.t_gammas_values = np.array([0])
        self.t_totals_values = np.array([0])
        self.c_workers_values = np.array([0])
        self.num_xiaogongniu_sales_values = np.array([0])
        self.num_xiaomuniu_sales_values = np.array([0])
        self.num_damuniu_sales_values = np.array([0])
        self.num_laomuniu_sales_values = np.array([0])
        self.w_xiaogongniu_values = np.array([0])
        self.w_xiaomuniu_values = np.array([0])
        self.w_damuniu_values = np.array([0])
        self.w_laomuniu_values = np.array([0])
        self.w_years_values = np.array([0])
        self.c_years_values = np.array([0])
        self.E_years_values = np.array([0])

    def simulate(self):
        total_penalty = 0
        
        for year in range(self.years + 1):
            x = self.xs[-1]

            # # 计算约束条件
            # constraint1 = (self.M / 200 + 130) - np.sum(x)
            # constraint2 = self.alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:]))
            # constraint3 = 200 - np.sum([self.alpha, *self.betas, self.gamma])
            # constraint4 = 50 - np.sum(x[2:])
            # constraint5 = np.sum(x[2:]) - 175

            # # 计算惩罚项
            # penalty1 = max(0, -constraint1) * 10
            # penalty2 = max(0, -constraint2) * 10
            # penalty3 = max(0, -constraint3) * 10
            # penalty4 = max(0, -constraint4) * 10 if year == self.years - 1 else 0
            # penalty5 = max(0, -constraint5) * 10 if year == self.years - 1 else 0

            # total_penalty += penalty1 + penalty2 + penalty3 + penalty4 + penalty5
            
            # # 调试信息
            # print(f"Year {year}: Profit = {self.calculate_total_profit()}, Total Penalty = {total_penalty}")

            self.update_metrics(x)
            self.update_x()

        profit = self.calculate_total_profit()

        # 将总惩罚项从利润中扣除
        return profit - total_penalty, np.array(self.xs)

    def validate(self):
        # 验证方法，不考虑惩罚
        for year in range(self.years + 1):
            x = self.xs[-1]
            self.update_metrics(x)
            self.update_x()
        
        return self.calculate_total_profit(), np.array(self.xs)

    def update_metrics(self, x):
        alpha = self.alpha
        beta1, beta2, beta3, beta4 = self.betas
        gamma = self.gamma
        
        q_beta = beta1 * 1.1 + beta2 * 0.9 + beta3 * 0.8 + beta4 * 0.6
        q_gamma = 1.5 * gamma
        l_beta = q_beta - 0.6 * np.sum(x[2:12])
        l_gamma = q_gamma - 0.7 * np.sum(x[2:12])
        
        w_beta = l_beta * 75 if l_beta > 0 else l_beta * 90
        w_gamma = l_gamma * 58 if l_gamma > 0 else l_gamma * 70

        self.alpha_values = np.append(self.alpha_values, alpha)
        self.betas_values = np.vstack([self.betas_values, [beta1, beta2, beta3, beta4]])
        self.gamma_values = np.append(self.gamma_values, gamma)
        self.q_betas_values = np.append(self.q_betas_values, q_beta)
        self.q_gammas_values = np.append(self.q_gammas_values, q_gamma)
        self.l_betas_values = np.append(self.l_betas_values, l_beta)
        self.l_gammas_values = np.append(self.l_gammas_values, l_gamma)
        self.w_betas_values = np.append(self.w_betas_values, w_beta)
        self.w_gammas_values = np.append(self.w_gammas_values, w_gamma)

        num_xiaogongniu_sales = self.L_p @ x @ self.y_1
        num_xiaomuniu_sales = self.L_p @ x @ self.y_1 * self.r
        num_damuniu_sales = self.L_r @ x @ self.y_3_12
        num_laomuniu_sales = self.L_r @ x @ self.y_12

        self.num_xiaogongniu_sales_values = np.append(self.num_xiaogongniu_sales_values, num_xiaogongniu_sales)
        self.num_xiaomuniu_sales_values = np.append(self.num_xiaomuniu_sales_values, num_xiaomuniu_sales)
        self.num_damuniu_sales_values = np.append(self.num_damuniu_sales_values, num_damuniu_sales)
        self.num_laomuniu_sales_values = np.append(self.num_laomuniu_sales_values, num_laomuniu_sales)

        self.w_xiaogongniu_values = np.append(self.w_xiaogongniu_values, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu_values = np.append(self.w_xiaomuniu_values, 40 * num_laomuniu_sales)
        self.w_damuniu_values = np.append(self.w_damuniu_values, 370 * num_damuniu_sales)
        self.w_laomuniu_values = np.append(self.w_laomuniu_values, 120 * num_laomuniu_sales)

        self.t_xiaomunius_values = np.append(self.t_xiaomunius_values, x @ self.y_1_2)
        self.t_damunius_values = np.append(self.t_damunius_values, x @ self.y_3_12)
        self.t_betas_values = np.append(self.t_betas_values, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas_values = np.append(self.t_gammas_values, 14 * gamma)
        self.t_totals_values = np.append(self.t_totals_values, x @ self.y_1_2 + x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)

        self.c_xiaomunius_values = np.append(self.c_xiaomunius_values, 500 * x @ self.y_1_2)
        self.c_damunius_values = np.append(self.c_damunius_values, 100 * x @ self.y_3_12)
        self.c_betas_values = np.append(self.c_betas_values, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas_values = np.append(self.c_gammas_values, 45 * gamma)
        self.c_workers_values = np.append(self.c_workers_values, 110 * np.sum(x[2:12]))

        self.w_years_values = np.append(self.w_years_values, np.sum([self.w_xiaogongniu_values[-1],self.w_xiaomuniu_values[-1], self.w_damuniu_values[-1],
                                                                     self.w_laomuniu_values[-1], w_beta, w_gamma]))
        
        self.c_years_values = np.append(self.c_years_values, np.sum([self.c_betas_values[-1], self.c_gammas_values[-1], self.c_xiaomunius_values[-1],
                                                                     self.c_damunius_values[-1], self.c_betas_values[-1], self.c_gammas_values[-1],
                                                                     self.c_workers_values[-1], self.m]))
        
        self.E_years_values = np.append(self.E_years_values, self.w_years_values[-1] - self.c_years_values[-1])

    def update_x(self):
        x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum(np.floor(x), 0))  # 取整并确保非负
        
    def calculate_total_profit(self):
        return np.sum(self.E_years_values)


class ConstraintManager:
    def __init__(self, cattle_params):
        # 初始化约束管理器，接收牛的参数
        self.cattle_params = cattle_params

    def constraint1(self, params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=5
        )
        _, xs = cattle.simulate()
        return np.array([(M / 200 + 130) - np.sum(x) for x in xs])
    
    def constraint2(self, params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=5
        )
        _, xs = cattle.simulate()
        return np.array([alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:])) for x in xs])

    def constraint3(self, params):
        # 约束3：确保参数的总和不超过200
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        return np.array([200 - np.sum([alpha, beta1, beta2, beta3, beta4, gamma])])

    def constraint4(self, params):
        # 约束4：确保最后一个时间点的某些种群数量不超过50
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=5
        )
        _, xs = cattle.simulate()
        return np.array([50 - np.sum(xs[-1][2:])])  # 返回最后一个时间点的总和与50的差值

    def constraint5(self, params):
        # 约束5：确保最后一个时间点的某些种群数量至少为175
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        cattle = Cattle(
            x0=np.ones(12) * 10,
            birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
            survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
            alpha=alpha,
            betas=[beta1, beta2, beta3, beta4],
            gamma=gamma,
            r=r,
            M=M,
            years=5
        )
        _, xs = cattle.simulate()
        return np.array([np.sum(xs[-1][2:]) - 175])  # 返回最后一个时间点的总和与175的差值

    def get_constraints(self, params):
        # 获取所有约束条件并合并为一个数组
        constraints = np.concatenate([
            self.constraint1(params),
            self.constraint2(params),
            self.constraint3(params),
            self.constraint4(params),
            self.constraint5(params)
        ])
        return constraints
    
    
# 示例用法：
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
            M=M,
            years=5
        )
        penalty = 0
        
        if beta1 < 10: penalty += 10000                 # 惩罚beta1过小
        if beta2 < 10: penalty += 10000                 # 惩罚beta2过小
        if beta3 < 10: penalty += 10000                 # 惩罚beta3过小
        if beta4 < 10: penalty += 10000                 # 惩罚beta4过小
        if gamma < 10: penalty += 10000                 # 惩罚gamma过小
        
        profit, _ = cattle.simulate()
        return -profit + penalty    # 包含惩罚项
    
    # 初始化约束管理器
    constraint_manager = ConstraintManager(cattle_params={})
    
    # 定义约束函数
    def constraint_function(params):
        return constraint_manager.get_constraints(params)
    
    # 初值猜想
    initial_guess = [1.0, 500000, 2/3*20+100, 10, 10, 10, 10, 70/1.5]
    
    bounds = [
        (0, 1),                     # r
        (0, 1000000),               # M
        (2/3*20+100, 200),          # alpha
        (0, 20),                    # beta1
        (0, 30),                    # beta2
        (0, 30),                    # beta3
        (0, 10),                    # beta4
        (70/1.5, 200)               # gamma
    ]
    
    result = minimize(objective_function, initial_guess, bounds=bounds,
                      constraints={'type': 'ineq', 'fun': constraint_function}, method='SLSQP')
    
    print(f"Optimal parameters: \n{result.x}")
    print(f"Maximum profit: \n{-result.fun}")

    # 使用返回的最佳参数进行模拟验证
    optimal_params = result.x
    r, M, alpha, beta1, beta2, beta3, beta4, gamma = optimal_params

    cattle = Cattle(
        x0=np.ones(12) * 10,
        birth_rates=np.array([0., 0., 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
        survival_rates=np.array([0.95, 0.95, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]),
        alpha=alpha,
        betas=[beta1, beta2, beta3, beta4],
        gamma=gamma,
        r=r,
        M=M,
        years=5
    )

    final_profit, _ = cattle.simulate()
    print(f"Final profit with optimal parameters:\n{final_profit}")
