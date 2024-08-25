import numpy as np
from scipy.optimize import differential_evolution


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

    def create_L_pp(self):
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        self.alpha_metrics = np.array([0])
        self.betas_metrics = np.array([[0, 0, 0, 0]])
        self.gammas = np.array([0])
        self.q_betas = np.array([0])
        self.q_gammas = np.array([0])
        self.l_betas = np.array([0])
        self.l_gammas = np.array([0])
        self.w_betas = np.array([0])
        self.w_gammas = np.array([0])
        self.c_xiaomunius = np.array([0])
        self.c_damunius = np.array([0])
        self.c_betas = np.array([0])
        self.c_gammas = np.array([0])
        self.t_xiaomunius = np.array([0])
        self.t_damunius = np.array([0])
        self.t_betas = np.array([0])
        self.t_gammas = np.array([0])
        self.t_totals = np.array([0])
        self.c_workers = np.array([0])
        self.num_xiaogongniu_sales = np.array([0])
        self.num_xiaomuniu_sales = np.array([0])
        self.num_damuniu_sales = np.array([0])
        self.num_laomuniu_sales = np.array([0])
        self.w_xiaogongniu = np.array([0])
        self.w_xiaomuniu = np.array([0])
        self.w_damuniu = np.array([0])
        self.w_laomuniu = np.array([0])
        self.w_years = np.array([0])
        self.c_years = np.array([0])
        self.E_years = np.array([0])

    def simulate(self):
        total_penalty = 0
        last_x = self.xs[-1].copy()  # 记录上一个状态

        for year in range(self.years):
            x = self.xs[-1]

            # 计算约束条件
            constraint1 = (self.M / 200 + 130) - np.sum(x)
            constraint2 = self.alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:]))
            constraint3 = 200 - np.sum([self.alpha, *self.betas, self.gamma])
            constraint4 = 50 - np.sum(x[2:])
            constraint5 = np.sum(x[2:]) - 175

            # 计算惩罚项
            penalty1 = max(0, -constraint1) * 10000
            penalty2 = max(0, -constraint2) * 10000
            penalty3 = max(0, -constraint3) * 10000
            penalty4 = max(0, -constraint4) * 1000 if year == self.years - 1 else 0
            penalty5 = max(0, -constraint5) * 1000 if year == self.years - 1 else 0

            total_penalty += penalty1 + penalty2 + penalty3 + penalty4 + penalty5

            self.update_metrics(x)
            self.update_x()

            # 检查惩罚是否增加，如果增加则回退
            if total_penalty > 0 and total_penalty > self.calculate_penalty(last_x):
                self.xs[-1] = last_x.copy()  # 回退到上一个状态
                break  # 结束模拟

            last_x = x.copy()  # 更新上一个状态

        profit = self.calculate_total_profit()

        # 调整参数
        if penalty1 > 0:
            self.M *= 1.1  # 增大 M
        if penalty2 > 0:
            self.alpha *= 1.1  # 增大 alpha
        if penalty3 > 0:
            self.betas *= 1.1  # 增大 betas
            self.gamma *= 1.1  # 增大 gamma

        return profit - total_penalty, np.array(self.xs)

    def calculate_penalty(self, x):
        constraint1 = (self.M / 200 + 130) - np.sum(x)
        constraint2 = self.alpha - (2 / 3 * np.sum(x[:2]) + np.sum(x[2:]))
        constraint3 = 200 - np.sum([self.alpha, *self.betas, self.gamma])
        constraint4 = 50 - np.sum(x[2:])
        constraint5 = np.sum(x[2:]) - 175

        penalty1 = max(0, -constraint1) * 100
        penalty2 = max(0, -constraint2) * 100
        penalty3 = max(0, -constraint3) * 100
        penalty4 = max(0, -constraint4) * 100
        penalty5 = max(0, -constraint5) * 100

        return penalty1 + penalty2 + penalty3 + penalty4 + penalty5

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

        self.alpha_metrics = np.append(self.alpha_metrics, alpha)
        self.betas_metrics = np.vstack([self.betas_metrics, [beta1, beta2, beta3, beta4]])
        self.gammas = np.append(self.gammas, gamma)
        self.q_betas = np.append(self.q_betas, q_beta)
        self.q_gammas = np.append(self.q_gammas, q_gamma)
        self.l_betas = np.append(self.l_betas, l_beta)
        self.l_gammas = np.append(self.l_gammas, l_gamma)
        self.w_betas = np.append(self.w_betas, w_beta)
        self.w_gammas = np.append(self.w_gammas, w_gamma)

        num_xiaogongniu_sales = self.L_p @ x @ self.y_1
        num_xiaomuniu_sales = self.L_p @ x @ self.y_1 * self.r
        num_damuniu_sales = self.L_r @ x @ self.y_3_12
        num_laomuniu_sales = self.L_r @ x @ self.y_12

        self.num_xiaogongniu_sales = np.append(self.num_xiaogongniu_sales, num_xiaogongniu_sales)
       
        self.num_xiaomuniu_sales = np.append(self.num_xiaomuniu_sales, num_xiaomuniu_sales)
        self.num_damuniu_sales = np.append(self.num_damuniu_sales, num_damuniu_sales)
        self.num_laomuniu_sales = np.append(self.num_laomuniu_sales, num_laomuniu_sales)

        self.w_xiaogongniu = np.append(self.w_xiaogongniu, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu = np.append(self.w_xiaomuniu, 40 * num_laomuniu_sales)
        self.w_damuniu = np.append(self.w_damuniu, 370 * num_damuniu_sales)
        self.w_laomuniu = np.append(self.w_laomuniu, 120 * num_laomuniu_sales)

        self.t_xiaomunius = np.append(self.t_xiaomunius, x @ self.y_1_2)
        self.t_damunius = np.append(self.t_damunius, x @ self.y_3_12)
        self.t_betas = np.append(self.t_betas, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas = np.append(self.t_gammas, 14 * gamma)
        self.t_totals = np.append(self.t_totals, x @ self.y_1_2 + x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)

        self.c_xiaomunius = np.append(self.c_xiaomunius, 500 * x @ self.y_1_2)
        self.c_damunius = np.append(self.c_damunius, 100 * x @ self.y_3_12)
        self.c_betas = np.append(self.c_betas, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas = np.append(self.c_gammas, 45 * gamma)
        self.c_workers = np.append(self.c_workers, 110 * np.sum(x[2:12]))

        self.w_years = np.append(self.w_years, np.sum([self.w_xiaogongniu[-1], self.w_xiaomuniu[-1], self.w_damuniu[-1], self.w_laomuniu[-1]]))
        self.c_years = np.append(self.c_years, np.sum([self.c_xiaomunius[-1], self.c_damunius[-1], self.c_betas[-1], self.c_gammas[-1], self.c_workers[-1]]))
        self.E_years = np.append(self.E_years, self.w_years[-1] - self.c_years[-1])

    def update_x(self):
        next_x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum(np.floor(next_x), 0))  # 取整并确保非负

    def calculate_total_profit(self):
        return np.sum(self.E_years)

    
if __name__ == '__main__':
    def objective_function(params):
        r, M, alpha, beta1, beta2, beta3, beta4, gamma = params
        
        penalty = 0
        if beta1 < 10: penalty += 10000  # 惩罚beta1过小
        if beta2 < 10: penalty += 10000  # 惩罚beta2过小
        if beta3 < 10: penalty += 10000  # 惩罚beta3过小
        if beta4 < 10: penalty += 10000  # 惩罚beta4过小
        if gamma < 10: penalty += 10000  # 惩罚gamma过小
    
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
        
        profit, _ = cattle.simulate()
        return -profit + penalty    # 包含惩罚项
    
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

    result = differential_evolution(objective_function, bounds)
    
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
    print(f"Final profit with optimal parameters: {final_profit}")
