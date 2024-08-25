import numpy as np
from scipy.optimize import differential_evolution


class Cattle:
    def __init__(self, x0, birth_rates, survival_rates, alpha, betas, gamma, r, M, years):
        self.x0 = np.array(x0)                    # 初始种群分布
        self.xs = [self.x0.copy()]                # 种群分布历史记录
        self.birth_rates = np.array(birth_rates)  
        self.survival_rates = np.array(survival_rates)
        self.alpha = alpha                        # 种植牧草
        self.betas = np.array(betas)              # 种植粮食
        self.gamma = gamma                        # 种植甜菜
        self.r = r                                # 小母牛出售率
        self.M = M                                # 贷款投资
        self.m = self.calculate_m(M)              # 年度固定成本
        self.years = years                        # 模拟年数
        self.n = len(x0)                          # 种群年龄组数量
        self.L_pp = self.create_L_pp()            # 种群转移矩阵 (所有)
        self.L_p = np.vstack([self.L_pp[0, :] * 0.5, self.L_pp[1:, :]])    # 种群转移矩阵 (公牛 或 母牛)
        self.L_r = np.vstack([self.L_p[0, :] * (1 - r), self.L_p[1:, :]])  # 种群转移矩阵 (考虑小母牛出售)
        self.y_1 = np.zeros(self.n); self.y_1[0] = 1                       # 用于选择第一年龄组的向量
        self.y_12 = np.zeros(self.n); self.y_12[-1] = 1                    # 用于选择最后一年龄组的向量
        self.y_1_2 = np.zeros(self.n); self.y_1_2[0:2] = 1                 # 用于选择前两个年龄组的向量
        self.y_3_12 = np.zeros(self.n); self.y_3_12[2:] = 1                # 用于选择3-12年龄组的向量
        self.reset_metrics()                      # 重置所有指标

    def calculate_m(self, M):
        # 计算年度还款金额
        return (M * 0.15 * (1 + 0.15)**10) / ((1 + 0.15)**10 - 1)

    def create_L_pp(self):
        # 创建种群转移矩阵
        L_pp = np.zeros((self.n, self.n))
        L_pp[0, :] = self.birth_rates
        for i in range(1, self.n):
            L_pp[i, i-1] = self.survival_rates[i-1]
        return L_pp

    def reset_metrics(self):
        # 重置所有指标为初始状态
        self.alpha_values = np.array([self.alpha])
        self.betas_values = np.array([self.betas])
        self.gamma_values = np.array([self.gamma])
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
        # 模拟牧场运营
        for year in range(self.years + 1):
            x = self.xs[-1]
            self.update_metrics(x)
            if year < self.years:
                self.update_x()
        
        return self.calculate_total_profit(), np.array(self.xs)
    
    def validate(self):
        # 验证最优参数
        for year in range(self.years + 1):
            x = self.xs[-1]
            self.update_metrics(x)
            if year < self.years:
                self.update_x()
        
        return self.calculate_total_profit(), np.array(self.xs)

    def update_metrics(self, x):
        # 更新所有指标
        alpha = self.alpha
        beta1, beta2, beta3, beta4 = self.betas
        gamma = self.gamma
        
        q_beta = beta1 * 1.1 + beta2 * 0.9 + beta3 * 0.8 + beta4 * 0.6
        q_gamma = 1.5 * gamma
        l_beta = q_beta - 0.6 * np.sum(x[2:12])
        l_gamma = q_gamma - 0.7 * np.sum(x[2:12])
        
        w_beta = l_beta * 75 if l_beta > 0 else -l_beta * 90
        w_gamma = l_gamma * 58 if l_gamma > 0 else -l_gamma * 70
    
        self.alpha_values = np.append(self.alpha_values, alpha)
        self.betas_values = np.vstack([self.betas_values, [beta1, beta2, beta3, beta4]])
        self.gamma_values = np.append(self.gamma_values, gamma)
        self.q_betas_values = np.append(self.q_betas_values, q_beta)
        self.q_gammas_values = np.append(self.q_gammas_values, q_gamma)
        self.l_betas_values = np.append(self.l_betas_values, l_beta)
        self.l_gammas_values = np.append(self.l_gammas_values, l_gamma)
        self.w_betas_values = np.append(self.w_betas_values, w_beta)
        self.w_gammas_values = np.append(self.w_gammas_values, w_gamma)
    
        num_xiaogongniu_sales = np.floor(self.L_p @ x @ self.y_1).astype(int)
        num_xiaomuniu_sales = np.floor(self.L_p @ x @ self.y_1 * self.r).astype(int)
        num_damuniu_sales = np.floor(self.L_r @ x @ self.y_3_12).astype(int)
        num_laomuniu_sales = np.floor(self.L_r @ x @ self.y_12).astype(int)
    
        self.num_xiaogongniu_sales_values = np.append(self.num_xiaogongniu_sales_values, num_xiaogongniu_sales)
        self.num_xiaomuniu_sales_values = np.append(self.num_xiaomuniu_sales_values, num_xiaomuniu_sales)
        self.num_damuniu_sales_values = np.append(self.num_damuniu_sales_values, num_damuniu_sales)
        self.num_laomuniu_sales_values = np.append(self.num_laomuniu_sales_values, num_laomuniu_sales)
    
        self.w_xiaogongniu_values = np.append(self.w_xiaogongniu_values, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu_values = np.append(self.w_xiaomuniu_values, 40 * num_laomuniu_sales)
        self.w_damuniu_values = np.append(self.w_damuniu_values, 370 * num_damuniu_sales)
        self.w_laomuniu_values = np.append(self.w_laomuniu_values, 120 * num_laomuniu_sales)
    
        self.t_xiaomunius_values = np.append(self.t_xiaomunius_values, 10 * x @ self.y_1_2)
        self.t_damunius_values = np.append(self.t_damunius_values, 42 * x @ self.y_3_12)
        self.t_betas_values = np.append(self.t_betas_values, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas_values = np.append(self.t_gammas_values, 14 * gamma)
        self.t_totals_values = np.append(self.t_totals_values, 10 * x @ self.y_1_2 + 42 * x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)
    
        self.c_xiaomunius_values = np.append(self.c_xiaomunius_values, 500 * x @ self.y_1_2)
        self.c_damunius_values = np.append(self.c_damunius_values, 100 * x @ self.y_3_12)
        self.c_betas_values = np.append(self.c_betas_values, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas_values = np.append(self.c_gammas_values, 10 * gamma)
        self.c_workers_values = np.append(self.c_workers_values, 4000 if self.t_totals_values[-1] <= 5500 else 4000 + 1.2 * (self.t_totals_values[-1] - 5500))
    
        self.w_years_values = np.append(self.w_years_values, np.sum([self.w_xiaogongniu_values[-1], self.w_xiaomuniu_values[-1], self.w_damuniu_values[-1],
                                                                     self.w_laomuniu_values[-1], self.w_betas_values[-1], self.w_gammas_values[-1]]))
        
        self.c_years_values = np.append(self.c_years_values, np.sum([self.c_betas_values[-1], self.c_gammas_values[-1], self.c_xiaomunius_values[-1],
                                                                     self.c_damunius_values[-1], self.c_workers_values[-1], self.m]))
        
        self.E_years_values = np.append(self.E_years_values, self.w_years_values[-1] - self.c_years_values[-1])

    def update_x(self):
        # 更新种群分布
        x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum((np.floor(x)).astype(int), 0))
        
    def calculate_total_profit(self):
        # 计算总利润
        return np.sum(self.E_years_values)

if __name__ == '__main__':
    
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
        
        profit, xs = cattle.simulate()
        
        penalty = 0
        xs = np.array(xs)
    
        for t in range(len(xs)):
            x = xs[t]
            
            # 约束1：牛舍数量大于牛数
            constraint1 = (M / 200 + 130) - np.sum(x)
            if not (constraint1 >= 0):
                penalty += 10000
    
            # 约束2：牧草面积下限
            constraint2 = alpha - (2/3 * np.sum(x[:2]) + np.sum(x[2:]))
            if not (constraint2 >= 0):
                penalty += 10000
    
            # 约束3：总面积上限
            constraint3 = 200 - (alpha + beta1 + beta2 + beta3 + beta4 + gamma)
            if not (constraint3 >= 0):
                penalty += 10000
    
            # 约束4：成年母牛数量下限（仅在最后一年检查）
            if t == len(xs) - 1:
                constraint4 = np.sum(x[2:]) - 50
                if not (constraint4 >= 0):
                    penalty += 10000
    
            # 约束5：成年母牛数量上限（仅在最后一年检查）
            if t == len(xs) - 1:
                constraint5 = 175 - np.sum(x[2:])
                if not (constraint5 >= 0):
                    penalty += 10000
    
            # 约束6：各年龄组牛的数量为非负整数
            if not (np.all(x >= 0) and np.all(x == np.floor(x))):
                penalty += 10000
    
        # 约束7：各种植面积为非负数
        if not (alpha >= 0 and beta1 >= 0 and beta2 >= 0 and beta3 >= 0 and beta4 >= 0 and gamma >= 0):
            penalty += 10000
    
        # 约束8：小母牛出售率在0到1之间
        if not (0 <= r <= 1):
            penalty += 10000
    
        return -profit + penalty

    # 定义参数的边界
    bounds = [
        (0, 1),                     # r: 小母牛出售率
        (0, 1000000),               # M:     贷款投资
        (2/3*20+100, 200),          # alpha: 种植牧草
        (0, 20),                    # beta1: 种植粮食
        (0, 30),                    # beta2: 种植粮食
        (0, 30),                    # beta3: 种植粮食
        (0, 10),                    # beta4: 种植粮食
        (0, 200- (2/3*20+100))      # gamma: 种植甜菜
    ]
    
    # 使用差分进化算法寻找最优参数
    result = differential_evolution(objective_function, bounds)
    
    print(f"Optimal parameters: \n{result.x}")
    print(f"Maximum profit: \n{-result.fun}")

    # 使用最优参数进行最终验证
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

    # 验证最终结果
    final_profit, _ = cattle.validate()
    print(f"Yearly profits:\n{cattle.E_years_values}")
    print(f"Alpha values:\n{cattle.alpha_values}")
    print(f"Beta values:\n{cattle.betas_values}")
    print(f"Gamma values:\n{cattle.gamma_values}")
    print(f"Final profit with optimal parameters:\n{final_profit}")
