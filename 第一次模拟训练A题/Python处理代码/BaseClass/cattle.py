import numpy as np


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
        self.m = self.calculate_m(M)                   # 调用方法计算 m
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
        # 初始化指标
        self.alpha_values = np.array([0])             # alpha指标
        self.betas_values = np.array([[0, 0, 0, 0]])  # beta指标
        self.gammas_values = np.array([0])            # gamma指标
        self.q_betas_values = np.array([0])           # q_beta指标
        self.q_gammas_values = np.array([0])          # q_gamma指标
        self.l_betas_values = np.array([0])           # l_beta指标
        self.l_gammas_values = np.array([0])          # l_gamma指标
        self.w_betas_values = np.array([0])           # w_beta指标
        self.w_gammas_values = np.array([0])          # w_gamma指标
        self.c_xiaomunius_values = np.array([0])      # 小母牛成本
        self.c_damunius_values = np.array([0])        # 大母牛成本
        self.c_betas_values = np.array([0])           # beta成本
        self.c_gammas_values = np.array([0])          # gamma成本
        self.t_xiaomunius_values = np.array([0])      # 小母牛总数
        self.t_damunius_values = np.array([0])        # 大母牛总数
        self.t_betas_values = np.array([0])           # beta总数
        self.t_gammas_values = np.array([0])          # gamma总数
        self.t_totals_values = np.array([0])          # 总数
        self.c_workers_values = np.array([0])         # 工人成本
        self.num_xiaogongniu_sales_values = np.array([0])     # 小公牛销售数量
        self.num_xiaomuniu_sales_values = np.array([0])       # 小母牛销售数量
        self.num_damuniu_sales_values = np.array([0])         # 大母牛销售数量
        self.num_laomuniu_sales_values = np.array([0])        # 老母牛销售数量
        self.w_xiaogongniu_values = np.array([0])             # 小公牛收入
        self.w_xiaomuniu_values = np.array([0])               # 小母牛收入
        self.w_damuniu_values = np.array([0])                 # 大母牛收入
        self.w_laomuniu_values = np.array([0])                # 老母牛收入
        self.w_years_values = np.array([0])                   # 年收入
        self.c_years_values = np.array([0])                   # 年成本
        self.E_years_values = np.array([0])                   # 年利润

    def simulate(self):
        for year in range(self.years):
            x = self.xs[-1]                            # 当前年份的种群数量分布向量
            self.update_metrics(x)                     # 更新指标
            self.update_x()                            # 更新种群数量分布向量
        profit = self.calculate_total_profit()         # 计算总利润
        return profit, np.array(self.xs)               # 返回总利润和种群数量分布向量数组

    def validate(self):
        # 验证方法，不考虑惩罚
        for year in range(self.years + 1):
            x = self.xs[-1]
            self.update_metrics(x)
            self.update_x()
        
        return self.calculate_total_profit()
    
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
        self.alpha_values = np.append(self.alpha_values, alpha)
        self.betas_values = np.vstack([self.betas_values, [beta1, beta2, beta3, beta4]])
        self.gammas_values = np.append(self.gammas_values, gamma)
        self.q_betas_values = np.append(self.q_betas_values, q_beta)
        self.q_gammas_values = np.append(self.q_gammas_values, q_gamma)
        self.l_betas_values = np.append(self.l_betas_values, l_beta)
        self.l_gammas_values = np.append(self.l_gammas_values, l_gamma)
        self.w_betas_values = np.append(self.w_betas_values, w_beta)
        self.w_gammas_values = np.append(self.w_gammas_values, w_gamma)

        # 销售数量计算
        num_xiaogongniu_sales = self.L_p @ x @ self.y_1
        num_xiaomuniu_sales = self.L_p @ x @ self.y_1 * self.r
        num_damuniu_sales = self.L_r @ x @ self.y_3_12
        num_laomuniu_sales = self.L_r @ x @ self.y_12

        # 更新销售数量
        self.num_xiaogongniu_sales_values = np.append(self.num_xiaogongniu_sales_values, num_xiaogongniu_sales)
        self.num_xiaomuniu_sales_values = np.append(self.num_xiaomuniu_sales_values, num_xiaomuniu_sales)
        self.num_damuniu_sales_values = np.append(self.num_damuniu_sales_values, num_damuniu_sales)
        self.num_laomuniu_sales_values = np.append(self.num_laomuniu_sales_values, num_laomuniu_sales)

        # 收入计算
        self.w_xiaogongniu_values = np.append(self.w_xiaogongniu_values, 30 * num_xiaogongniu_sales)
        self.w_xiaomuniu_values = np.append(self.w_xiaomuniu_values, 40 * num_laomuniu_sales)
        self.w_damuniu_values = np.append(self.w_damuniu_values, 370 * num_damuniu_sales)
        self.w_laomuniu_values = np.append(self.w_laomuniu_values, 120 * num_laomuniu_sales)

        # 总数和成本计算
        self.t_xiaomunius_values = np.append(self.t_xiaomunius_values, x @ self.y_1_2)
        self.t_damunius_values = np.append(self.t_damunius_values, x @ self.y_3_12)
        self.t_betas_values = np.append(self.t_betas_values, 4 * (beta1 + beta2 + beta3 + beta4))
        self.t_gammas_values = np.append(self.t_gammas_values, 14 * gamma)
        self.t_totals_values = np.append(self.t_totals_values, x @ self.y_1_2 + x @ self.y_3_12 + 4 * (beta1 + beta2 + beta3 + beta4) + 14 * gamma)

        self.c_xiaomunius_values = np.append(self.c_xiaomunius_values, 500 * x @ self.y_1_2)
        self.c_damunius_values = np.append(self.c_damunius_values, 100 * x @ self.y_3_12)
        self.c_betas_values = np.append(self.c_betas_values, 15 * (beta1 + beta2 + beta3 + beta4))
        self.c_gammas_values = np.append(self.c_gammas_values, 10 * gamma)
        self.c_workers_values = np.append(self.c_workers_values, 4000 if self.t_totals_values[-1] <= 5500 else 4000 + self.t_totals_values[-1] * 1.2)

        # 年收入和年成本计算
        w_year = (self.w_xiaogongniu_values[-1] + self.w_xiaomuniu_values[-1] + self.w_damuniu_values[-1] + self.w_laomuniu_values[-1] + self.w_betas_values[-1] + self.w_gammas_values[-1])
        self.w_years_values = np.append(self.w_years_values, w_year)

        m_value = self.calculate_m()  # 计算m的值
        c_year = (self.c_betas_values[-1] + self.c_gammas_values[-1] + self.c_xiaomunius_values[-1] + self.c_damunius_values[-1] + self.c_workers_values[-1] + m_value)
        self.c_years_values = np.append(self.c_years_values, c_year)

        E_year = w_year - c_year
        self.E_years_values = np.append(self.E_years_values, E_year)

    def update_x(self):
        x = self.L_r @ self.xs[-1]
        self.xs.append(np.maximum(np.floor(x), 0))  # 取整并确保非负

    def calculate_total_profit(self):
        return np.sum(self.E_years_values)
