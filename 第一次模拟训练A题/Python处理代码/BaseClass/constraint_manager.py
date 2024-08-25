from cattle import Cattle
import numpy as np


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
            M=M
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
            M=M
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
            M=M
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
            M=M
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