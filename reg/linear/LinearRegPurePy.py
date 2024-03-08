from reg.AbsReg import AbsReg


class LinearRegPurePy(AbsReg):
    def get_mse(self, training_data, target_data, w, b):
        total = 0.0
        for i in range(self.n):
            pred_i = w * training_data[i] + b
            total += (pred_i - target_data[i]) ** 2
        return total / self.n

    def get_gradients(self, training_data, target_data, w, b):
        total_w = 0.0
        total_b = 0.0
        for i in range(self.n):
            pred_i = w * training_data[i] + b
            total_b += (pred_i - target_data[i])
            total_w += (pred_i - target_data[i]) * training_data[i]
        return [total_w / self.n, total_b / self.n]

    def get_pred(self, training_data, w, b):
        return w * training_data + b
