from reg.NumPyReg import NumPyReg


class LinearRegNumPy(NumPyReg):
    def get_pred(self, training_data, w, b):
        return w * training_data + b

