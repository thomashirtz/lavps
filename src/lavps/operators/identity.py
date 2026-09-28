from lavps.operators.base import H_functions


class Identity(H_functions):

    def __init__(self):
        super(Identity).__init__()

    def H(self, x):
        return x.reshape(x.shape[0], -1)

    def H_pinv(self, x):
        return x
