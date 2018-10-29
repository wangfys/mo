from .Base import BaseOptimizer

class GradientDescent(BaseOptimizer):
    """
    This is the gradient descent optimizer.
    """
    def __init__(self, **args):
        BaseOptimizer.__init__(self, args)

    def applyFunc(self, param, gradient):
        return param - self.learning_rate * gradient