class BaseOptimizer(object):
    """
    This is the base class of all optimizers.
    """
    def __init__(self, args):
        self.name = args["name"]
        self.target = args["target"]
        self.learning_rate = args["learning_rate"]
    
    def applyFunc(self):
        pass
    
    def minimize(self):
        self.target.forward()
        self.target.backward(self.applyFunc)