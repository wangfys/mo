import numpy as np
from ..lib import forwardStatus, backwardStatus

class BaseOptimizer(object):
    """
    This is the base class of all optimizers.
    """
    def __init__(self, args):
        self.name = args["name"]
        self.target = args["target"]
        self.learning_rate = args["learning_rate"]
        self.target.outNodes.append(self)
        self.forwardStatus = forwardStatus.uninitialized
        self.backwardStatus = backwardStatus.unforwarded
        self.inputGradients = {self.target.name:np.ones(self.target.outShape)}
    
    def applyFunc(self):
        pass
    
    def minimize(self):
        self.target.clearBackward()
        self.target.backward(np.frompyfunc(self.applyFunc, 2, 1))
    
    def getAllParams(self, result=None, returnJSON=True):
        return None