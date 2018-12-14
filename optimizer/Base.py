import numpy as np
from ..lib import forwardStatus, backwardStatus
from ..globalvar import *

class BaseOptimizer(object):
    """
    This is the base class of all optimizers.
    """
    def __init__(self, args):
        if "name" in args:
            if args["name"] in Nodes:
                raise Exception("already have a node named '%s'" % args["name"])
            else:
                self.name = args["name"]
        else:
            className = self.__class__.__name__
            if className in UnnamedNodes:
                UnnamedNodes[className] += 1
            else:
                UnnamedNodes[className] = 0
            self.name = className + str(UnnamedNodes[className]) + "_auto"
        self.target = args["target"]
        self.learning_rate = args["learning_rate"]
        self.target.outNodes.append(self)
        self.forwardStatus = forwardStatus.uninitialized
        self.backwardStatus = backwardStatus.unforwarded
        self.inputGradients = {self.target.name:np.diag(np.ones(self.target.outSize))}
        if Config["imperative"]:
            self.target.init()
            self.target.execute({})
    
    def applyFunc(self):
        pass
    
    def minimize(self):
        self.target.clearBackward()
        self.target.backward(np.frompyfunc(self.applyFunc, 2, 1))
    
    def getAllParams(self, result=None, returnJSON=True):
        return None