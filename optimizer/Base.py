import numpy as np
from ..globalvar import *

class BaseOptimizer():
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
            if className in UnnamedNodesCount:
                UnnamedNodesCount[className] += 1
            else:
                UnnamedNodesCount[className] = 0
            self.name = className + str(UnnamedNodesCount[className]) + "_auto"
        Nodes[self.name] = self
        self.target = args["target"]
        self.learning_rate = args["learning_rate"]
        self.inputGradients = {self.target.name:np.diag(np.ones(self.target.outSize))}
        self.computeSequence = self.target.computeSequence.copy()
        self.params = []

    def applyFunc(self):
        pass

    def calcGradients(self, feedInput=None):
        self.target.execute(feedInput)
        for name in self.computeSequence:
            Nodes[name].outNodes = []
        self.target.outNodes = [self]
        for i in range(len(self.computeSequence), 0, -1):
            name = self.computeSequence[i - 1]
            Nodes[name].calcGradient()
            for inNode in Nodes[name].inNodes:
                if not Nodes[name] in inNode.outNodes:
                    inNode.outNodes.append(Nodes[name])

    def minimize(self, feedInput=None):
        self.target.execute(feedInput)
        for name in self.computeSequence:
            Nodes[name].outNodes = []
        self.target.outNodes = [self]
        for i in range(len(self.computeSequence), 0, -1):
            name = self.computeSequence[i - 1]
            Nodes[name].backward(np.frompyfunc(self.applyFunc, 2, 1))
            for inNode in Nodes[name].inNodes:
                if not Nodes[name] in inNode.outNodes:
                    inNode.outNodes.append(Nodes[name])
