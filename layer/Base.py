import numpy as np
import json
from functools import reduce
from ..lib import mergeComputeSequence
from ..globalvar import *

class BaseLayer():
    """
    This is the base class of all layers.
    """
    def __add__(self, node):
        from .operator import Add, Constant
        if node.__class__.__base__ == BaseLayer:
            return Add(input=[self, node])
        else:
            tmp = Constant(data=node)
            return Add(input=[self, tmp])

    def __init__(self, args, inputNum=None):
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
        if inputNum != None and len(args["input"]) != inputNum:
            raise Exception("the number of inputs for '%s' is invalid" % self.name)
        self.inNodes = [inNode for inNode in args["input"]]
        self.outNodes = []
        self.params = []
        self.paramsForInference = []
        self.inputGradients = {}
        self.paramGradients = {}
        self.inShapes = [inNode.outShape for inNode in self.inNodes]
        self.inSizes = [np.prod(inNode.outShape) for inNode in self.inNodes]
        self.fix = args["fix"] if "fix" in args else False
        self.isTrain = args["isTrain"] if "isTrain" in args else False
        self.output = None
        if len(self.inNodes) > 0:
            self.computeSequence = reduce(mergeComputeSequence, [inNode.computeSequence for inNode in self.inNodes]).copy()
            self.computeSequence.append(self.name)
        else:
            self.computeSequence = [self.name]

    def __mul__(self, node):
        from .operator import Multiply, Constant
        if node.__class__.__base__ == BaseLayer:
            return Multiply(input=[self, node])
        else:
            tmp = Constant(data=node)
            return Multiply(input=[self, tmp])

    def __neg__(self):
        from .operator import Negative
        return Negative(input=[self])

    def __radd__(self, node):
        return self.__add__(node)

    def __repr__(self):
        return "%s: %s" % (self.name, self.output.__repr__())

    def __rmul__(self, node):
        return self.__mul__(node)

    def __rsub__(self, node):
        from .operator import Add, Negative, Constant
        tmp1 = Constant(data=node)
        tmp2 = Negative(input=[self])
        return Add(input=[tmp1, tmp2])

    def __sub__(self, node):
        from .operator import Add, Negative, Constant
        if node.__class__.__base__ == BaseLayer:
            tmp = Negative(input=[node])
            return Add(input=[self, tmp])
        else:
            tmp = Constant(data=-node)
            return Add(input=[self, tmp])

    def applyGradientDescent(self, applyFunc):
        pass

    def backward(self, applyFunc):
        """
        inputGradient is a M*N matrix. Consider that the output and input are flattened. M represents the output size of this layer. N represents represents the input size of this layer. The (i,j) element of this matrix means the derivative of output_i of input_j.
        self.inputGradients is a dict. Each element of it is a M*N matrix (the index is the name of the corresponding input layer). M represents the size of final output. In fact M=1 because the goal of minimize is a single number. The element is the matrix product of the output nodes' inputGradients and inputGradient.
        """
        self.calcGradient()
        self.applyGradientDescent(applyFunc)

    def execute(self, feedInput):
        for name in self.computeSequence:
            if Config["Perf"]:
                import time
                start = time.time()
                Nodes[name].forward(feedInput)
                end = time.time()
                print(name, "forward", end-start)
            else:
                Nodes[name].forward(feedInput)

    def getAllParams(self, returnJSON=True):
        result = {}
        for name in Nodes:
            if len(Nodes[name].params) != 0:
                params = {}
                for param in Nodes[name].params:
                    params[param] = Nodes[name].__dict__[param].tolist()
                for param in Nodes[name].paramsForInference:
                    params[param] = Nodes[name].__dict__[param].tolist()
                result[name] = params
            else:
                result[name] = None
        if returnJSON:
            return json.dumps(result)
        else:
            return result

    def calcGradient(self):
        pass

    def init(self, jsonParam=None):
        pass

    def initialize(self, jsonParam=None):
        for name in self.computeSequence:
            Nodes[name].init(jsonParam)
