import numpy as np
import json
from functools import reduce
from ..lib import mergeComputeSequence
from ..globalvar import *

class BaseLayer(object):
    """
    This is the base class of all layers.
    """
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
        for inNode in self.inNodes:
            if not self in inNode.outNodes:
                inNode.outNodes.append(self)
        self.outNodes = []
        self.params = []
        self.inputGradients = {}
        self.paramGradients = {}
        self.inShapes = [inNode.outShape for inNode in self.inNodes]
        self.inSizes = [np.prod(inNode.outShape) for inNode in self.inNodes]
        self.fix = args["fix"] if "fix" in args else False
        self.output = None
        if len(self.inNodes) > 0:
            self.computeSequence = reduce(mergeComputeSequence, [inNode.computeSequence for inNode in self.inNodes]).copy()
            self.computeSequence.append(self.name)
        else:
            self.computeSequence = [self.name]

    def __repr__(self):
        return str(self.output)

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
            Nodes[name].forward(feedInput)

    def getAllParams(self, returnJSON=True):
        result = {}
        for name in Nodes:
            if len(Nodes[name].params) != 0:
                params = {}
                print(name, Nodes[name].params)
                for param in Nodes[name].params:
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





