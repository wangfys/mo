import numpy as np
import json
from enum import Enum

forwardStatus = Enum("forwardStatus", ("uninitialized", "initialized", "computed"))
backwardStatus = Enum("backwardStatus", ("unforwarded", "forwarded", "computed"))

class BaseLayer(object):
    """
    This is the base class of all layers.
    """
    def __init__(self, args):
        self.name = args["name"]
        self.inNodes = [inNode for inNode in args["input"]]
        for inNode in self.inNodes:
            inNode.outNodes.append(self)
        self.outNodes = []
        self.params = []
        self.inputGradients = []
        self.paramGradients = []
        self.forwardStatus = forwardStatus.uninitialized
        self.backwardStatus = backwardStatus.unforwarded
        self.inShapes = [inNode.outShape for inNode in self.inNodes]
    
    def init(self, jsonParam=None):
        if jsonParam != None:
            for param in self.params:
                try:
                    self.__dict__[param] = np.array(jsonParam[self.name][param])
                except:
                    raise Exception("can not initialize parameter '%s' of '%s'" % (param, self.name))
        self.forwardStatus = forwardStatus.initialized
        for inNode in self.inNodes:
            if inNode.forwardStatus != forwardStatus.initialized:
                inNode.init(jsonParam)

    def forward(self, feedInput):
        if self.forwardStatus == forwardStatus.computed:
            return True
        for inNode in self.inNodes:
            if inNode.forwardStatus != forwardStatus.computed:
                inNode.forward(feedInput)
            inNode.forwardStatus = forwardStatus.computed
            inNode.backwardStatus = backwardStatus.forwarded
    
    def preBackward(self):
        for outNode in self.outNodes:
            if outNode.forwardStatus == forwardStatus.computed and outNode.backwardStatus != backwardStatus.computed:
                return True
    
    def backward(self, applyGradient):
        self.backwardStatus = backwardStatus.computed
        for inNode in self.inNodes:
            print(inNode.name)
            inNode.backward(applyGradient)
    
    def clearForward(self):
        for inNode in self.inNodes:
            if inNode.forwardStatus == forwardStatus.computed:
                inNode.clearForward()
        if self.forwardStatus == forwardStatus.uninitialized:
            raise Exception("the parameters of '%s' is uninitialized" % self.name)
        elif self.forwardStatus == forwardStatus.computed:
            self.forwardStatus = forwardStatus.initialized
            self.backwardStatus = backwardStatus.unforwarded
        for outNode in self.outNodes:
            if outNode.forwardStatus == forwardStatus.computed:
                outNode.clearForward()

    def clearBackward(self):
        for inNode in self.inNodes:
            if inNode.backwardStatus == backwardStatus.computed:
                inNode.clearBackward()
        if self.backwardStatus == backwardStatus.unforwarded:
            raise Exception("the forward computing of '%s' is unprocessed" % self.name)
        elif self.backwardStatus == backwardStatus.computed:
            self.backwardStatus = backwardStatus.forwarded
        for outNode in self.outNodes:
            if outNode.backwardStatus == backwardStatus.computed:
                outNode.clearBackward()

    def execute(self, feedInput):
        self.clearForward()
        self.forward(feedInput)
        self.forwardStatus = forwardStatus.computed
        self.backwardStatus = backwardStatus.forwarded
    
    def getAllParams(self, result=None, returnJSON=True):
        if returnJSON:
            result = {}
        if self.name not in result:
            for inNode in self.inNodes:
                inNode.getAllParams(result=result, returnJSON=False)
            if len(self.params) != 0:
                params = {}
                for param in self.params:
                    params[param] = self.__dict__[param].tolist()
                result[self.name] = params
            else:
                result[self.name] = None
            for outNode in self.outNodes:
                outNode.getAllParams(result=result, returnJSON=False)
        if returnJSON:
            return json.dumps(result)