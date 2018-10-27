import numpy as np
import json
from enum import Enum

layerStatus = Enum("layerStatus", ("uninitialized", "ready", "finished"))

class BaseLayer(object):
    """
    This is the base class of all layers.
    """
    def __init__(self, args):
        self.name = args["name"]
        self.inNodes = [args["input"]]
        for inNode in self.inNodes:
            inNode.outNodes.append(self)
        self.outNodes = []
        self.params = []
        self.status = layerStatus.uninitialized
        self.inShapes = [inNode.outShape for inNode in self.inNodes]
    
    def init(self, jsonParam=None):
        if jsonParam != None:
            for param in self.params:
                try:
                    self.__dict__[param] = np.array(jsonParam[self.name][param])
                except:
                    raise Exception("can not initialize parameter '%s' of '%s'" % (param, self.name))
        self.status = layerStatus.ready
        for inNode in self.inNodes:
            if inNode.status != layerStatus.ready:
                inNode.init(jsonParam)

    def forward(self, feedInput):
        for inNode in self.inNodes:
            if inNode.status != layerStatus.finished:
                inNode.forward(feedInput)
            inNode.status = layerStatus.finished
    
    def backward(self):
        pass
    
    def clearForward(self):
        if self.status == layerStatus.uninitialized:
            raise Exception("the parameters of '%s' is uninitialized" % self.name)
        elif self.status == layerStatus.finished:
            self.status = layerStatus.ready
        for inNode in self.inNodes:
            if inNode.status == layerStatus.finished:
                inNode.clearForward()

    def execute(self, feedInput):
        self.clearForward()
        self.forward(feedInput)
        self.status = layerStatus.finished
    
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