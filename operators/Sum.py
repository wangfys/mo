import numpy as np
from ..layers.Base import BaseLayer

class Sum(BaseLayer):
    """
    This is the reduce sum layer which can add the elements in the tensor.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.axis = args["axis"] if "axis" in args else None
        self.outShape = np.sum(np.zeros(self.inShapes[0]), axis=self.axis).shape if not self.axis == None else np.array([1])
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        self.output = np.sum(inputTensor, axis=self.axis)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        rowNumber = self.outSize
        columnNumber = self.inSizes[0]
        thisInputGradient = np.zeros((rowNumber, columnNumber))
        for i in range(columnNumber):
            tmp = np.zeros(self.inShapes[0])
            tmp.ravel()[i] = 1
            tmp = np.sum(tmp, axis=self.axis)
            for j in np.argwhere(tmp.flatten()==1):
                thisInputGradient[j, i] = 1
        inputGradient = np.zeros((1, columnNumber))
        for outNode in self.outNodes:
            inputGradient += np.dot(outNode.inputGradients[self.name], thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)