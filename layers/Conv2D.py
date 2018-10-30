import numpy as np
from .Base import BaseLayer

class Conv2D(BaseLayer):
    """
    This is the 2D convolution layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        kernel: the shape of kernel, like (output_channels, output_height, output_width), notice that output_height should be the same as output_width
        K_init: the initializer of K, np.zeros if not set
        b_init: the initializer of b, np.zeros if not set
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.kernelSize = args["kernel"]
        self.K_init = args["K_init"] if "K_init" in args else np.zeros
        self.b_init = args["b_init"] if "b_init" in args else np.zeros
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], self.kernelSize[0], self.inShapes[0][2]-self.kernelSize[1]+1, self.inShapes[0][3]-self.kernelSize[2]+1))
        self.outSize = np.prod(self.outShape)
    
    def init(self, jsonParam=None):
        if jsonParam == None:
            self.K = self.K_init((self.kernelSize[0], self.inShapes[0][0], self.kernelSize[1], self.kernelSize[2]))
            self.b = self.b_init((self.kernelSize[0]))
        BaseLayer.init(self, jsonParam)
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        outputTensor = np.zeros(self.outShape)
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for h in range(self.outShape[2]):
                    for w in range(self.outShape[3]):
                        outputTensor[n, c, h, w] = np.sum(inputTensor[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]] * self.K[c, :]) + self.b[c]
        self.output = outputTensor
    
    def backward(self, applyGradient):
        pass