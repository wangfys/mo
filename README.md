## 1. About mo
Mo is a toy framework for the project of my Machine Learning course. According to the project requirement, no exisiting open source machine learning framework can be used to build a neural network for the given image recogintion task. Mo implicitly implement a compute graph without a graph executor because that's a little bit sophisticated for a toy.

## 2. Details of mo

### 2.1 Layers
Layers module now provide `Conv2D`, `Dense`, `Flatten`, `Input`, `LeakyReLU` and `Sigmoid` layers. You can use `execute()` method to compute the related part of the compute graph.

### 2.2 Optimizers
Optimizers module now only provide `GradientDescent` optimizer. You can use `minimize()` method to back propagate the related part of the compute graph.

## 3. Example Code
```python
import mo
import numpy as np
import json

a = mo.layers.Input(name="input", input=[], shape=(2, 2, 5, 5))
b = mo.layers.Conv2D(name="conv", input=[a], kernel=(3,3,3), K_init=mo.initializers.UniformRandom(-0.01, 0.01), b_init=mo.initializers.UniformRandom(-0.01, 0.01))
c = mo.layers.Flatten(name="flatten", input=[b])
d = mo.layers.Dense(name="dense", input=[c], unitNum=2, K_init=mo.initializers.UniformRandom(-0.01, 0.01), b_init=mo.initializers.UniformRandom(-0.01, 0.01))
e = mo.LeakyReLU(name="LeakyReLU", input=[d], k=0.3)
f = mo.Sigmoid(name="sigmoid", input=[e])
g = mo.Log(name="log", input=[f], epsilon=0.0001)
h = mo.Sum(name="sum", input=[g])
i = mo.optimizers.GradientDescent(name="gradient descent", target=h, learning_rate=0.001)

jsonData = '{"input": null, "conv": {"K": [[[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]]],[[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]],[[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]]]], "b": [-1.0,0.0,1.0]}, "flatten": null, "dense": {"K": [[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027],[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027]], "b": [0.001,-0.001]}, "log": null,"sum": null}'

#h.init()
h.init(json.loads(jsonData))

inputTensor = np.array([[[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]],[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]],[[[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3],[-4,-4,-4,-4,-4],[-5,-5,-5,-5,-5]],[[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5]]]])
h.execute({"input":inputTensor})
print(h.output)
print(h.getAllParams())

for _ in range(10):
    i.minimize()
    h.execute({"input":inputTensor})
    print(h.getAllParams())
    print(h.output)
```