## 1. About mo
Mo is a toy framework for the project of my Machine Learning course. According to the project requirement, no exisiting open source machine learning framework can be used to build a neural network for the given image recogintion task. Mo implicitly implement a compute graph without a graph executor because that's a little bit sophisticated for a toy. You can use both "symbolic" style and "imperative" style (in fact, Mo just execute the compute graph at once) to build a network. You can see there is almost no differences between "symbolic" style and "imperative" style in the examples.

## 2. Details of mo

### 2.1 Initializer

Mo provides `mo.initializer.Constant` and `mo.initializer.UniformRandom` initializers.

### 2.2 Layer

For any layers in `mo.layers`, `mo.activation`, `mo.loss` and any operators, you can use `execute(feedInput)` method to execute the compute graph.

#### 2.2.0 Basic

Mo provides `mo.layers.Conv2D`, `mo.layers.Dense`, `mo.layers.Flatten`, `mo.layers.Input` layers. 

#### 2.2.1 Activation

Mo provides `mo.activation.LeakyRelu`, `mo.activation.Sigmoid` for activation layers.

#### 2.2.2 Loss

Mo provides `mo.loss.CrossEntropy` for losses.

#### 2.2.3 Operator

Mo provides `mo.Add`, `mo.Log`, `mo.Mean`, `mo.Sum` for operators. They act like the corresponding methods in `numpy`.

### 2.3 Optimizer

Mo provides `mo.optimizers.GradientDescent` optimizer. You can use `minimize()` method to back propagate the related part of the compute graph.

## 3. Example Code
If you want to use the "symbolic" style, here is the example.
```python
import mo
import numpy as np
import json

inputTensor = np.array([[[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]],[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]],[[[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3],[-4,-4,-4,-4,-4],[-5,-5,-5,-5,-5]],[[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5]]]])

a = mo.layer.Input(name="input", input=[], shape=(2, 2, 5, 5))
b = mo.layer.Conv2D(name="conv", input=[a], kernel=(3,3,3), K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
c = mo.layer.Flatten(name="flatten", input=[b])
d = mo.layer.Dense(name="dense", input=[c], unitNum=2, K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
e = mo.activation.LeakyReLU(name="LeakyReLU", input=[d], k=0.3)
f = mo.activation.Sigmoid(name="sigmoid", input=[e])
g = mo.Log(name="log", input=[f], epsilon=0.0001)
h = mo.Mean(name="mean", input=[g], axis=0)
i = mo.Sum(name="sum", input=[h])
j = mo.Add(name="add", input=[i, i, i])
k = mo.optimizer.GradientDescent(name="gradient descent", target=j, learning_rate=0.001)

jsonData = '{"input": null, "conv": {"K": [[[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]]],[[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]],[[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]]]], "b": [-1.0,0.0,1.0]}, "flatten": null, "dense": {"K": [[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027],[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027]], "b": [0.001,-0.001]}, "log": null, "mean": null, "sum": null, "add": null}'

j.init()
# You can use a json string to initialize the network
# j.init(json.loads(jsonData))

j.execute({"input":inputTensor})
print(j.output)
print(j.getAllParams())

for _ in range(1):
    k.minimize()
    j.execute({"input":inputTensor})
    print(j.getAllParams())
    print(j.output)
```

If you want to use the "imperative" style, here is the example.
```python
import mo
import numpy as np
import json

mo.Config["imperative"] = True # Set mo.Config["imperative"]=True to use "imperative" style.

inputTensor = np.array([[[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]],[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]],[[[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3],[-4,-4,-4,-4,-4],[-5,-5,-5,-5,-5]],[[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5],[-1,-2,-3,-4,-5]]]])

a = mo.layer.Input(name="input", input=[], shape=(2, 2, 5, 5), data=inputTensor) # Using "imperative" style, don't forget to provide data at once
print(a) # You can use a print() to get the result at once
b = mo.layer.Conv2D(name="conv", input=[a], kernel=(3,3,3), thisParam={"K": [[[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]]],[[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]],[[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]]]], "b": [-1.0,0.0,1.0]}) # Using "imperative" style, you can provide params of this layer via "thisParam" arguments
c = mo.layer.Flatten(name="flatten", input=[b])
d = mo.layer.Dense(name="dense", input=[c], unitNum=2, K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
e = mo.activation.LeakyReLU(name="LeakyReLU", input=[d], k=0.3)
f = mo.activation.Sigmoid(name="sigmoid", input=[e])
g = mo.Log(name="log", input=[f], epsilon=0.0001)
h = mo.Mean(name="mean", input=[g], axis=0)
i = mo.Sum(name="sum", input=[h])
j = mo.Add(name="add", input=[i, i, i])
k = mo.optimizer.GradientDescent(name="gradient descent", target=j, learning_rate=0.001)

jsonData = '{"input": null, "conv": {"K": [[[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]]],[[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]],[[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]]]], "b": [-1.0,0.0,1.0]}, "flatten": null, "dense": {"K": [[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027],[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027]], "b": [0.001,-0.001]}, "log": null, "mean": null, "sum": null, "add": null}'

# You can use a json string to initialize the network
# j.init(json.loads(jsonData))

print(j.output)
print(j.getAllParams())

for _ in range(1):
    k.minimize()

j.execute({"input": inputTensor})
print(j.getAllParams())
print(j.output)
```