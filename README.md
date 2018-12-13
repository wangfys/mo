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
b = mo.layer.Conv2D(input=[a], kernel=(3,3,3), K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01)) # You don't need to assign a name for each layer. But don't forget to assign a name to "Input" layer because you will need to feed inputs
c = mo.layer.Flatten(input=[b])
d = mo.layer.Dense(input=[c], unitNum=2, K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
e = mo.activation.LeakyReLU(input=[d], k=0.3)
f = mo.activation.Sigmoid(input=[e])
g = mo.Log(input=[f], epsilon=0.0001)
h = mo.Mean(input=[g], axis=0)
i = mo.Sum(input=[h])
j = mo.Add(input=[i, i, i])
k = mo.optimizer.GradientDescent(target=j, learning_rate=0.001)

jsonData = '{"input": null, "Conv2D0_auto": {"K": [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]], [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]], "b": [-1.0, 0.0, 1.0]}, "Flatten0_auto": null, "Dense0_auto": {"K": [[-0.0076919259030139055, -0.007210624331558169, 0.0036341556885749783, 0.0075499088114949955, -0.009066081285552106, 0.008601194706675105, 0.0026018414605049386, 0.00040925811497360444, -0.007103225951877972, -0.007370711451927471, -0.0037572843117287683, 0.00199902293618698, -0.0010403650346203722, 0.005705727463691192, -0.0015607089287016148, 0.008662504314585208, 0.003531218006491572, -0.005640029444121004, -0.008024277938359893, 0.005167679620891688, -0.009534745066773275, 0.003543125590848864, -0.007483986142108727, -0.0035694099493973662, 0.0032114840686200633, -0.0030097061370108463, 0.0063743103598121], [-0.009963063250305989, 0.00029350471797199434, -0.003991757916636223, -0.0012107150110901247, -0.005185530705467805, 0.0029878176960746287, -0.0009635599280887214, -0.000329792647066016, -0.0006313156545209846, 0.007101715684036681, -0.005401279112744799, 0.0049990639955273895, -0.004227433363581996, 0.004461053869183711, 0.003283222823157246, 6.341362467873236e-05, 0.00878190835677089, 0.0008870679309325481, -0.006712328404702368, -0.004552033194958116, 0.005906590584462886, -0.0012815865535775114, 0.008417399102788362, 0.002753559544150885, -0.009366215135782579, 0.008197929861744642, -0.006523668633838249]], "b": [-0.005432726885472872, -8.961439883307075e-05]}, "LeakyReLU0_auto": null, "Sigmoid0_auto": null, "Log0_auto": null, "Mean0_auto": null, "Sum0_auto": null, "Add0_auto": null}'

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
b = mo.layer.Conv2D(input=[a], kernel=(3,3,3), thisParam={"K": [[[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]]],[[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]],[[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]]]], "b": [-1.0,0.0,1.0]}) # Using "imperative" style, you can provide params of this layer via "thisParam" arguments
c = mo.layer.Flatten(input=[b])
d = mo.layer.Dense(input=[c], unitNum=2, K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
e = mo.activation.LeakyReLU(input=[d], k=0.3)
f = mo.activation.Sigmoid(input=[e])
g = mo.Log(input=[f], epsilon=0.0001)
h = mo.Mean(input=[g], axis=0)
i = mo.Sum(input=[h])
j = mo.Add(input=[i, i, i])
k = mo.optimizer.GradientDescent(target=j, learning_rate=0.001)

jsonData = '{"input": null, "Conv2D0_auto": {"K": [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]], [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]], "b": [-1.0, 0.0, 1.0]}, "Flatten0_auto": null, "Dense0_auto": {"K": [[-0.0076919259030139055, -0.007210624331558169, 0.0036341556885749783, 0.0075499088114949955, -0.009066081285552106, 0.008601194706675105, 0.0026018414605049386, 0.00040925811497360444, -0.007103225951877972, -0.007370711451927471, -0.0037572843117287683, 0.00199902293618698, -0.0010403650346203722, 0.005705727463691192, -0.0015607089287016148, 0.008662504314585208, 0.003531218006491572, -0.005640029444121004, -0.008024277938359893, 0.005167679620891688, -0.009534745066773275, 0.003543125590848864, -0.007483986142108727, -0.0035694099493973662, 0.0032114840686200633, -0.0030097061370108463, 0.0063743103598121], [-0.009963063250305989, 0.00029350471797199434, -0.003991757916636223, -0.0012107150110901247, -0.005185530705467805, 0.0029878176960746287, -0.0009635599280887214, -0.000329792647066016, -0.0006313156545209846, 0.007101715684036681, -0.005401279112744799, 0.0049990639955273895, -0.004227433363581996, 0.004461053869183711, 0.003283222823157246, 6.341362467873236e-05, 0.00878190835677089, 0.0008870679309325481, -0.006712328404702368, -0.004552033194958116, 0.005906590584462886, -0.0012815865535775114, 0.008417399102788362, 0.002753559544150885, -0.009366215135782579, 0.008197929861744642, -0.006523668633838249]], "b": [-0.005432726885472872, -8.961439883307075e-05]}, "LeakyReLU0_auto": null, "Sigmoid0_auto": null, "Log0_auto": null, "Mean0_auto": null, "Sum0_auto": null, "Add0_auto": null}'

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