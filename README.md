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

Mo provides `mo.activation.LeakyRelu`, `mo.activation.Sigmoid`, `mo.activation.Softmax` for activation layers.

#### 2.2.2 Loss

Mo provides `mo.loss.CrossEntropy` for losses.

#### 2.2.3 Operator

Mo provides `mo.Add`, `mo.Constant`, `mo.Log`, `mo.Mean`, `mo.Multiply`, `mo.Sum` for operators. They act like the corresponding methods in `numpy`. `+` and `*` have been overloaded so you can use them directly instead of using `mo.Add` and `mo.Multiply`.

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
j = i + i
# j = mo.Add(input=[i, i])
# j = 2 * i
k = mo.optimizer.GradientDescent(target=j, learning_rate=0.001)

jsonData = '{"input": null, "Conv2D0_auto": {"K": [[[[0.004862449857932575, 0.006767899904416843, -0.0008096887109488807], [0.00959381660611047, -0.009503178321175325, -0.005147067204972779], [0.003751546440222433, -0.001258794461824227, -0.0033188522522613504]], [[0.0037555331639702663, 0.002858983587688848, 0.002693469409711817], [-0.002221391740012608, 0.005951732951620875, 0.00883602234331222], [0.005971710562975645, 0.004306089184247963, -0.0020676836904856485]]], [[[-0.002881099824215641, -0.003458728467275316, 0.0018138999151051348], [0.009195137392865019, -0.0022480246109310544, 0.003088267287227083], [-0.003260075420130875, -0.005324158798465082, 0.008976171196331898]], [[0.007112715314455132, 0.004404906286747352, -0.0003720591888250954], [-0.006239684102508298, 0.0081988187326069, 0.0052455123935790555], [0.006746532489813011, -0.0025415325370410627, 0.007561774461689019]]], [[[0.003235182576530573, 0.005877781199907549, 0.0014844991138454281], [0.008606093760165271, 0.00931391029647828, 0.0012409449761303579], [0.008013827130536699, -0.0050276077579992865, -0.008117579445945882]], [[0.0064614413430345765, 0.005508870390871062, -0.00820191988738148], [0.009570890321480193, 0.006253146342155845, 0.0025108526287341646], [0.007618030470186001, -0.007373876904255561, 0.0016835536282627343]]]], "b": [0.008633211800035028, 0.002381862577269935, 0.00534638283929372]}, "Flatten0_auto": null, "Dense0_auto": {"K": [[-0.0005044496860177224, -0.0029054984139039817, -0.009516873342844656, 0.006561434420591377, -0.006225171808667846, 0.0038713488101366073, -0.0031995767293507968, -0.0003059625818624609, -0.0010222942540358028, 0.006288344739917047, -0.007678831479109945, -0.00963701525087554, 0.0026721026622479396, -0.0028639222276129316, 0.005253465898465939, -0.005588143802521321, -0.004241817480336918, -0.006693941883635577, 0.0047140847338763454, 0.003764056774782247, 0.00982509388489586, -0.0016492250574600114, 0.006247663334994524, 0.007042815057581217, 0.008746587494459142, 0.005664339564981477, -0.005706852870822874], [0.006528488390578803, 0.009534092551068156, 0.009952212619741765, -0.00891490326368511, 0.00512965585607518, -0.008733307107715716, -1.6023267815801212e-05, 0.00874970902408607, -0.0025468394043906793, 0.0027024436147346093, -0.003935273145925448, -0.0058529831895254205, -0.007134829511586007, -0.0039687978082817476, 0.0020143654491023514, 0.00813058391603116, -0.005715389611138095, -0.007271019203500628, 0.00044696638514347097, -0.002548268938375984, -0.004383063712728617, -0.00042988640023883945, -0.007817601705274378, 0.002977932032886498, 0.002267502075605091, -0.0006683774535776333, -0.0019508402302870834]], "b": [0.0020875179496451617, -0.009215511088712913]}, "LeakyReLU0_auto": null, "Sigmoid0_auto": null, "Log0_auto": null, "Mean0_auto": null, "Sum0_auto": null, "Add0_auto": null, "GradientDescent0_auto": null}'

j.initialize()
# You can use a json string to initialize the network
# j.initialize(json.loads(jsonData))

j.execute({"input":inputTensor})
print(j.output)
print(j.getAllParams(returnJSON=False)) # If you don't set returnJSON=False, getAllParams() will automatically return a JSON string. You can directly save it to a file.

for _ in range(10):
    k.minimize({"input":inputTensor})

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
b = mo.layer.Conv2D(input=[a], kernel=(3,3,3), thisParam={"K": np.random.uniform(-0.01, 0.01, (3,2,3,3)), "b": np.random.uniform(-0.01, 0.01, (3,))}) # Using "imperative" style, you can provide params of this layer via "thisParam" arguments
c = mo.layer.Flatten(input=[b])
d = mo.layer.Dense(input=[c], unitNum=2, K_init=mo.initializer.UniformRandom(-0.01, 0.01), b_init=mo.initializer.UniformRandom(-0.01, 0.01))
e = mo.activation.LeakyReLU(input=[d], k=0.3)
f = mo.activation.Sigmoid(input=[e])
g = mo.Log(input=[f], epsilon=0.0001)
h = mo.Mean(input=[g], axis=0)
i = mo.Sum(input=[h])
j = mo.Add(input=[i, i])
k = mo.optimizer.GradientDescent(target=j, learning_rate=0.001)

jsonData = '{"input": null, "Conv2D0_auto": {"K": [[[[0.004862449857932575, 0.006767899904416843, -0.0008096887109488807], [0.00959381660611047, -0.009503178321175325, -0.005147067204972779], [0.003751546440222433, -0.001258794461824227, -0.0033188522522613504]], [[0.0037555331639702663, 0.002858983587688848, 0.002693469409711817], [-0.002221391740012608, 0.005951732951620875, 0.00883602234331222], [0.005971710562975645, 0.004306089184247963, -0.0020676836904856485]]], [[[-0.002881099824215641, -0.003458728467275316, 0.0018138999151051348], [0.009195137392865019, -0.0022480246109310544, 0.003088267287227083], [-0.003260075420130875, -0.005324158798465082, 0.008976171196331898]], [[0.007112715314455132, 0.004404906286747352, -0.0003720591888250954], [-0.006239684102508298, 0.0081988187326069, 0.0052455123935790555], [0.006746532489813011, -0.0025415325370410627, 0.007561774461689019]]], [[[0.003235182576530573, 0.005877781199907549, 0.0014844991138454281], [0.008606093760165271, 0.00931391029647828, 0.0012409449761303579], [0.008013827130536699, -0.0050276077579992865, -0.008117579445945882]], [[0.0064614413430345765, 0.005508870390871062, -0.00820191988738148], [0.009570890321480193, 0.006253146342155845, 0.0025108526287341646], [0.007618030470186001, -0.007373876904255561, 0.0016835536282627343]]]], "b": [0.008633211800035028, 0.002381862577269935, 0.00534638283929372]}, "Flatten0_auto": null, "Dense0_auto": {"K": [[-0.0005044496860177224, -0.0029054984139039817, -0.009516873342844656, 0.006561434420591377, -0.006225171808667846, 0.0038713488101366073, -0.0031995767293507968, -0.0003059625818624609, -0.0010222942540358028, 0.006288344739917047, -0.007678831479109945, -0.00963701525087554, 0.0026721026622479396, -0.0028639222276129316, 0.005253465898465939, -0.005588143802521321, -0.004241817480336918, -0.006693941883635577, 0.0047140847338763454, 0.003764056774782247, 0.00982509388489586, -0.0016492250574600114, 0.006247663334994524, 0.007042815057581217, 0.008746587494459142, 0.005664339564981477, -0.005706852870822874], [0.006528488390578803, 0.009534092551068156, 0.009952212619741765, -0.00891490326368511, 0.00512965585607518, -0.008733307107715716, -1.6023267815801212e-05, 0.00874970902408607, -0.0025468394043906793, 0.0027024436147346093, -0.003935273145925448, -0.0058529831895254205, -0.007134829511586007, -0.0039687978082817476, 0.0020143654491023514, 0.00813058391603116, -0.005715389611138095, -0.007271019203500628, 0.00044696638514347097, -0.002548268938375984, -0.004383063712728617, -0.00042988640023883945, -0.007817601705274378, 0.002977932032886498, 0.002267502075605091, -0.0006683774535776333, -0.0019508402302870834]], "b": [0.0020875179496451617, -0.009215511088712913]}, "LeakyReLU0_auto": null, "Sigmoid0_auto": null, "Log0_auto": null, "Mean0_auto": null, "Sum0_auto": null, "Add0_auto": null, "GradientDescent0_auto": null}'

# You can use a json string to initialize the network
# j.initialize(json.loads(jsonData))

print(j.output)
print(j.getAllParams())

for _ in range(10):
    k.minimize({"input":inputTensor})

j.execute({"input": inputTensor})
print(j.getAllParams())
print(j.output)
```
