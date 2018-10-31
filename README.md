## 1. About mo
Mo is a toy framework for the project of my Machine Learning course. According to the project requirement, no exisiting open source machine learning framework can be used to build a neural network for the given image recogintion task. Mo implicitly implement a compute graph without a graph executor because that's a little bit sophisticated for a toy.

## 2. Details of mo

### 2.1 Layers
Layers module now provide `Conv2D`, `Dense`, `Flatten`, `Input`, `LeakyReLU` and `Sigmoid` layers. You can use `execute()` method to compute the related part of the compute graph.

### 2.2 Optimizers
Optimizers module now only provide `GradientDescent` optimizer. You can use `minimize()` method to back propagate the related part of the compute graph.