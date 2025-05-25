%pip install nnfs

import numpy as np

inputs=[1,10,17668,175]
weights=[[1,1,14746,1],[1,1564,-1,1],[-1,1,-145646,1],[146,-1,-1,-1]]
biases=[1,1,1,1,]

output=np.dot(weights,inputs)+biases
print(output)

inputs=[[1,1,1,1],[1,1,1,1],[1,1,1,1]]

output1=np.dot(inputs,np.array(weights).T)+biases
output2=np.dot(output1,np.array(weights).T)+biases
output3=np.dot(output2,np.array(weights).T)+biases
print(output3)

class dense_layer:
  def __init__(self,n_inputs,n_neurons):
    self.weights=0.10*np.random.randn(n_inputs,n_neurons)
    self.biases=np.zeros((1,n_neurons))
  def forward(self,inputs):
    self.output=np.dot(inputs,self.weights)+self.biases

class activation_relu:
  def forward(self,inputs):
    self.output=np.maximum(0,inputs)


layer1=dense_layer(4,10)
layer2=dense_layer(10,8)
layer1.forward(inputs)
activation=activation_relu()
activation.forward(layer1.output)
layer2.forward(activation.output)
activation2=activation_relu()
activation2.forward(layer2.output)
print(activation2.output)

class activation_softmax:
  def forward(self,inputs):
    exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
    probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
    self.output=probabilities


class Loss:
  def calculate(self,output,y):
    sample_losses=self.forward(output,y)
    data_loss=np.mean(sample_losses)
    return data_loss
class Loss_CategoricalCrossEntropy(Loss):
  def forward (self,y_pred,y_true):
    samples=len(y_pred)
    y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)

    if len(y_true.shape)==1:
      correct_confidences=y_pred_clipped[range(samples),y_true]
    elif len(y_true.shape)==2:
      correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)
    negative_log_likelihood=-np.log(correct_confidences)
    return negative_log_likelihood


import nnfs
from nnfs.datasets import spiral_data
X,y=spiral_data(samples=100,classes=3)
layer1 = dense_layer(2, 10)       # Spiral data has 2 features
layer2 = dense_layer(10, 3)       # 3 output classes

layer1.forward(X)
activation1 = activation_relu()
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2 = activation_softmax()
activation2.forward(layer2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

#Normalisation in general
layer1=dense_layer(4,10)
layer2=dense_layer(10,8)
layer1.forward(inputs)
activation=np.exp(layer1.output)
activation=activation/sum(activation)
layer2.forward(activation)
activation2=np.exp(layer2.output)
activation2=activation2/sum(activation2)
print(activation2)

exp_values=np.exp(inputs)
exp_values=exp_values/np.sum(exp_values,axis=1,keepdims=True)

import math

softmax_output=[0.7,0.1,0.2]
one_hot_endcoding=[1,0,0]
loss=0
for i in range (3):
  loss+=((one_hot_endcoding[i])*math.log(softmax_output[i]))
print(-loss)

import math

softmax_output=[0.7,0.1,0.2]
one_hot_endcoding=[1,0,0]
loss=0
for i in range (3):
  loss+=((one_hot_endcoding[i])*math.log(softmax_output[i]))
print(-loss)
