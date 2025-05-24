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


layer1=dense_layer(4,10)
layer2=dense_layer(10,8)
layer1.forward(inputs)
activation=np.exp(layer1.output)
activation=activation/sum(activation)
layer2.forward(activation)
activation2=np.exp(layer2.output)
activation2=activation2/sum(activation2)
print(activation2)


