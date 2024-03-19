#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
# ********************************* The prdection Function ********************************
def normalization_process (X, max_X,min_X):    # normalization function 
  Value = (X-min_X)/(max_X - min_X)
  return Value

def de_normalization_process(X,max_X,min_X):   # denormalization function 
  a = max_X-min_X
  b = X
  Value = min_X +( b * a )
  return Value

import numpy as np
class predection_fun():
      def __init__(self): 
        #************************* The weights after train the model ******************************************      
        self.lamda= 0.5
        self.wh = np.array(pd.read_csv('wh.csv',header = None))
        self.bh = np.array(pd.read_csv('bh.csv',header = None))
        self.wout = np.array(pd.read_csv('wout.csv',header = None))
        self.bout = np.array(pd.read_csv('bout.csv',header = None))

# ****************************** Feed forward ****************************************
      def forward_P (self, input):       
    # from input to hidden layer output
        hidden_layer_input1 = self.weightMulitply(input,self.wh)
        hidden_layer_input1= hidden_layer_input1
        hidden_layer_activation = self.sigActivationFunc(hidden_layer_input1)
        output_hidden_layer= hidden_layer_activation 
    # from hidden to output layer
        output_layer_input2 = self.weightMulitply(hidden_layer_activation,self.wout)
        output_layer_input2=output_layer_input2
        output_layer_activation = self.sigActivationFunc(output_layer_input2)
        output = output_layer_activation      
        return output 
# weight multiplication 
      def weightMulitply(self,input,weight):
        V=np.dot(input,weight)     
        return V   

# activation function 
      def sigActivationFunc(self,V):
        activation=self.sigmoid(V)
        return activation

# defining the Sigmoid Function
      def sigmoid (self, z):
          return 1/(1 + np.exp(-self.lamda*z))
        
pp=predection_fun()

