#!/usr/bin/env python
# coding: utf-8

# # CE889: Neural Network and Deep learning
# **Individual assignment**

# # Import the important libraries

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import gc


# # Read the CSV file for data collection after play the game

# In[3]:


data = pd.read_csv(r'ce889_dataCollection.csv')


# # Data Processing

# In[4]:


data.head()


# In[5]:


data = pd.DataFrame(np.vstack([data.columns, data]))


# In[6]:


data.reset_index(inplace = True, drop = True)


# In[7]:


data.rename({0:"X1",1:"X2",2:"Y1",3:"Y2"},axis = 1,inplace = True)


# In[8]:


data['X1']


# In[9]:


data.describe()


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# In[12]:


data.duplicated().sum()


# In[13]:


data.dtypes


# In[14]:


data.drop(0,axis = 0,inplace = True)


# In[15]:


data["X1"] = data["X1"].astype(float)
data["X2"] = data["X2"].astype(float)
data["Y1"] = data["Y1"].astype(float)
data["Y2"] = data["Y2"].astype(float)


# In[16]:


data.dtypes


# In[17]:


data.reset_index(inplace = True, drop = True)


# In[18]:


data.head(3)


# check the normal distribution of the data 

# In[19]:


data.skew()


# # Normalizing the Data

# In[20]:


tem_data_norm=data.copy(deep=True)
tem_data_norm.head(2)


# In[21]:


def normalization_process (X, max_X,min_X):
  Value = (X-min_X)/(max_X - min_X)
  return Value


# In[22]:





# In[23]:


data.head(2)


# In[24]:


tem_data_norm["X1"] = normalization_process(tem_data_norm["X1"],tem_data_norm["X1"].max(),tem_data_norm["X1"].min())
tem_data_norm["X2"] = normalization_process(tem_data_norm["X2"],tem_data_norm["X2"].max(),tem_data_norm["X2"].min())
tem_data_norm["Y1"] = normalization_process(tem_data_norm["Y1"],tem_data_norm["Y1"].max(),tem_data_norm["Y1"].min())
tem_data_norm["Y2"] = normalization_process(tem_data_norm["Y2"],tem_data_norm["Y2"].max(),tem_data_norm["Y2"].min())


# In[25]:


tem_data_norm.head(2)


# In[26]:


tem_data_denorm = tem_data_norm.copy(deep=True)


# In[28]:


tem_data_denorm.head(3)


# In[29]:


data.head(3)


# # Split the data to Train and test 

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(tem_data_norm[['X1', 'X2']], tem_data_norm[['Y1', 'Y2']],random_state=100, test_size=0.30, shuffle=True)


# In[33]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[34]:


print(' the values of input to train the model \n',X_train)
print(' \n the values of output to train the model \n',y_train)


# # Neural network stracture

# In[41]:


# *****************input to neural network X_train,y_train// X_test,y_test***************************
inputx=np.array(X_train)
input_p = np.array (X_test)
outy=np.array(y_train)
out_p= np.array(y_test)

#********************************structure of the neural network *************************************
inputlayer_neurons  = inputx.shape[1] # number of input layers neurons--> the shape for input data,2 neurons.
hiddenlayer_neurons = 20              # number of hidden layers neurons--> get the values from MATLAP
output_neurons      = outy.shape[1]   # number of output layers neurons---> the shape of output data,2 neurons

#**************** Hyper prameters get the values from MATLAP*******************************************
lr= 0.0001 # learning rate
lamda= 1 # the value of lamda
momentum=0.6
epoch = 250 # number of iteration

#************************************** class Neural network *****************************************#
class NeuralNetwork:
    def __init__(self,  epoch,inputlayer_neurons, hiddenlayer_neurons, output_neurons ):
        self.inputx = inputx
        self.outy = outy
        self.lr = lr
        self.epoch = epoch
        self.lamda= lamda
        self.momentum = momentum
# structure of neural network:
        self.inputlayer_neurons  = inputlayer_neurons
        self.hiddenlayer_neurons = hiddenlayer_neurons
        self.output_neurons      = output_neurons
# intialize the weights and bias:
        global wh,bh,wout,bout
        self.wh=np.random.uniform(size=(self.inputlayer_neurons,self.hiddenlayer_neurons))
        self.bh=np.random.uniform(size=(1,self.hiddenlayer_neurons))
        self.wout=np.random.uniform(size=(self.hiddenlayer_neurons,self.output_neurons))
        self.bout=np.random.uniform(size=(1,self.output_neurons))

#**************************Feed Forward *************************************
    def forward (self, input):       
    # from input to hidden layer output
        hidden_layer_input1 = self.weightMulitply(input,self.wh)
        hidden_layer_input1= hidden_layer_input1+self.bh
        hidden_layer_activation = self.sigActivationFunc(hidden_layer_input1)
        output_hidden_layer= hidden_layer_activation 
    # from hidden to output layer
        output_layer_input2 = self.weightMulitply(hidden_layer_activation,self.wout)
        output_layer_input2=output_layer_input2+self.bout
        output_layer_activation = self.sigActivationFunc(output_layer_input2)
        output = output_layer_activation      
        return output, output_hidden_layer 
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

#************************** Back Propogation *************************************
    def back_propgation(self,output,error,out_hidden,row,i):
        gradient_descent = self.gradient_descent(output,error)
        local_gradient = self.local_gradient(gradient_descent,out_hidden)

        self.wout = self.output_weight_updation(out_hidden, gradient_descent,row,i)         
        self.wh = self.input_weight_updation(inputx[row], gradient_descent,row,i)
        return self.wout,self.wh

# clculate The gradients (output layer and hidden layer)   
    def gradient_descent(self, output, error):    # out layer:
        gradient_descent = self.derivatives_sigmoid(output) * error
        
        return gradient_descent

    def local_gradient(self,gradient_descent,out_hidden):  #hidden layer:
        a =  np.dot(gradient_descent,self.wout.T)
        local_gradient = a * self.derivatives_sigmoid(out_hidden)
        return local_gradient

# derivative of Sigmoid Function
    def derivatives_sigmoid(self, z):
        return lamda *z * (1 - z)      
#*********************** Error calculation *****************************************
# calculate the error:   
    def error (self,output, actualOutput):
        E = actualOutput - output  
        E_sq = E**2          
        return E, E_sq

#********************** Updating the weights **************************************        
#updating weights:
    def output_weight_updation(self,output, gradient_descent,row,i):  # output weights.

        Delta_weight_out1 = np.dot(output.T, gradient_descent) * self.lr 
        Delta_weight_out2 = Delta_weight_out1 + self.momentum * Delta_weight_out1        
        if row==0 and i==0:#for first row in first epoic 
          self.wout += np.sum(Delta_weight_out1,axis=0,keepdims=True)          
        if row>0 :
          self.wout += np.sum(Delta_weight_out2,axis=0,keepdims=True)          
        return self.wout

    def input_weight_updation(self,input, local_gradient,row,i): # input weights.

        Delta_weight_inp1 = np.dot(input, local_gradient.T) * self.lr 
        Delta_weight_inp2 = Delta_weight_inp1 + self.momentum * Delta_weight_inp1
        if row == 0 and i ==0: #for first row in first epoic 
          self.wh += np.sum( Delta_weight_inp1,axis =0, keepdims=True )          
        if row > 0 :
          self.wh += np.sum( Delta_weight_inp2,axis =0, keepdims=True )                  
        return self.wh

#******************************** Tain The Model by (X_train) and Y_train) ***************************************

    def train(self,inputx, outy):
      self.MSE_list =[]
      self.nu_it_lst=[]     
      for i in range(epoch):
        self.nu_it_lst.append(i)    
        Error=[]
        Error_squared =[]
        for row in range(len(inputx)):
            output, out_hidden = self.forward(inputx[row])         # calling feed forward
            error,E_sq = self.error(output,outy[row])              # calculate the error
            Error.append(error)                                    # saving the error after each epoch
            Error_squared.append(E_sq)                             # saving the error square after epoch to calculate MSE
            self.back_propgation(output,error,out_hidden,row,i)    # calling back Propgation 

        MSE = np.mean(Error_squared)                               # calculate Mean square error
        print("Epoch--",i,"--MSE--------------", MSE)              # print MSE with every epoic               
        self.MSE_list.append(MSE)                                 # saving mean square error to plot the graph with each epoch
#********************************* validate the Model *************************************************************

#********************************* Testing the Model by (X_test,Y_test) *************************************************

    def Testing (self,input_p,out_p):         
          Error_V =[] 
          Error_sq_v= []
          self.MSE_v=[]
          for row in range(len(input_p)):
              output, out_hidden = self.forward(input_p[row])
              error , E_sq = self.error(output,out_p[row])     
              Error_V.append(error)
              Error_sq_v.append(E_sq)
              MSE = np.mean(E_sq)
              self.MSE_v.append(MSE)
          print (" |*** The Average of Mean Square error : = ", sum(self.MSE_v)/int(len(input_p)))   

#*********************************** Predection from the model****************************************

# The predection function will be in the predection files.
#*********************************** Saving Weights after training *************************************************      
    def savingweights(self):
      # updated_weights_input=list(self.wh)
      print('\ninput layers weights==== wh', self.wh)
      print('_________________________________________')
      print('\nbias at input layer=== bh', self.bh)
      print('_________________________________________')
      print('\n hidden layer weights= wout', self.wout) 
      print('_________________________________________')
      print('\n bias at hidden layer= bout ', self.bout)
     # Export the weights to csv files

      my_df = pd.DataFrame(self.wh)
      my_df.to_csv('wh.csv',header = False, index= False)

      my_df = pd.DataFrame(self.bh)
      my_df.to_csv('bh.csv',header = False, index= False)

      my_df = pd.DataFrame(self.wout)
      my_df.to_csv('wout.csv',header = False, index= False)

      my_df = pd.DataFrame(self.bout)
      my_df.to_csv('bout.csv',header = False, index= False)
      return

#****************** Plotting the graph for visulaze the MSE with every epoic ***************************************

    def plottingGraph(self):
      print(" \n ########## The graph after training #############\n ")
      plt.figure()
      plt.plot(self.nu_it_lst,self.MSE_list)   
      plt.title('Mean Sum Squared Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Mean Square Error')
      plt.show() 
#**********************************
    def plottingGraph_V(self):
      print(" \n #### The graph for Distribution of MSE at test data #########\n ")
      plt.figure()
      plt.plot(self.MSE_v)   
      plt.title('Distribution of Mean Sum Squared Loss')
      plt.xlabel('length of Test data')
      plt.ylabel('Mean Square Error')
      plt.show() 

#**************************# calling the Neural network #*************************************#
network = NeuralNetwork(epoch,inputlayer_neurons,hiddenlayer_neurons,output_neurons) # number of epoch,number of input layers,number of hidden layers neuron,number of output layer
network.train(inputx,outy)    # train the model in input and output from the data"X_train,y_train.
network.savingweights()       # saving weights after train
network.plottingGraph()       # plotting the graph for MSE with every epoch.
print( " \n *********Testing of the Model************** \n ")
network.Testing(input_p,out_p)# validate the model and get Mse
network.plottingGraph_V()     # plotting a graph for distribution of MSE on testin data   




gc.collect()


# # Predection Function 

# In[ ]:


# I create a separete python file for this function to read the weights and make predection


# # Neural Holder to integrate with the game

# In[ ]:


# a separete python fiel for integration with the game 

