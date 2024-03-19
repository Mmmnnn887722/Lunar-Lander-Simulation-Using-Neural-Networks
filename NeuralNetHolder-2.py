#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import predection       # import the file that contain predection function and normalization and denormalization  

from predection import predection_fun
from Main_game_code import predection
from predection import de_normalization_process
from predection import normalization_process

class NeuralNetHolder:
    def __init__(self):
        super().__init__()   

    def predict(self, input):
      #read the data files to normalize the input and output to the same data collection.
        Data=pd.read_csv('ce889_dataCollection.csv',index_col=False,header=None)
        object = predection_fun()

    #************split the input to convert to float because its object********
        z = input.split(",")  # 
        z=list(z)
        a=[]
        for i in z:
            a.append(float(i))
    #*****# Normalize the data before feed forward****************************      
    
        a=np.array(a)
        a1 = normalization_process(a[0], Data[0].max() , Data[0].min())
        a2 = normalization_process(a[1], Data[1].max() ,Data[1].min())
        c=np.array([a1,a2])

    #****# calling the feed forward function from the prdection file.**************
        
        o = object.forward_P (c)     
        
    #****** de-normalize the data before send to the game **************
        y1 = de_normalization_process(o[0],Data[2].max(),Data[2].min() )
        y2 = de_normalization_process(o[1],Data[3].max(),Data[3].min())
        print('first value after denormalization is:-',y1,"Second value after denormalization is:-",y2)
        return y1,y2  # return y1 and y2 to run the game.

#***************############# END #########################***************************************

