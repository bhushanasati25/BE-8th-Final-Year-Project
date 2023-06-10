# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graph(c):
    
    har=[0,0,0,0,0,0]
    for i in c:
        if i=='Happiness':
            har[0]+=1
        elif i=='Anger':
            har[1]+=1
        elif i=='Neutral':
            har[5]+=1
        elif i=='Fear':
            har[2]+=1
        elif i=='Sadness':
            har[3]+=1
        elif i=='Disgust':
            har[4]+=1
    s=float(sum(har))
    for i in range(len(har)):
        har[i]=har[i]/s
    return har


IIT = pd.read_csv("FINAL_AIIMS_SENTIMENTS.csv")

IIT = graph(IIT['Sentiment'].values)

Index=['Happiness','Anger','Fear','Sadness','Disgust','Neutral']
gr={'IIT':IIT}
qw=pd.DataFrame(gr,index=Index)
qw.plot(y=["IIT"],kind="bar")
plt.show()

        
    
    
    
        
    
                
                    
                
                
                

    
        
        
    
            
    