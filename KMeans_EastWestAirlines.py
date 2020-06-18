# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:37:22 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale


df=pd.read_excel("E:\\Data Science\\Assignments\\Python code\\K_Means_Clustering\\EastWestAirlines.xlsx")
df=pd.get_dummies(data=df,columns={"cc1_miles","cc2_miles","cc3_miles"},drop_first=True)
df_new=scale(df.iloc[:,1:])

#To create elbow or screew plot

k=list(range(2,15))
k
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i).fit(df_new)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(df_new.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_new.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
#Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
    
    
model=KMeans(n_clusters=5).fit(df_new)
model.labels_
df["Clust"]=pd.Series(model.labels_)
df=df.iloc[:,[19,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
