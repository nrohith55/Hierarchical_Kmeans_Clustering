# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:05:08 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\K_Means_Clustering\\Universities.csv")
df.head()
df_norm=scale(df.iloc[:,1:])
help(KMeans)

model=KMeans(n_clusters=5).fit(df_norm)
model.labels_
df["clust"]=model.labels_
df=df.iloc[:,[7,1,2,3,4,5,6]]
df.iloc[:,1:].groupby(df.clust).mean()


df_norm = norm_func(Univ.iloc[:,1:])


df_norm.head(10)  # Top 10 rows
    
#To plot elbow curve and screw plot    
    
k=list(range(2,15))
k
TWSS=[]#Variable to store total within sum of squares
for i in k:
    kmeans=KMeans(n_clusters=i).fit(df_norm)
    WSS=[]#variables for storingh within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
    
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)











    



































