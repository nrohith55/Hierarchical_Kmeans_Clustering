# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:57:09 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale 

df=pd.read_excel("E:\Data Science\Assignments\Python code\Hierarchical_Clustering\\EastWestAirlines.xlsx")
df.isnull().sum()
df_new=scale(df.iloc[:,1:])

z=linkage(df_new,method='complete',metric='euclidean')

plt.figure(figsize=(15,5));
plt.title("Hierarchical Clustering Dendogram");
plt.xlabel("Index");
plt.ylabel("Distance");
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=8);

from sklearn.cluster import AgglomerativeClustering

h_clust=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete').fit(df_new)

cluster_labels=pd.Series(h_clust.labels_)

df["clust"]=cluster_labels

df=df.iloc[:,[12,1,2,3,4,5,6,7,8,9,10,11]]

df

df.to_csv("crime_data.csv",encoding="utf-8")
