# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:21:49 2020

@author: Rohith
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

df=pd.read_csv("E:\Data Science\Assignments\Python code\Hierarchical_Clustering\\crime_data.csv")

df_new=scale(df.iloc[:,1:])

help(linkage)

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

df=df.iloc[:,[5,1,2,3,4]]

df

df.to_csv("crime_data.csv",encoding="utf-8")





