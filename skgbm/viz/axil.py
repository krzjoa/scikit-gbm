#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:36:18 2023

@author: krzysztof
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

# https://github.com/pgeertsema/AXIL_paper/blob/main/wdi.py
def plot_axil_heatmap(axil_weights):
    ax = sns.heatmap(axil_weights, xticklabels=1, yticklabels=1, cmap="Blues", cbar=False)
    ax.yaxis.tick_right()
    ax.set(ylabel=None)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.yticks(rotation=0)
    ax.figure.tight_layout()
    

def plot_axil_clustermap(axil_weights):
    linkage = hc.linkage(axil_weights, method='average')
    ax = sns.clustermap(axil_weights, row_linkage=linkage, col_linkage=linkage, 
                        xticklabels=1, yticklabels=1, cmap="Blues", cbar_pos=None)
    ax.figure.tight_layout()
    ax.ax_heatmap.set_ylabel("")
    

def plot_axil_network(axil_weights, y_test, labels):
    graph = axil_weights.copy()
    np.fill_diagonal(graph, 0)
    G = nx.from_numpy_matrix(graph)
    nx.draw_circular(G, with_labels=True, labels=labels, font_weight='light', 
                     node_color = y_test.tolist(), cmap="Reds", edge_color="gray")
    


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMRegressor 
    
    from skgbm.xai import AXIL
    import numpy as np
    
    X, y = make_regression(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    lgb_regressor = LGBMRegressor()
    lgb_regressor.fit(X_train, y_train)
    
    axil = AXIL(lgb_regressor)
    axil.fit(X_train, y_train)
     
    k_test = axil.transform(X_test)
    k_train = axil.transform(X_train)
    
    # Plots
    plot_axil_heatmap(k_test)
    plot_axil_clustermap(k_test)
    plot_axil_network(k_train, y_train, y_train.tolist())
