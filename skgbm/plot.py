#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:09:32 2023

@author: krzysztof
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree

# https://mljar.com/blog/visualize-decision-tree/

def plot_tree(tree):
    dtc = DecisionTreeClassifier()
    dtc.tree_ = dt.tree_
    plot_tree(dtc)
    
