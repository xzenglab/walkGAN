#!/usr/bin/python
# encoding=utf-8
# -*- coding:utf-8 -*

import os
import sys
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])#切换工作路径
import numpy
from numpy import *
import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print('##')
file = '../save/wiki/'
pre = 'wiki(sdne).embeddings'
fdata = file+pre

class chj_data(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

def visual_data():
    with open('../data/wiki/wiki_labels_three.txt', 'r') as f:
        lines_label = f.readlines()
        label = []
        for line in lines_label:
            line = line.strip()
            line = line.split('\t')
            line = [int(x) for x in line]
            label.append(line)
    id = ['label_0', 'label_1', 'label_2']
    c = ['r', 'b', 'g']


    target = zeros(2363,dtype = int)
#    data = zeros((2363, 64), dtype=float)
    data = np.loadtxt(file + pre)
    print(data.shape[0],data.shape[1])
    data = data[np.argsort(data[:, 0])]
    data = data[:,1:]
    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data, target)
    plt.figure(figsize=(9, 6))

    for j in range(3):
        temp=(X_tsne[label[j], :])
        plt.scatter(temp[:, 0], temp[:, 1], c=c[j], label=id[j])

    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    visual_data()

