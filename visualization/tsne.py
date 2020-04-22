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
pre = 'wiki(5_10_deepwalk).embeddings'
#file = '../../node2vec-master/emb/'
#pre = 'wiki(node2vec).emb'
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
    fin = open(fdata, "r")
    first_line = fin.readline()
    N = int(first_line.split(' ')[0])
    E = int(first_line.split(' ')[1])
    print('N {} E {}'.format(N, E))
    # get target -------------------
    ##target = numpy.loadtxt(ftarget, dtype=int32)
    target = zeros(N,dtype = int)
    # data = scio.loadmat(fdata)['rep']

    data = zeros((N, E), dtype=float)
    lines = fin.readlines()
    A_row = 0;

    for line in lines:
        values = line.replace('\t', ' ').replace(' ', ' ').split(' ')
        data[A_row] = values[0:E]
        A_row += 1

    fin.close()
    data = np.sort(data, axis=0)
    #data = data[:,1:]
    plt.figure(figsize=(9, 6))

    rep_temp=np.append(data[label[0], :],data[label[1], :],axis=0)
    rep_temp=np.append(rep_temp,data[label[2], :],axis=0)

    '''
    for j in range(3):
        temp=(data[label[j], :])
        rep_temp=np.append(rep_temp,temp,axis=0)
    '''
    rep_temp = rep_temp[:, 1:]
    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(rep_temp, target)
    list=[103,212,319]
    count=0;
    for j in range(3):
        temp = (X_tsne[count:list[j], :])
        plt.scatter(temp[:, 0], temp[:, 1], c=c[j], label=id[j])
        count = list[j]

    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    visual_data()

