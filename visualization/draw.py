import numpy as np
from sklearn.manifold import TSNE
#import pandas as pd
import numpy as np
import os
import sys
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# label = np.zeros(34)
def main():
    #with open('../data/wiki/wiki_labels_three.txt', 'r') as f:
    with open('../data/wiki/wiki_labels_three.txt', 'r') as f:
        lines = f.readlines()
        label = []
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            line = [int(x) for x in line]
            label.append(line)
    c = ['r','b','g']
    id = ['label_0','label_1','label_2']
    file = '../save/karate/'
    pre = 'karate.embeddings'
    rep = np.loadtxt(file+pre)
    rep = np.sort(rep, axis=0)
    rep = rep[:,1:]
    if rep.shape[1]>2:
        rep = TSNE().fit_transform(rep)
    plt.figure(figsize=(9, 6))
    for j in range(3):
        rep_temp = rep[label[j],:]
        plt.scatter(rep_temp[:,0],rep_temp[:,1],c=c[j],label=id[j])
    pre = pre.split('.')[0]
    plt.savefig('../save/wiki/'+pre+'.png')
    plt.legend()
'''
files = os.listdir('rep')
for file in files:
    if 'club' in file:
        rep = np.loadtxt('rep/'+file)
        rep = np.sort(rep, axis=0)
        rep = rep[:,1:]
        if rep.shape[1]>2:
            rep = TSNE().fit_transform(rep)
        plt.figure()
        for j in range(2):
            rep_temp = rep[label[j],:]
            plt.scatter(rep_temp[:,0],rep_temp[:,1],c=c[j],label=id[j])
        pre = file.split('.')[0]
        plt.savefig('fig/'+pre+'.jpg')
        plt.legend()
'''

if __name__ == "__main__":
  main()