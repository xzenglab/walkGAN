import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
import os
import seaborn as sns
from gensim.models import Word2Vec, KeyedVectors
length = [10,20,30,40]
walk = [60,80,100,120]
df = pd.read_csv('total/fake_wiki.csv')
df = df.set_index(['length','walk'])
df = df[df['epoch']==5]
# score=[0.831508092,0.849446615,0.846493676,0.859316871,0.847330729,0.879882813,0.854747954,0.860735212,0.857363746,0.853224981,0.87593006,0.868466332,0.833321708,0.848679315,0.882684617,0.839960007]
plt.figure()
for i in range(4):
    score = []
    len = length[i]
    for j in range(4):
        wa = walk[j]
        score.append(df.loc[(len,wa),'macro'])
    plt.plot(walk,score,label='length={}'.format(len),marker='o')
plt.xlabel('walk_per_node')
plt.xticks(walk)
plt.ylabel('macro-f1')
plt.legend(loc="lower right")
plt.show()
dim = [32,64,96,128]
xlabel = [32,64,128,256]
# score=[0.848,0.883,0.880,0.871]
score=[0.408733939,0.425668477,0.460659851,0.473023048,]
plt.figure()
plt.plot(dim,score,marker='o',c='black')
plt.xlabel('dimension')
plt.xticks(dim,xlabel)

plt.ylabel('macro-f1')
plt.show()

