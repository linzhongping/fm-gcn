import pickle as pkl
import scipy.io as sio
from scipy.sparse import vstack
import numpy as np
# get A

a = pkl.load(open('ind.pubmed.graph','rb'),encoding='latin1')
A = np.zeros((19717,19717))
for _ in a:
    for i in a[_]:
        A[_,i] = 1
print(A.shape)







allx = pkl.load(open('ind.pubmed.allx','rb'),encoding='latin1') #稀疏矩阵
tx = pkl.load(open('ind.pubmed.tx','rb'),encoding='latin1') #稀疏矩阵

X = vstack([allx,tx]).toarray()
print(X.shape)


ally = pkl.load(open('ind.pubmed.ally','rb'),encoding='latin1')
ty = pkl.load(open('ind.pubmed.ty','rb'),encoding='latin1')
Y = vstack([ally,ty]).toarray()
print(Y.shape)
#读取旧文件

sio.savemat('pubmed_with_attributes.mat',{'A':A,'Y':Y,'X':X})