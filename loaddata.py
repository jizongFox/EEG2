# #coding:utf8
# import pyedflib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
#
# f = pyedflib.EdfReader('chb01_01.edf')
# n= f.signals_in_file
# signalbufs = np.zeros((f.getNSamples().shape[0],f.getNSamples()[0]))
# for i in np.arange(n):
#     signalbufs[i, :] = f.readSignal(i)
#
# signalbufs = pd.DataFrame(signalbufs.T,columns=f.getSignalLabels())
from eeglearn.eeg_cnn_lib import azim_proj
import scipy.io as sio
print('Loading data...')
locs = sio.loadmat('Neuroscan_locs_orig.mat')
locs_3d,locs_2d = locs['A'],[]
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(locs_3d[:,0],locs_3d[:,1],locs_3d[:,2])
plt.show(block=False)
# Convert to 2D


for e in locs_3d:
    locs_2d.append(azim_proj(e))
locs_2d = np.array(locs_2d)
plt.figure()
plt.scatter(locs_2d[:,0],locs_2d[:,1]);plt.show()

# from sklearn.decomposition import PCA,KernelPCA
# pca = PCA(n_components=2)
# kpca = KernelPCA(n_components=2,kernel='rbf',fit_inverse_transform=True, gamma=10)
# locs_2d_pca = kpca.fit_transform(locs_3d)
# plt.scatter(locs_2d_pca[:,0],locs_2d_pca[:,1]);plt.show()
# print()