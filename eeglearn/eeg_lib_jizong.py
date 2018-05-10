from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
import math as m
from functools import reduce
import scipy.io
import torch.utils.data as data,torch
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from eeglearn.utils import augment_EEG, cart2sph, pol2cart
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
import torch.nn as nn,torchvision
from torchvision.models import vgg19,vgg13
import torch.nn.functional as F
from torchvision.models.vgg import VGG

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

def load_preprocessed_data():
    print('Loading data...')
    locs = scipy.io.loadmat('../Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = scipy.io.loadmat('../FeatureMat_timeWin.mat')['features']

    subj_nums = np.squeeze(scipy.io.loadmat('../trials_subNums.mat')['subjectNum'])
    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))
    return (np.array(locs_2d),locs_3d), feats[:,:-1], feats[:,-1]-1,fold_pairs

def gen_images(locs, features, n_gridpoints, normalize=True,augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)]) ## theta, alpha and beta frequency band
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ] # here 32 j means the amount of points are 32.
    temp_interp = []
    for c in range(n_colors): # for the theta, alpha and beta channels:
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

def vgg_net(pretrain=True,disable_Classifier=False):
    net = vgg13(pretrained=pretrain)
    if disable_Classifier:
        import types
        def forward(self,x):
            x=self.features(x)
            return x
        net.forward = types.MethodType(forward,net)
    net.classifier = nn.Sequential(
        nn.Linear(512,1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024,4),
    )
    return net

class cov_max (nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net = net
        self.classification2 = nn.Sequential(
            nn.Linear(3584, 10000),
            nn.Dropout(p=0.5),
            nn.Linear(10000, 4),
        )

    def forward(self, input):
        output = net(input[:,0])
        for i in range(1,input.shape[1]):
            output=torch.cat((output,net(input[:,i])),1)
        output = output.view(output.size(0),-1)
        output = self.classification2(output)
        return output

class LTSM_conv(nn.Module):
    def __init__(self,embedding,hidden,net):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.hidden = hidden
        self.rnn = nn.GRU(input_size=embedding,hidden_size=hidden,batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden+512,out_features=200),
            nn.Dropout(p=0.5),
            nn.Linear(200,4)
            )
        self.conv = nn.Conv1d(in_channels=7,out_channels=1,kernel_size=1)
    def forward(self, input):
        shapes = input.shape
        features = self.net(input.view(shapes[0]*shapes[1],shapes[2],shapes[3],shapes[4]))
        features_reshape = features.view(shapes[0],shapes[1],512)
        output_cnn = self.conv(features_reshape).squeeze()

        output, hidden = self.rnn(features_reshape)

        output = self.classifier(torch.cat([hidden.squeeze(), output_cnn], 1))
        return output

class LTSM(nn.Module):
    def __init__(self,embedding,hidden,net):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.hidden = hidden
        self.rnn = nn.GRU(input_size=embedding,hidden_size=hidden,batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden,out_features=200),
            nn.Dropout(p=0.5),
            nn.Linear(200,4)
            )
        self.conv = nn.Conv1d(in_channels=7,out_channels=1,kernel_size=1)
    def forward(self, input):
        shapes = input.shape
        features = self.net(input.view(shapes[0]*shapes[1],shapes[2],shapes[3],shapes[4]))

        features_reshape = features.view(shapes[0],shapes[1],512)
        output,hidden = self.rnn(features_reshape)

        output = self.classifier(hidden.squeeze())
        return output

class dataset(data.Dataset):
    def __init__(self,imgs,target,transform=None) -> None:
        super().__init__()

        np.random.seed(1)
        self.imgs =imgs
        self.target = target

    def __getitem__(self, index):
        if len(self.imgs.shape)==4:
            image = torch.FloatTensor(self.imgs[index])
            target = torch.LongTensor([int(self.target[index])])
            return image,target
        if len(self.imgs.shape)==5:
            image = torch.FloatTensor(self.imgs[:,index])
            target = torch.LongTensor([int(self.target[index])])
            return image,target

    def __len__(self):
        if len(self.imgs.shape)==4:
            return len(self.imgs)
        elif len(self.imgs.shape)==5:
            return self.imgs.shape[1]

def val(dataloader,net):
    net.eval()
    acc = AverageValueMeter()
    acc.reset()
    for i, (img, label) in enumerate(dataloader):
        batch_size = len(label)
        images = Variable(img).cuda()
        labels = Variable(label.squeeze()).cuda()
        output = net(images)
        predictedLabel = torch.max(output,1)[1]
        acc_ = (predictedLabel==labels).sum().type(torch.FloatTensor)/batch_size
        acc.add(acc_.item())
    net.train()
    return acc.value()[0]

def save_function(obj,name):
    import pickle
    with open(name+'.pkl','wb') as f:
        pickle.dump(obj,f)
        f.close()
        print(name+' being saved.')

def load_function(name):
    with open(name+'.pkl','rb') as f:
        obj= pickle.load(f)
        f.close()
        return obj

def subject_based_train_test_splite(X,y,folder,option='3sets'):
    if option=='3sets':
        dim = len(X.shape)
        indice_train = folder[0][len(folder[1]):]
        indice_validation = folder[0][:len(folder[1])]
        indice_test = folder[1]
        if dim ==2 or dim==4: # for feats split:
            X_train,y_train = X[indice_train],y[indice_train]
            X_validation, y_validation = X[indice_validation],y[indice_validation]
            X_test, y_test = X[indice_test], y[indice_test]
        if dim ==5:
            X_train,y_train = X[:,indice_train],y[indice_train]
            X_validation, y_validation = X[:,indice_validation],y[indice_validation]
            X_test, y_test = X[:,indice_test], y[indice_test]
        return (X_train,y_train),(X_validation,y_validation),(X_test,y_test)
    else:
        dim = len(X.shape)
        indice_train = folder[0]
        indice_test = folder[1]
        if dim == 2 or dim == 4:  # for feats split:
            X_train, y_train = X[indice_train], y[indice_train]
            X_test, y_test = X[indice_test], y[indice_test]
        if dim == 5:
            X_train, y_train = X[:, indice_train], y[indice_train]

            X_test, y_test = X[:, indice_test], y[indice_test]
        return (X_train, y_train), (None, None), (X_test, y_test)

def show_image(image):
    assert image.shape[0] ==3
    gray_image = image[0,:,:]+ image[1,:,:] + image[2,:,:]
    plt.imshow(gray_image)

def show_images(images):
    f=plt.figure()
    j=1
    for i in images[0:9]:
        plt.subplot(3,3,j)
        show_image(i)
        j+=1
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch, pickle
    import matplotlib.pyplot as plt
    import torch.backends.cudnn as cudnn
    from sklearn.model_selection import train_test_split

    (locs_2d,_),feats,target,fold_pairs = load_preprocessed_data()

    # Image generations
    '''
    # CNN Mode
    print('Generating images for average images...')
    av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(int(feats.shape[1] / 192))])/ int(feats.shape[1] / 192)
    images = gen_images(locs_2d,av_feats,32, normalize=False)
    save_function(images,'av_images')

    print('Generating images for all time windows...')
    images_timewin = [gen_images(
        locs_2d, feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
         range(int(feats.shape[1] / 192))
         ]
    images_timewin = np.array(images_timewin)
    save_function(images_timewin,'timewindow_images')
    '''
    # av_images = load_function('av_images')

    timewin_images=load_function('timewindow_images')
    #
    (X_train,y_train),(X_validation,y_validation),(X_test,y_test) = subject_based_train_test_splite(timewin_images,target,fold_pairs[2],option='2sets')
    # (X_train,y_train),(X_validation,y_validation),(X_test,y_test) = subject_based_train_test_splite(av_images,target,fold_pairs[11],option='2sets')
    # show_images(X_train[y_train==3])
    # X_train, X_test, y_train, y_test = train_test_split(timewin_images,target,test_size=0.05)

    net = vgg_net(pretrain=False,disable_Classifier=True)
    net.cuda()
    net_lstm = LTSM_conv(embedding=512,hidden=64,net=net)
    net_lstm.cuda()
    # net2 = cov_max(net).cuda()
    # net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True

    train_dataset = dataset(imgs=X_train,target=y_train)
    test_dataset = dataset(imgs=X_test,target=y_test)
    train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False,num_workers=4)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
    optimizer_lstm = torch.optim.Adam(net_lstm.parameters(),lr=1e-4,weight_decay=1e-8)
    # optimizer_max = torch.optim.Adam([
    #     {'params':net2.module.net.features.parameters(),'lr':1e-4},
    #     {'params':net2.module.classification2.parameters()}],lr=1e-4,weight_decay=1e-8)
    # optimizer_max=torch.optim.Adam(net2.parameters(),lr=1e-4)
    epochs =100


    for epoch in range(epochs):
        for i, (img, label) in enumerate(train_loader):
            images = Variable(img).cuda()
            labels = Variable(label.squeeze()).cuda()
            # optimizer_max.zero_grad()
            # optimizer_lstm.zero_grad()
            # optimizer.zero_grad()
            optimizer_lstm.zero_grad()
            output= net_lstm(images)
            loss = criterion(output,labels)
            loss.backward()
            ec = torch.nn.utils.clip_grad_norm_(net_lstm.parameters(), 5)
            optimizer_lstm.step()
        acc_train = val(train_loader,net_lstm)
        acc_test = val(test_loader,net_lstm)
        print(acc_train,acc_test)

    # Class labels should start from 0
    print('Training the CNN Model...')
    # train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'cnn')
