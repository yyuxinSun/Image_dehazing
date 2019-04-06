
# In[1]:
# -*- coding: utf-8 -*-
import math
import numpy as np
import cv2

import matplotlib.pyplot as plt
import time

from collections import deque
from collections import defaultdict
from itertools import combinations_with_replacement

from numpy.linalg import inv


# ## Implement of Faster Min Filter

# In[2]:


def fastmaxminFilter_array(a, w):
    maxfifo, minfifo = deque((0,)), deque((0,))
    lena = len(a)
    maxvalues = [None]*(lena-w+1)
    
    
    minvalues = [None]*(lena-w+1)
    for i in range(1, lena):
        if i >= w:
            maxvalues[i-w] = a[maxfifo[0]]
            minvalues[i-w] = a[minfifo[0]]
        if a[i] > a[i-1]:
            maxfifo.pop()
            while maxfifo:
                if a[i] <= a[maxfifo[-1]]:
                    break
                maxfifo.pop()
        else:
            minfifo.pop()
            while minfifo:
                if a[i] >= a[minfifo[-1]]:
                    break
                minfifo.pop()
        maxfifo.append(i)
        minfifo.append(i)
        if i == (w+maxfifo[0]):
            maxfifo.popleft()
        elif i == (w + minfifo[0]):
            minfifo.popleft()
        maxvalues[lena-w] = a[maxfifo[0]]
        minvalues[lena-w] = a[minfifo[0]]

    return minvalues


# ## Implement of Guided Image Filter

# In[3]:


def boxfilter(I, r):      
    M, N = I.shape
    dest = np.zeros((M, N))

    # cumulative sum over Y axis
    sumY = np.cumsum(I, axis=0)
    # difference over Y axis
    dest[:r + 1] = sumY[r: 2 * r + 1]
    dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    # difference over Y axis
    dest[:, :r + 1] = sumX[:, r:2 * r + 1]
    dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) -         sumX[:, N - 2 * r - 1:N - r - 1]

    return dest


def guided_filter(I, p, r=40, eps=1e-3):
    R, G, B = 0, 1, 2  # index for convenience
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)

    # each channel of I filtered with the mean filter
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base
    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in range(3)]
    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(
            I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R] * means[R] -         a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) *
         I[:, :, G] + boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

    return q


# ## Implement of Dark Channel Prior

# In[4]:


class Node(object):
    def __init__(self,x,y,value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print('%s:%s:%s' %(self.x,self.y,self.value))
        
def getMinChannel(img):

    if len(img.shape)==3 and img.shape[2]==3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    
    return np.min(img, axis=2)

def getAtomsphericLight(darkChannel,img,meanMode = False, percent = 0.001):

    size = darkChannel.shape[0]*darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]

    nodes = []

    for i in range(0,height):
        for j in range(0,width):
            oneNode = Node(i,j,darkChannel[i,j])
            nodes.append(oneNode)	

    nodes = sorted(nodes, key = lambda node: node.value,reverse = True)

    atomsphericLight = 0

    if int(percent*size) == 0:
        for i in range(0,3):
            if img[nodes[0].x,nodes[0].y,i] > atomsphericLight:
                atomsphericLight = img[nodes[0].x,nodes[0].y,i]
        return atomsphericLight

    if meanMode:
        sum = 0
        for i in range(0,int(percent*size)):
            for j in range(0,3):
                sum = sum + img[nodes[i].x,nodes[i].y,j]
        atomsphericLight = int(sum/(int(percent*size)*3))
        return atomsphericLight

    for i in range(0,int(percent*size)):
        for j in range(0,3):
            if img[nodes[i].x,nodes[i].y,j] > atomsphericLight:
                atomsphericLight = img[nodes[i].x,nodes[i].y,j]
    return atomsphericLight


# In[8]:


def getRecoverScene(img, omega=0.8, t0=0.1, blockSize=15, meanMode=False, percent=0.001, refine=False, fasterMin = False):

    imgGray = getMinChannel(img)
    imgDark = getDarkChannel(imgGray, blockSize, fasterMin)
    atomsphericLight = getAtomsphericLight(imgDark,img,meanMode = meanMode,percent= percent)

    imgDark = np.float64(imgDark)
    transmission = 1 - omega * imgDark / atomsphericLight

    transmission[transmission<0.1] = 0.1     
    
    if refine:        
        normI = (img - img.min()) / (img.max() - img.min())  # normalize I
        transmission = guided_filter(normI, transmission, r=40, eps=1e-3)

    sceneRadiance = np.zeros(img.shape)
    img = np.float64(img)
    
    for i in range(3):        
        SR = (img[:,:,i] - atomsphericLight)/transmission + atomsphericLight
                
        SR[SR>255] = 255
        SR[SR<0] = 0                    
        sceneRadiance[:,:,i] = SR  
            
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance


# In[5]:


def getDarkChannel(img, blockSize = 3, optimized = False):

    # validation check
    if len(img.shape)==2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None

    A = int((blockSize-1)/2) #AddSize

    # New height and new width
    H = img.shape[0] + blockSize - 1
    W = img.shape[1] + blockSize - 1

    imgMiddle = 255 * np.ones((H,W))    

    imgMiddle[A:H-A, A:W-A] = img
    
    imgDark = np.zeros_like(img, np.uint8)
    
    lineResult = 255 * np.ones((H,img.shape[1]))

    if(optimized):
        for i in range(A, H-A):
            lineResult[i] = fastmaxminFilter_array(imgMiddle[i], blockSize)

        for j in range(A, W-A):
            imgDark[:,j-A] = fastmaxminFilter_array(lineResult[:,j-A], blockSize)
    else:
        localMin = 255
        for i in range(A, H-A):
            for j in range(A, W-A):
                x = range(i-A, i+A+1)
                y = range(j-A, j+A+1)
                imgDark[i-A,j-A] = np.min(imgMiddle[x,y]) 
            
    return imgDark


# ## Read in Data & Dehazing

# In[6]:


img = cv2.imread('./haze_image/Foggy_Street.jpg',cv2.IMREAD_COLOR)

# Dark Channel Prior with Guided Image Filter
t0 = time.clock()
dehazed_img1 = getRecoverScene(img, refine=True)
print ("guided filter matting cost: %f" % (time.clock() - t0))

# Dark Channel Prior with Guided Image Filter - faster Min filter
t0 = time.clock()
dehazed_img2 = getRecoverScene(img, refine=True, fasterMin = True)
print ("guided filter matting cost(Faster Min Filter Applied): %f" % (time.clock() - t0))

# Dark Channel Prior without Guided Image Filter
t0 = time.clock()
dehazed_img3 = getRecoverScene(img, refine=False)
print ("no matting cost: %f" % (time.clock() - t0))

# Dark Channel Prior without Guided Image Filter - faster Min filter
t0 = time.clock()
dehazed_img4 = getRecoverScene(img, refine=False, fasterMin = True)
print ("no matting cost(Faster Min Filter Applied): %f" % (time.clock() - t0))


# ## Plot the Result

# In[7]:


fig = plt.figure()
fig.set_size_inches(16, 15)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.imshow(cv2.cvtColor(dehazed_img3, cv2.COLOR_BGR2RGB))

plt.subplot(223)
plt.imshow(cv2.cvtColor(dehazed_img1, cv2.COLOR_BGR2RGB))

img_patch = cv2.imread('./out/Foggy_Street_out.png',cv2.IMREAD_COLOR)
plt.subplot(224)
plt.imshow(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))

plt.show()

