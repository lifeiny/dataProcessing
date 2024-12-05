#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:00:52 2024

@author: lifei
"""

import numpy as np
import re
import h5py
import glob
from ts2vg import NaturalVG
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from pyinform.transferentropy import transfer_entropy
import matplotlib as mpl
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # front type
    "axes.unicode_minus": False #negative sign display
}
rcParams.update(config)
rcParams['text.usetex'] = True

yloc = '-100'

if yloc =='100':
    fs = 4000
    delta = 0.356
if yloc =='000':
    fs = 20000
    delta = 0.284
if yloc == '-100':
    fs = 20000
    delta = 0.188

folderList = glob.glob(f'/home/lifei/HDD_GH/Nanshan/rewriteData/rewritePlanes/y{yloc}mm/x*')

savePath = f'/home/lifei/HDD_TOWER/Nanshan/code/TransferEntropy/te_y{yloc}.h5'
folderList.sort()
print(folderList)

#%%
nyquist_freq = fs / 2 
num_bins = 10
for iFolder in range(len(folderList)):
    z = []
    te_ul2us = []
    te_us2ul = []
    # for iFolder in range(len(folderList)):
    
    
    fileList = glob.glob(f'{folderList[iFolder]}/*.out')
    fileList.sort()
    print(fileList)
    nyquist_freq = fs / 2 
    
   
    
    
    folderNameSplit = re.split('x|mm', folderList[iFolder])
    
    for iFile in range(len(fileList)):
        fileNameSplit = re.split( 'z|.out|_',fileList[iFile])
        z.append(float(fileNameSplit[-2]))
        
        data = np.loadtxt(fileList[iFile], skiprows=1)
            
        
        u = data[:,0]
        w = -1*data[:,1]
        
        flu_u = u - np.mean(u)
        flu_w = w - np.mean(w)
         # Nyquist frequency for original 20000 Hz data
        order = 4  # Filter order
        cutoff_freq = np.mean(u)/delta
        b, a = butter(order, cutoff_freq / nyquist_freq, btype='low')
        
        # Apply the filter to the original data
        ul = filtfilt(b, a, flu_u)
        us_en = abs(hilbert(flu_u - ul))
        us = filtfilt(b, a, us_en)
        
        ul_d = np.digitize(ul, np.linspace(min(ul), max(ul), num_bins)) - 1
        us_d = np.digitize(us, np.linspace(min(us), max(us), num_bins)) - 1
        # Assuming X (large scale) and Y (small scale) are your discretized signals
        # embedding length = 1 for first-order transfer entropy
        
        te_ul2us.append(transfer_entropy(us_d, ul_d, k=1)) # Large scale to small scale
        te_us2ul.append(transfer_entropy(ul_d, us_d, k=1))  # Small scale to large scale
        print(iFolder)
        print(folderList[iFolder])
        print(iFile)
        print(fileList[iFile])
        print(f'fs = {fs}')
        
    with h5py.File(savePath, 'a') as f:
        f.create_dataset(f'te_ul2us_y{yloc}_x{folderNameSplit[-2]}', data = te_ul2us)
        f.create_dataset(f'te_us2ul_y{yloc}_x{folderNameSplit[-2]}', data = te_us2ul)

with h5py.File(savePath, 'a') as f:
    f.create_dataset('z', data = z)
#%%
# fig, ax = plt.subplots(dpi=600)

# ax.scatter(te_ul2us,z,marker = 'o',s = 4, c = 'b', label = '$u_L$ to $u_S$')
# ax.scatter(te_us2ul,z,marker = 'o',s = 4, c = 'r', label = '$u_S$ to $u_L$')
# ax.set_xlabel(' Transfer entropy')
# ax.set_ylabel(' Transfer entropy')
# ax.set_title(f'Transfer entropy Nanshan x = 000, y={yloc}mm')
# ax.legend()