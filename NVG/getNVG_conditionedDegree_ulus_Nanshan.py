#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:37:20 2024

@author: lifei
"""
from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import re
import h5py
import glob
from ts2vg import NaturalVG
import matplotlib.pyplot as plt
from matplotlib import rcParams
from concurrent.futures import ProcessPoolExecutor
# Configure matplotlib
config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False
}
rcParams.update(config)
rcParams['text.usetex'] = True

# yloc configuration
yloc = '100'
if yloc == '100':
    delta = 0.356
    fs = 4000
elif yloc == '000':
    delta = 0.284
    fs = 20000
elif yloc == '-100':
    delta = 0.188
    fs = 20000

# Define paths
folderList = glob.glob(f'/home/lifei/HDD_TOWER/data_f/lifei/Nanshan/threePlanes/y{yloc}mm/x*')
savePath = f'/home/lifei/HDD_TOWER/data_f/lifei/Nanshan/code/NVGMethod/NVG_ulk_Nanshan_y{yloc}.h5'
folderList.sort()
print("Folders found:", folderList)

#%%

def getNVG_degree(signal, chunk_size=10000):
    """
    Calculate the degree of a Natural Visibility Graph in chunks.

    Parameters:
        signal (array-like): The input signal.
        chunk_size (int): The number of points in each chunk.

    Returns:
        degrees (list): List of all degrees computed from chunks.
    """
    degrees = []
    for i in range(0, len(signal), chunk_size):
        # Get a chunk of the signal
        chunk = signal[i:i + chunk_size]
        
        # Build the visibility graph for the chunk
        vg = NaturalVG()
        vg.build(chunk)
        
        # Append the degrees
        degrees.extend(vg.degrees)
        
        # Free memory
        vg = None
        print(i)

    return degrees
#%%
def compute_chunk_degree(chunk):
    """
    Compute the degree of a Natural Visibility Graph for a given chunk.
    This function will run in parallel for multiple chunks.

    Parameters:
        chunk (array-like): A portion of the signal.

    Returns:
        list: The degrees computed for the chunk.
    """
    vg = NaturalVG()
    vg.build(chunk)
    degrees = vg.degrees
    vg = None  # Free memory
    return degrees
#%%
def parallel_getNVG_degree(signal, chunk_size=1000, max_workers=4):
    """
    Compute NVG degrees in parallel while preserving order.

    Parameters:
        signal (array-like): The input signal.
        chunk_size (int): Number of points in each chunk.
        max_workers (int): Number of parallel processes.

    Returns:
        list: Degrees computed for the entire signal in order.
    """
    # Split the signal into chunks
    chunks = [signal[i:i + chunk_size] for i in range(0, len(signal), chunk_size)]

    degrees = []

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_chunk_degree, chunks))

    # Combine results while preserving order
    for result in results:
        degrees.extend(result)

    return degrees
#%%
for iFolder in range(len(folderList)):
    folderNameSplit = re.split('x|_', folderList[iFolder])
    

    fileList = glob.glob(f"{folderList[iFolder]}/*0.out")
    fileList.sort()
    ul_kp_ave = []
    ul_kn_ave = []
    ul_knp_ave = []
    z = []
    for iFile in range(len(fileList)):
    
        fileNameSplit = re.split('z|-|.out', fileList[iFile])
        z.append(float(fileNameSplit[-2]))
        
        
        data = np.loadtxt(fileList[iFile],skiprows=1)
        u = np.array(data[:,0])
        u_mean = np.mean(u)
        flu_u = u - u_mean
        fc = u_mean/delta
        nyquist = fs / 2
        b, a = butter(4, fc / nyquist, btype='low')
        
        ul = filtfilt(b, a, flu_u)  
        us_en = abs(hilbert(flu_u - ul))      
        us_tmp = filtfilt(b, a, us_en)        
        us = us_tmp - np.std(us_tmp) 
        ul_pos = np.where(ul>0)
        ul_neg = np.where(ul<0)
        
        # if __name__ == "__main__":
        #     print(f"Processed {iFile}th file degrees from the u.")
        #     signal = flu_u  
        #     chunk_size = 45000  # Process in chunks of 5,000 points
        #     max_workers = 30  # Number of parallel processes
    
        #     u_tmp = parallel_getNVG_degree(signal, chunk_size=chunk_size, max_workers=max_workers)
            
        # u_degree_ave.append(np.mean(u_tmp))
        
        if __name__ == "__main__":
            print(f"Processed {iFile}th file degrees from the ul.")
            signal = ul  
            chunk_size = 45000  # Process in chunks of 5,000 points
            max_workers = 30  # Number of parallel processes
    
            ul_tmp = parallel_getNVG_degree(signal, chunk_size=chunk_size, max_workers=max_workers)
            ul_tmp = np.array(ul_tmp)
            kp = ul_tmp[ul_pos]
            kn = ul_tmp[ul_neg]
            
            knp = np.mean(kn)/np.mean(kp)
            
        ul_kp_ave.append(np.mean(kp))
        
        ul_kn_ave.append(np.mean(kn))
        
        ul_knp_ave.append(knp)
        # if __name__ == "__main__":
        #     print(f"Processed {iFile}th file degrees from the us.")
        #     signal = us  # Example signal with 100,000 points
        #     chunk_size = 45000  # Process in chunks of 5,000 points
        #     max_workers = 30  # Number of parallel processes
    
        #     us_tmp = parallel_getNVG_degree(signal, chunk_size=chunk_size, max_workers=max_workers)
            
        # us_degree_ave.append(np.mean(us_tmp))
        del ul_tmp
        print(f'{iFolder}th for x = {folderNameSplit[-1]}mm finished')
        print(f'{iFile}th for z = {fileNameSplit[-2]}mm finished')
#%%
    with h5py.File(savePath,'a') as f:
        f.create_dataset(f'ul_kp_x{folderNameSplit[-1]}mm',data = ul_kp_ave)
        f.create_dataset(f'ul_kn_x{folderNameSplit[-1]}mm',data = ul_kn_ave)
        f.create_dataset(f'ul_knp_x{folderNameSplit[-1]}mm',data = ul_knp_ave)

#%%
with h5py.File(savePath,'a') as f:
    f.create_dataset('z',data = z)