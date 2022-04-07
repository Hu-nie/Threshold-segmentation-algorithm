import numpy as np
import os
import glob
import pydicom
from lib.dicom_numpy import *
import cv2
import time



## 이미지 해상도 확인 후 데이터 3D 배열로 결합
def extract_voxel_data(path):
    datasets = [pydicom.dcmread(f) for f in glob.glob(os.path.join(path, '*.dcm'))]

    try:
        voxel_ndarray, _ = combine_slices(datasets, rescale=True)
    except DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray.T


def Otsu(normArr):
    normal = []
    norm_arr= normArr
    print(normArr.shape)
    re_arr = normArr.reshape(norm_arr.shape[1]*norm_arr.shape[2],norm_arr.shape[0])
    cutoff , _ = cv2.threshold(re_arr, -1, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )
    

    for arr in norm_arr:
        _ , t_otsu = cv2.threshold(arr,cutoff, 255, cv2.THRESH_BINARY_INV)
        tempDiff = cv2.subtract(arr, t_otsu)
        normal.append(tempDiff)
    normal = np.array(normal)

    return cutoff, normal[normal !=0]

def voxelNorm(whole_arr):
    min_v = np.min(whole_arr)
    max_v = np.max(whole_arr)

    for n, arr in enumerate(whole_arr):
        whole_arr[n] = 255 * (arr - min_v) * (1 / (max_v - min_v))
        whole_arr = whole_arr.astype(np.uint16)

    return whole_arr, min_v, max_v


def deNorm(norm, min_v, max_v):
    return (norm / 255) * (max_v - min_v) + min_v


def newtonRaphson(weight,sd):
    x = 3

    while True:
        d =  ftn(x,sd,weight)/ftn_drv(x)

        if abs(d) <= 1e-7: break
        x = x-d    

    return x

def ftn(x,sd,weight):
    return (x**2)/2-x-np.exp(-x)+np.log(1/sd)-np.log(weight/np.sqrt(2*np.pi*sd))

def ftn_drv(x):
    return x-1+np.exp(-x)

def meanStd(arr):
    return np.mean(arr), np.std(arr)