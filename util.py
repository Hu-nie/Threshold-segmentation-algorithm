import numpy as np
import os
import glob
import pydicom
from shapely.geometry import LineString
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
    for arr in normArr:
        _ , t_otsu = cv2.threshold(arr, -1, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )
        tempDiff = cv2.subtract(arr, t_otsu)
        normal.append(tempDiff)
        
    # return normal
    return np.array(normal)
    

def meanStd(arr):
    return np.mean(arr), np.std(arr)

def voxelNorm(whole_arr):
    min_v = np.min(whole_arr)
    max_v = np.max(whole_arr)
    std_v = np.std(whole_arr)
    mean_v = np.mean(whole_arr)
    
    for n, arr in enumerate(whole_arr):
        whole_arr[n] = ((arr - min_v) * (1 / (max_v - min_v) * 255))
        whole_arr = whole_arr.astype(np.uint8)

    return whole_arr, min_v, max_v, std_v,mean_v


def deNorm(norm, min_v, max_v):
    value = (norm / 255) * (max_v - min_v) + min_v

    return value


def normal_dist(x, mean, sd):
    return (1 / np.sqrt(2 * (np.pi * sd))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)



def gumbel_dist(x, mean, sd):
    return 1 / sd * np.exp(-((x - mean) / sd) - np.exp(-((x - mean) / sd)))


def getIntersection(weight, normal):

    arr_nonzero = normal[normal !=0]
    mean = np.mean(arr_nonzero)
    std = np.std(arr_nonzero)

    arr2list = np.unique((arr_nonzero.flatten()).tolist())
    
    n_pdf = normal_dist(arr2list, mean, std)
    g_pdf = gumbel_dist(arr2list, mean, std)
    # print(len(n_pdf))
    line_1 = LineString(np.column_stack((arr2list, g_pdf)))
    line_2 = LineString(np.column_stack((arr2list, weight * n_pdf)))
    inter = line_1.intersection(line_2)
 
    return arr2list, n_pdf, g_pdf, inter

def newtonRaphson(weight,sd):
    x = 3
    while True:
        x_numeric = x - ((x**2*0.5)-x-np.exp(-x)+np.log(1/sd)-np.log(weight/np.sqrt(2*np.pi*sd)))/(x-1+ np.exp(-x))
        if x-x_numeric <= 1e-6: break
        x = x_numeric
    return (x_numeric)
    # # print('근사해: '+ str(x_numeric))

def intersection(normal,weight):
    arr_nonzero = normal[normal !=0]
    mean = np.mean(arr_nonzero)
    sd = np.std(arr_nonzero)

    for i in range(0,256):
        if weight*normal_dist(i,mean,sd) <= gumbel_dist(i,mean,sd): break 
    
    return i