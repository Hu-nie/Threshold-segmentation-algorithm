import multiprocessing as mp
from pydicom import dcmread
import glob, os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

def getResolution(path):
    image = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[0])
    image_array = sitk.GetArrayFromImage(image)
    whole_array = np.expand_dims(np.empty(((image_array[0].shape)[0],(image_array[0].shape)[1])),axis=0)

    return whole_array

def train(path):
    dcm_paths = glob.glob(os.path.join(path, "*.dcm"))
    mp.freeze_support()
    pool = mp.Pool(processes=16)
    dcms_info = [pool.apply(dcmread, args=(dcm_path, None, False, False, None)) for dcm_path in dcm_paths]
    
    return dcms_info

if __name__ == '__main__':
    whole_data = train('D:/3_jeonbuk university/TOF_MR/LCH/TOF_1/')
    whole_arr = getResolution('D:/3_jeonbuk university/TOF_MR/LCH/TOF_1/')
    
    for i in tqdm(whole_data):
        
            img_arr = i.pixel_array
            
            img_arr = np.expand_dims(img_arr, axis=0)
            whole_arr = np.concatenate((whole_arr, img_arr), axis=0)
            
    print(whole_arr.shape)