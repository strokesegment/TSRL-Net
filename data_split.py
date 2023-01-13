import nibabel as nib
import numpy as np
import os
import random
import copy
from PIL import Image
import time
from collections import Counter
from matplotlib import pyplot as plt

def nii_to_h5(path_nii, path_save, ratio=0.8):
    data = []
    label = []
    ori = []
    list_site = os.listdir(path_nii)
    list_data = []
    ori_min = 10000
    ori_max = 0
    for dir_num, dir_site in enumerate(list_site):
        if dir_site[-3:] == 'csv':
            continue

        list_patients = os.listdir(path_nii + '/' + dir_site)
        for dir_patients in list_patients:
            for t0n in ['/t01/', '/t02/']:
                try:
                    location = path_nii + '/' + dir_site + '/' + dir_patients + t0n
                    location_all = os.listdir(location)
                    for i in range(len(location_all)):
                        location_all[i] = location + location_all[i]
                    list_data.append(location_all)
                except:
                    continue
    random.shuffle(list_data)

    for num, data_dir in enumerate(list_data):
        for i, deface in enumerate(data_dir):
            if deface.find('deface') != -1:
                ori = nib.load(deface)
                ori = ori.get_fdata()
                ori = np.array(ori)
                ori = ori.transpose((2, 1, 0))
                if ori_max < ori.max():
                    ori_max = ori.max()
                if ori_min > ori.min():
                    ori_min = ori.min()
                del list_data[num][i]
                break

        label_merge = []
        for i, dir_data in enumerate(list_data[num]):
            img = nib.load(dir_data)
            img = np.array(img.get_fdata())
            img = img.transpose((2, 1, 0))
            img[img>0]=1
            label_merge.append(img)
        label_merge = np.sum(label_merge, axis=0)
        label_merge[label_merge>1] = 1
        print(str(num) + '/' + str(len(list_data)), 'max=', str(ori.max()), 'min=', str(ori.min()))
        if num == 0 or num == int(ratio * len(list_data)): #or num == int(0.9 * len(list_data)):
            data = copy.deepcopy(ori)
            label = copy.deepcopy(label_merge)
        else:
            data = np.concatenate((data, ori), axis=0)
            label = np.concatenate((label, label_merge), axis=0)

        if num == int(ratio * len(list_data))-1:
            print('saving train set...')
            data = np.array(data, dtype=float)
            label = np.array(label, dtype=int)
            
            np.save(path_save + '/train_' + str(ratio)+'data', data)
            np.save(path_save + '/train_' + str(ratio) + 'label', label)
            data = []
            label = []
            print('Finished!')

        
        elif num == len(list_data) - 1:
            print('saving test set...')
            data = np.array(data, dtype=float)
            label = np.array(label, dtype=int)
            np.save(path_save + '/test_' + str(ratio) + 'data', data)
            np.save(path_save + '/test_' + str(ratio) + 'label', label)
            print('Finished!')
    
    return ori_max, ori_min
    # '''



def load_h5(path_data,path_label, size=None, test_programme=None, only=False):

    data=np.load(path_data)
    label=np.load(path_label)

    if test_programme is not None:
        data = data[:test_programme]
        label = label[:test_programme]

    data_only = []
    label_only = []
    if only is True:
        for i in range(len(data)):
            if label[i].max() == 1:
                data_only.append(data[i])
                label_only.append(label[i])
        del data, label
        data = data_only
        label = label_only

    data = np.uint8(np.multiply(data, 2.55))
    # label = np.uint8(np.multiply(label, 255))

    if size is not None:
        data_resize = []
        label_resize = []
        for i in range(len(data)):
            data_resize_single = Image.fromarray(np.float32(data[i])).crop((10, 40, 190, 220))
            data_resize_single = data_resize_single.resize(size, Image.ANTIALIAS)
            data_resize_single = np.asarray(data_resize_single)

            label_resize_single = Image.fromarray(np.int8(label[i])).crop((10, 40, 190, 220))
            label_resize_single = label_resize_single.resize(size, Image.ANTIALIAS)
            label_resize_single = np.asarray(label_resize_single)

            data_resize.append(data_resize_single)
            label_resize.append(label_resize_single)

        data = np.array(data_resize, dtype=float)
        label = np.array(label_resize, dtype=int)

    data = data - data.min()
    data = data / data.max()
    # print(Counter(label.flatten()))

    return data, label


def data_toxn(data, z):
    data_xn = np.zeros((data.shape[0], data.shape[1], data.shape[2], z))
    for patient in range(int(len(data) / 189)):
        for i in range(189):
            for j in range(z):
                if i + j - z // 2 >= 0 and i + j - z // 2 < 189:
                    data_xn[patient * 189 + i, :, :, j] = data[patient * 189 + i + j - z // 2]
                    # print(i, i + j - z // 2)
                else:
                    data_xn[patient * 189 + i, :, :, j] = np.zeros_like(data[0])
    return data_xn

if __name__ == "__main__":
    start = time.time()
    path_nii = '/home//ATLAS_R1.1'
    path_save = '/home//h5'
    ratio = 0.8
    img_size = [192, 192]
    ori_max, ori_min = nii_to_h5(path_nii, path_save, ratio=ratio)

    print('using :{}'.format(time.time() - start))

    print('loading training-data...')
    time_start = time.time()
    original, label = load_h5(path_save + '/train_' + str(ratio) + 'data.npy',
                              path_save + '/train_' + str(ratio) + 'label.npy',
                              size=(img_size[1], img_size[0]))

    original = data_toxn(original, 4)
    original = original.transpose((0, 3, 1, 2))
    np.save(path_save + '/data',original)
    del original

    label = data_toxn(label, 1)
    label = label.transpose((0, 3, 1, 2))
    print("train label:", Counter(label.flatten()))
    np.save(path_save + '/label', label)
    del label
    print('training_data done!, using:', str(time.time() - time_start) + 's\n\nloading validation-data...')

    

    time_start = time.time()
    original_test, label_test = load_h5(path_save + '/test_' + str(ratio) + 'data.npy',
                              path_save + '/test_' + str(ratio) + 'label.npy',
                                      size=(img_size[1], img_size[0]))
    original_test = data_toxn(original_test, 4)
    original_test = original_test.transpose((0, 3, 1, 2))
    np.save(path_save + '/test_data', original_test)
    del original_test

    label_test = data_toxn(label_test, 1)
    label_test = label_test.transpose((0, 3, 1, 2))
    np.save(path_save + '/test_label', label_test)
    
    del label_test

    # print('validation_data done!, using:', str(time.time() - time_start) + 's\n\n')
