#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import matplotlib.pyplot as plt
#import spectral as spy
import torch.nn as nn
import random
import scipy.io
import os

class my_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, ms, lms,gt

    def __len__(self):
        return self.gt_set.shape[0]

class my_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, ms, lms

    def __len__(self):
        return self.pan_set.shape[0]
    
class MatDataset(data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(os.path.join(img_dir, 'MS_32/'))
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        pan = scipy.io.loadmat(os.path.join(self.img_dir, 'PAN_128', self.img_list[idx]))['pan0'][...]
        ms = scipy.io.loadmat(os.path.join(self.img_dir, 'MS_32', self.img_list[idx]))['ms0'][...]
        gt = scipy.io.loadmat(os.path.join(self.img_dir, 'GT_128', self.img_list[idx]))['gt0'][...]

        pan = np.array(pan, dtype=np.float32)
        ms = np.array(ms, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        
        return ms, pan, gt
    
class MatWithTextDataset(data.Dataset):
    def __init__(self, img_dir, text_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(os.path.join(img_dir, 'MS_32/'))
        self.img_list.sort()
        self.text_dir = text_dir

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        pan = scipy.io.loadmat(os.path.join(self.img_dir, 'PAN_128', self.img_list[idx]))['pan0'][...]
        ms = scipy.io.loadmat(os.path.join(self.img_dir, 'MS_32', self.img_list[idx]))['ms0'][...]
        gt = scipy.io.loadmat(os.path.join(self.img_dir, 'GT_128', self.img_list[idx]))['gt0'][...]
        text = os.path.join(self.text_dir, self.img_list[idx]).replace("mat", "npy")
        # print("text_path:", text)
        pan = np.array(pan, dtype=np.float32)
        ms = np.array(ms, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        text = np.load(text).astype(np.float32).squeeze(0)
        
        return ms, pan, gt, text
    
class CombineMatDataset(data.Dataset):
    """
    遥感图像mat文件混合数据集，这里以混合数据集中最小的长度为整个数据集的长度，但会导致较长的数据集中的某些数据训练不到，所以每次epoch都要进行shuffle
    """
    def __init__(self, datasets, dataset_labels):
        self.datasets = datasets
        self.dataset_labels = dataset_labels
        self.min_num = min(len(dataset) for dataset in self.datasets)
        self.dataset_indices = [list(range(len(dataset))) for dataset in self.datasets]

    def __len__(self):
        return self.min_num
    
    def shuffle(self):
        for indices in self.dataset_indices:
            random.shuffle(indices)
    
    def __getitem__(self, index):
        data_lists = []
        for i, dataset in enumerate(self.datasets):
            data = dataset[self.dataset_indices[i][index]]
            label = self.dataset_labels[i]
            one_hot = self.get_one_hot(label, len(self.dataset_labels))
            data_lists.append((data, one_hot))
        return tuple(data_lists)
    
    def get_one_hot(self, label, num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[label] = 1
        return one_hot
    
class CombineMatMaxDataset(data.Dataset):
    """
    遥感图像mat文件混合数据集，这里以混合数据集中最大的长度为整个数据集的长度，对于较少的数据集采样通过取模采样。
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.max_length = max(self.dataset_lengths)
        self.dataset_indices = [list(range(len(dataset))) for dataset in self.datasets]

    def __len__(self):
        return self.max_num
    
    def shuffle(self):
        for indices in self.dataset_indices:
            random.shuffle(indices)
    
    def __getitem__(self, index):
        data_lists = []
        for i, dataset in enumerate(self.datasets):
            # 使用模运算来循环遍历较大数据集
            data_index = index % self.dataset_lengths[i]
            data_lists.append(dataset[data_index])
        return tuple(data_lists)

if __name__ == "__main__":
    dataset_folder = "/data/datasets/pansharpening/NBU_dataset0730"
    text_dataset_path = "/data/cjj/dataset/pansharpening/NBU_dataset0730"
    dataset_namelist = ['GF', 'QB', 'WV2', 'WV4']
    gf_path = os.path.join(dataset_folder,'GF1/train')
    qb_path = os.path.join(dataset_folder,'QB/train')
    wv2_path = os.path.join(dataset_folder,'WV2/train')
    wv4_path = os.path.join(dataset_folder,'WV4/train')

    gf_text_path = os.path.join(text_dataset_path,'GF1/train')
    qb_text_path = os.path.join(text_dataset_path,'QB/train')
    wv2_text_path = os.path.join(text_dataset_path,'WV2/train')
    wv4_text_path = os.path.join(text_dataset_path,'WV4/train')

    gf_dataset, qb_dataset, wv2_dataset, wv4_dataset = MatWithTextDataset(gf_path, gf_text_path), MatWithTextDataset(qb_path, qb_text_path), MatWithTextDataset(wv2_path, wv2_text_path), MatWithTextDataset(wv4_path, wv4_text_path)

    dataset_labels = {'GF': 0, 'QB': 1, 'WV2': 2, 'WV4': 3}
    train_dataset = CombineMatDataset(datasets=[gf_dataset, qb_dataset, wv2_dataset, wv4_dataset],
                            dataset_labels=[dataset_labels['GF'], dataset_labels['QB'], dataset_labels['WV2'], dataset_labels['WV4']])
    
    from torch.utils.data import DataLoader 
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    for i,train_data in enumerate(train_dataloader):
        for list,data_name in zip(train_data, dataset_namelist):
            datas, one_hot = list
            inp_ms, inp_pan, inp_gt, text= datas
            print("text:", text.shape)
            # print(len(a))
        # print("a.shape", a.shape)
        # print("b.shape", b.shape)
        # print("c.shape", c.shape)
        # print("d.shape", d.shape)
        exit(0)
    

