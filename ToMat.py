
from __future__ import print_function
import os
import numpy as np
import torch.utils.data as data
import torch
import os
import sys
import importlib
from Dataset import MatDataset
from Model2D_3D import DURE2D_3DWithAdaptiveConv
importlib.reload(sys)
import scipy.io 
from skimage.metrics import peak_signal_noise_ratio as PSNR

def get_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

 
dtype = torch.cuda.FloatTensor
if __name__ == "__main__":
    torch.cuda.set_device(4)
    ##### read dataset #####

    dataset_folder = '/data/datasets/pansharpening/NBU_dataset0730'
    dataset_full_path = '/data/datasets/pansharpening/NBU_dataset_FR_0924'

    weight_path = '/data/cjj/projects/UnifiedPansharpening/experiment/Dim24[2,3,3,4]Stage3/epoch=295.pth'
    # validation
    # val_gf_path = os.path.join(dataset_folder,'GF1/test')
    # val_qb_path = os.path.join(dataset_folder,'QB/test')
    # val_wv2_path = os.path.join(dataset_folder,'WV2/test')
    # val_wv4_path = os.path.join(dataset_folder,'WV4/test')
    # val_gf_dataset, val_qb_dataset, val_wv2_dataset, val_wv4_dataset = \
    #                             MatDataset(val_gf_path),MatDataset(val_qb_path), MatDataset(val_wv2_path), \
    #                             MatDataset(val_wv4_path)
    # list_val_dataset = [val_gf_dataset, val_qb_dataset, val_wv2_dataset, val_wv4_dataset]

    dataset_namelist = ['GF1', 'QB', 'WV2', 'WV4']
    num_datasets = len(dataset_namelist)
    dataset_labels = {'GF1': 0, 'QB': 1, 'WV2': 2, 'WV4': 3}
    dataset_name = "WV2"
    label = dataset_labels[dataset_name]
    one_hot = get_one_hot(label, num_datasets).unsqueeze(0)
    
    # full = True
    full = True
    save_root = '/PublicData/cjj/UnifiedPansharpening'

    
    if full:
        test_path = os.path.join(dataset_full_path, dataset_name)
        SaveDataPath = os.path.join(save_root, dataset_name, 'FR') 
    else:
        test_path = os.path.join(dataset_folder, dataset_name, 'test')
        SaveDataPath = os.path.join(save_root, dataset_name, 'RR')
    if not os.path.exists(SaveDataPath):
        os.makedirs(SaveDataPath)

    
    with torch.no_grad():
        # 初始化模型 (使用与训练相同的参数)
        Generator = DURE2D_3DWithAdaptiveConv(8,3,32).cuda()
        
        # 加载checkpoint
        checkpoint = torch.load(weight_path)
        Generator.load_state_dict(checkpoint['model_state_dict'])
        Generator.eval()
        # exit(0)
        count = 0    
        psnr_values = []
 
        for img_name in os.listdir(os.path.join(test_path,'PAN_128')):
            
            # 加载数据
            ms = scipy.io.loadmat(os.path.join(test_path,'MS_32',img_name))['ms0'][...]
            pan = scipy.io.loadmat(os.path.join(test_path,'PAN_128',img_name))['pan0'][...]
            if full:
                gt = scipy.io.loadmat(os.path.join(test_path,'MS_128',img_name))['usms0'][...]
            else:
                gt = scipy.io.loadmat(os.path.join(test_path,'GT_128',img_name))['gt0'][...]
            
            # 转换数据格式，与训练代码保持一致
            ms = torch.from_numpy(ms).float().cuda()
            pan = torch.from_numpy(pan).float().cuda()
            gt = torch.from_numpy(gt).float().cuda()
            
            # 调整维度
            ms = ms.unsqueeze(0).permute(0,3,1,2)  # (1,C,H,W)
            pan = pan.unsqueeze(0).unsqueeze(1)    # (1,1,H,W)
            gt = gt.unsqueeze(0).permute(0,3,1,2)  # (1,C,H,W)
            
            # 模型推理
            out = Generator(ms, pan, one_hot)
            
            print(out.shape)

            # 计算PSNR，与训练代码验证部分保持一致
            out = out.cpu().numpy()
            gt = gt.cpu().numpy()
            
            # 计算PSNR
            
            psnr = PSNR(gt[0].transpose(1,2,0), out[0].transpose(1,2,0), data_range=1.0)
            psnr_values.append(psnr)
            print(f"Image: {img_name}, PSNR: {psnr:.4f}")
            
            # 保存结果
            scipy.io.savemat(os.path.join(SaveDataPath,img_name),{'sr':out})
            
        # 计算并打印平均PSNR
        avg_psnr = np.mean(psnr_values)
        print(f"\nAverage PSNR: {avg_psnr:.4f}")