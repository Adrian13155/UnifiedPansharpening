import sys
sys.path.append("/data/cjj/projects/UnifiedPansharpening")
import os
from datetime import datetime
import torch.nn.functional as F
import argparse
from Dataset import MatDataset, CombineMatDataset, MatWithTextDataset
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from tqdm import tqdm
import logging 
from Model3D import DURE3D
from Model2D_3D import DURE2D_3D, DURE2D_3DWithAdaptiveConv
from Model3Dwith2D import DURE3Dwith2D
from skimage.metrics import peak_signal_noise_ratio as PSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from Compared_Methods.PromptIR.PromptIR import PromptIR_multitasks

def get_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main(opt):

    now = datetime.now()

    # 格式化为字符串，包含月、日、小时和分钟
    formatted_time = now.strftime("%m-%d_%H:%M")
    save_dir = os.path.join(opt.save_dir, f'{formatted_time}_{opt.exp_name}')
    os.makedirs(save_dir,exist_ok=True)
    lr = opt.learning_rate

    total_iteration = opt.total_iteration
    num_epoch = opt.num_epochs
    batch_size =  opt.batch_size

    # train
    dataset_folder = opt.pan_root

    best_psnr_gf, best_psnr_qb = -float('inf'), -float('inf')
    best_psnr_wv4, best_psnr_wv2 = -float('inf'), -float('inf')

    gf_path = os.path.join(dataset_folder,'GF1/train')
    qb_path = os.path.join(dataset_folder,'QB/train')
    wv2_path = os.path.join(dataset_folder,'WV2/train')
    wv4_path = os.path.join(dataset_folder,'WV4/train')
    gf_dataset, qb_dataset, wv2_dataset, wv4_dataset = MatDataset(gf_path), MatDataset(qb_path), MatDataset(wv2_path), MatDataset(wv4_path)
    dataset_labels = {'GF': 0, 'QB': 1, 'WV2': 2, 'WV4': 3}
    train_dataset = CombineMatDataset(datasets=[gf_dataset, qb_dataset, wv2_dataset, wv4_dataset],
                            dataset_labels=[dataset_labels['GF'], dataset_labels['QB'], dataset_labels['WV2'], dataset_labels['WV4']])


    del gf_dataset, qb_dataset, wv2_dataset, wv4_dataset

    # validation
    val_gf_path = os.path.join(dataset_folder,'GF1/test')
    val_qb_path = os.path.join(dataset_folder,'QB/test')
    val_wv2_path = os.path.join(dataset_folder,'WV2/test')
    val_wv4_path = os.path.join(dataset_folder,'WV4/test')
    val_gf_dataset, val_qb_dataset, val_wv2_dataset, val_wv4_dataset = \
                                MatDataset(val_gf_path),MatDataset(val_qb_path), MatDataset(val_wv2_path), \
                                MatDataset(val_wv4_path)
    list_val_dataset = [val_gf_dataset, val_qb_dataset, val_wv2_dataset, val_wv4_dataset]

    dataset_namelist = ['GF', 'QB', 'WV2', 'WV4']
    num_datasets = len(dataset_namelist)

    model = PromptIR_multitasks(dim = 20, num_blocks=[4, 6, 6, 8], decoder=True, prompt_dim=[20,40,104]).cuda()
    

    logger = get_logger(os.path.join(save_dir,f'run_{opt.exp_name}.log'))
    logger.info(opt)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters() )/1e6} M")
    logger.info(f"Network Structure: {str(model)}")

    if opt.checkpoint_path is not None:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint)
        logger.info(f"Load Checkpoint from {opt.checkpoint_path}")


    optimizer_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08) 
    lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-6)
    L1 = nn.L1Loss().cuda() 

    for epoch in range(opt.epoch_start,num_epoch):
        train_dataset.shuffle()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        pbar = tqdm(train_dataloader)
        model.train()

        for index, train_data_lists in enumerate(pbar):
            for list,data_name in zip(train_data_lists, dataset_namelist):
                optimizer_G.zero_grad() 
                datas, one_hot = list
                inp_ms, inp_pan, inp_gt = datas
                inp_ms = inp_ms.type(torch.FloatTensor).cuda().permute(0,3,1,2)
                inp_pan = inp_pan.type(torch.FloatTensor).cuda().unsqueeze(1)
                inp_gt = inp_gt.type(torch.FloatTensor).cuda().permute(0,3,1,2) 

                restored  = model(inp_ms, inp_pan)

                loss_l1 = L1(restored, inp_gt)
                loss_G = loss_l1 
                loss_G.backward()
                optimizer_G.step()
                torch.cuda.empty_cache() 
                lr_scheduler_G.step()
            current_lr = optimizer_G.param_groups[0]['lr']
            pbar.set_description("Epoch:{}   loss_G:{:6}  lr:{:.6f}".format(epoch, loss_G.item(), current_lr))
            pbar.update()
        
        if epoch % 5== 0:
            model.eval() 
            with torch.no_grad():
                psnr = []
                ## 分别对四个数据集验证
                for dataset, data_name in zip(list_val_dataset,dataset_namelist):
                    val_dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
                    count = 0
                    sum_psnr = 0
                    label = dataset_labels[data_name]
                    one_hot = get_one_hot(label, num_datasets).unsqueeze(0)
                    for index, datas in enumerate(tqdm(val_dataloader,desc=f"Validating {data_name}")):
                        count += 1
                        inp_ms, inp_pan, inp_gt = datas[0], datas[1], datas[2]
                        inp_ms = inp_ms.type(torch.FloatTensor).cuda().permute(0,3,1,2)
                        inp_pan = inp_pan.type(torch.FloatTensor).cuda().unsqueeze(1)
                        inp_gt = inp_gt.type(torch.FloatTensor).permute(0,3,1,2)
                        output = model(inp_ms, inp_pan)

                        netOutput_np = output.cpu().numpy()[0]
                        gtLabel_np = inp_gt.numpy()[0]
                        psnrValue = PSNR(gtLabel_np, netOutput_np)
                        sum_psnr += psnrValue                         
                    avg_psnr = sum_psnr / count
                    psnr.append(avg_psnr)
                    torch.cuda.empty_cache()   
                if psnr[0] > best_psnr_gf:
                    best_psnr_gf = psnr[0]
                    best_index = epoch
                    torch.save(model.state_dict(), os.path.join(save_dir,'Best_gf.pth'))
                if psnr[1] > best_psnr_qb:
                    best_psnr_qb = psnr[1]
                    torch.save(model.state_dict(), os.path.join(save_dir,'Best_qb.pth'))
                if psnr[2] > best_psnr_wv2: 
                    best_psnr_wv2 = psnr[2]
                    torch.save(model.state_dict(), os.path.join(save_dir,'Best_wv2.pth'))
                if psnr[3] > best_psnr_wv4:
                    best_psnr_wv4 = psnr[3]
                    torch.save(model.state_dict(), os.path.join(save_dir,'Best_wv4.pth'))
                ## record
                logger.info('Epoch:[{}]\t PSNR_GF = {:.4f}\t  PSNR_QB = {:.4f}\t PSNR_WV2 = {:.4f}\t PSNR_WV4 = {:.4f}\t BEST_GF_PSNR = {:.4f}\t BEST_epoch = {}'.format(
                            epoch, psnr[0], psnr[1], psnr[2], psnr[3], best_psnr_gf, best_index))
                print(
                    'Epoch:[{}]\t PSNR_GF = {:.4f}\t  PSNR_QB = {:.4f}\t PSNR_WV2 = {:.4f}\t PSNR_WV4 = {:.4f}\t BEST_GF_PSNR = {:.4f}\t BEST_epoch = {}'.format(
                        epoch, psnr[0], psnr[1], psnr[2], psnr[3], best_psnr_gf, best_index))
                
            torch.save(model.state_dict(), os.path.join(save_dir,f'epoch={epoch}.pth'))
            
def get_opt():
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('--exp_name', type=str, default='Compared:PromptIRContinue', help='experiment name')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=0.00009, type=float)
    parser.add_argument('-batch_size', help='Set the training batch size', default=4, type=int)
    parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=76, type=int)
    parser.add_argument('-num_epochs', help='', default=400, type=int)
    parser.add_argument('-pan_root', help='', default='/data/datasets/pansharpening/NBU_dataset0730', type=str)
    parser.add_argument('-save_dir', help='日志保存路径', default='/data/cjj/projects/UnifiedPansharpening/Compared_Methods/experiment', type=str)
    parser.add_argument('-gpu_id', help='', default=0, type=int)
    parser.add_argument('-Ch', help='', default=8, type=int)
    parser.add_argument('-Stage', help='', default=4, type=int)
    parser.add_argument('-nc', help='', default=32, type=int)
    parser.add_argument('-total_iteration', help='', default=2e5, type=int)
    parser.add_argument('-checkpoint_path', help='', default='/data/cjj/projects/UnifiedPansharpening/Compared_Methods/experiment/04-22_22:45_Compared:PromptIR/epoch=75.pth', type=str)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    opt = get_opt()
    torch.cuda.set_device(opt.gpu_id)
    main(opt)

