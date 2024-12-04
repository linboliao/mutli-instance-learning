import os
import sys

from batchgenerators.utilities.file_and_folder_operations import listdir

os.environ['CUDA_VISIBLE_DEVICES']='7'
import random
import numpy as np
import argparse
from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from my_utils.dataset import WSIDataset
from my_utils.model import ABMIL, TransMIL, FCLayer, BClassifier,MILNet, PatchGCN, CLAM_SB
import openslide
import h5py
from PIL import Image
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None

sys.path.append('/data2/yhhu/LLB/Code/aslide/')
from aslide import Aslide


def get_aslide_image(wsi, location, patch_level, dimension):
    (W, H) = dimension
    full_image = np.zeros((H, W, 3), dtype=np.uint8)

    read_size = 20000
    (x0, y0) = location
    # 分块读取
    for x in range(x0, W, read_size):
        for y in range(y0, H, read_size):
            # 计算当前块的宽度和高度
            current_w = min(read_size, W - x)
            current_h = min(read_size, H - y)

            # 读取指定区域
            region = wsi.read_region((x, y), patch_level, (current_w, current_h))
            region_array = np.array(region)
            # 将读取的区域放入完整图像中
            full_image[y:y + current_h, x:x + current_w] = region_array

    return Image.fromarray(full_image)

def set_random_seed(seed):
    random.seed(seed)                 # python random module
    np.random.seed(seed)              # numpy module
    torch.manual_seed(seed)           # 为cpu设置
    torch.cuda.manual_seed(seed)      # 为当前gpu设置
    torch.cuda.manual_seed_all(seed)  # 为所有gpu设置
    os.environ['PYTHONHASHSEED'] = str(seed) #为了禁止hash随机化，使得实验可复现
    torch.backends.cudnn.benchmark = False  #设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
## 对dataloader设置随机种子：如果dataloader采用了多线程(num_workers > 1), 那么由于读取数据的顺序不同，最终运行结果也会有差异。
## 在PyTorch的DataLoader函数中为不同的work设置初始化函数，确保您的dataloader在每次调用时都以相同的顺序加载样本（随机种子固定时）
# def worker_init_fn(worker_id):
#     np.random.seed(1024 + worker_id)

def creat_dirs_for_result(args):

    timestamp = datetime.now().strftime("%m%d%H%M")
    save_dir_name = str(timestamp)+'_'+str(args.model_name)+'_'+str(args.label_name)
    save_dir = os.path.join(args.output_dir, save_dir_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for i in range(args.num_class):
        if args.label_name in ['label1','label2']:
            label_save_dir = os.path.join(save_dir, str(i+1))
        else :
            label_save_dir = os.path.join(save_dir, str(i))
        if not os.path.exists(label_save_dir):
            os.makedirs(label_save_dir, exist_ok=True)
    
    return save_dir
    


def colormap_jet(value, a):
    r = int(min(max(0, 1.5 - abs(value * 4 - 3)), 1)*255)
    g = int(min(max(0, 1.5 - abs(value * 4 - 2)), 1)*255)
    b = int(min(max(0, 1.5 - abs(value * 4 - 1)), 1)*255)
    return (r, g, b, a)

def save_args_json(data, log_dir):

    file = os.path.join(log_dir, 'args.json')
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
 

def assertLevelDownsamples(wsi):
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]
    
    # 如果估算的降采样因子与实际降采样因子不同，将估算的降采样因子添加到列表中；否则，添加实际降采样因子
    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
    
    return level_downsamples

def eval(args, test_set):

    # 1定义dataset
    test_set = WSIDataset(args.fea_dir, args.label_path, preload = False)

    # 2定义dataloader
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)

    # 3定义model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(args.gpus)
    if args.model_name == 'ABMIL':
        model = ABMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'DSMIL':
        i_classifier = FCLayer(in_size=args.feat_size, out_size=args.num_class)
        b_classifier = BClassifier(input_size=args.feat_size, output_class=args.num_class)
        model = MILNet(i_classifier, b_classifier).to(device)
    elif args.model_name == 'CLAM':
        model = CLAM_SB(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    # elif args.model_name == 'PatchGCN':
    #     model = PatchGCN(n_classes=args.num_class).to(device)
    else:
        raise NotImplementedError(f'no model:{args.model_name}')
    
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device)) # 加载模型参数
    save_dir = creat_dirs_for_result(args)
    save_args_json(args.__dict__, save_dir)
    os.makedirs(os.path.join(args.output_dir, 'ABMIL/0'), exist_ok=True)
    a = listdir(os.path.join(args.output_dir, 'ABMIL/0'))
    a = [os.path.splitext(s)[0] for s in a]
    os.makedirs(os.path.join(args.output_dir, 'ABMIL/1'), exist_ok=True)
    b = listdir(os.path.join(args.output_dir, 'ABMIL/1'))
    b = [os.path.splitext(s)[0] for s in b]
    model.eval()
    with torch.no_grad():
        for slide_id, bag, label in test_loader:
            if slide_id[0] in a :
                print(f"{slide_id[0]}continue")
                continue
            if slide_id[0] in b:
                print(f"{slide_id[0]}continue")
                continue
            # 获取attention scores
            slide_id = slide_id[0]
            bag = bag.squeeze(0)
            bag = bag.to(device)
            A = model.get_attention_scores(bag)
            if args.model_name == 'DSMIL':
                A=A[:,label]
                A = [value.item() for value in A]
                # A = [np.log10(value) for value in A]                
            if args.model_name == 'ABMIL':
                A = [value.item() for value in A[0]]
                A = [np.log10(value) for value in A]

            # 注意力分数归一化
            min_val = min(A)
            max_val = max(A)
            A = [(value - min_val) / (max_val - min_val) for value in A] 
            
            # 获取对应坐标等信息
            f = h5py.File(os.path.join(args.fea_dir, slide_id+'.h5'))

            # 获取source data img
            full_path = os.path.join(args.source_dir, slide_id+'.kfb')
            # wsi = openslide.open_slide(full_path)
            wsi = Aslide(full_path)
            level_dim = wsi.level_dimensions
            # img = wsi.read_region((0,0), 0, level_dim[0])
            img = get_aslide_image(wsi, (0,0), 0, level_dim[0])
            down_sample = 16
            img.thumbnail((img.size[0]//down_sample, img.size[1]//down_sample))
            down_patch_size = args.patch_size//down_sample

            # 准备attension Score的img，不是A_color
            A_img = np.zeros(img.size, dtype=np.float32)
            for i in range(len(A)):
                w,h = f['coords'][i] // down_sample
                if 0 <= w < img.size[0] and 0 <= h < img.size[1]:
                    A_img[h:h+down_patch_size,w:w+down_patch_size] = A[i]
                else:
                    print(f"Warning: Coordinates ({w}, {h}) out of image range.")
            print("A_img : ",len(A_img))
            
            # 添加断言以确保图像和热图维度一致
            assert img.size[0] == A_img.shape[0] and img.size[1] == A_img.shape[1], "Image and heatmap dimensions do not match."
            
            # np.save('/data1/yhh/gastritis/HEATMAP_RESULT/new/new.npy',A_img)
            # A_img = Image.fromarray((A_img*255).astype(np.uint8))
            # os.makedirs(heatmap_save,exist_ok=True)

            # heatmap_data_with_alpha = np.ma.masked_where(A_img == 2.0, A_img)
            # 添加调试输出，打印图像和热图的维度信息
            print(f"Image dimensions: {img.size}")
            print(f"Origin heatmap: {A_img.shape}")

            fig,ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (10,5))

            # axs0.set_xlabel('X-axis')
            # axs0.set_ylabel('Y-axis')
            # axs0.axis('off')
            # axs1.axis('off')
            
            ax[0].imshow(img)
            # 使用掩码将零值处设为透明
            A_img = np.ma.masked_where(~(A_img != 0), A_img)
            ax[1].imshow(A_img,cmap='coolwarm')

            # if args.label_name in ['label1','label2']:
            #     heatmap_save = os.path.join(save_dir, str(label.item()+1), slide_id+'.png')
            # else :
            #     heatmap_save = os.path.join(save_dir, str(label.item()), slide_id+'.png')
            
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            heatmap_save = os.path.join(args.output_dir, f'ABMIL/0/{slide_id}.png')
            plt.savefig(heatmap_save,dpi=500)
            # plt.show()
            plt.close()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--gpus', type=int, default=7, help="GPU indices ""comma separated, e.g. '0,1' ")
    parser.add_argument('--source_dir', type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/test/slides/', help="dir of source data")
    parser.add_argument('--fea_dir',type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/test/features/256/uni_v1/h5_files/',help='.h5 featrure files extracted by CLAM')
    parser.add_argument('--feat_size', default=1024, type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--patch_size', default=256, type=int, help='Dimension of the feature size [1024]')

    parser.add_argument('--label_path',type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/test/labels/label.csv', help='splitted label file')
    parser.add_argument('--label_name',type=str, default='label0', help='What label to use.')
    parser.add_argument('--pretrained_model',type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/CKPan/patch/256/train_results/ABMIL/model_best.pth', help='best-fold model')

    parser.add_argument('--model_name',type=str, default='ABMIL', help='What model architecture to use.')   # ['ABMIL', 'TransMIL', 'DSMIL', 'CLAM']
    parser.add_argument('--transparency', type=int, default=150, help='0-255')
    parser.add_argument('--output_dir', default='/data2/yhhu/LLB/Data/前列腺癌数据/test/heatmaps', help='Path to experiment output, config, checkpoints, etc.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_parser()#声明中给定了各个参数的默认值
    set_random_seed(args.seed)# 固定随机数
    class_feat ={
                  'label0':2,
                  'label1':3,
                  'label2':3,
                  'label3':4,
                  'label4':4,
                  }
    args.num_class = class_feat[args.label_name]

    feat_name = args.fea_dir.split('/')[-2]
    # args.feat_size = int(feat_name.split('_')[-2])
    # args.patch_size = int(feat_name.split('_')[1])
    print('visualization:', args.label_name, args.model_name, 'feat_name:', feat_name)
    print("label name: ", args.label_name)
    test_set = WSIDataset(args.fea_dir, args.label_path, preload = False)
    eval(args, test_set)

 
