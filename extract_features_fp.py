import sys

sys.path.append('/data2/yhhu/LLB/Code/aslide/')
from aslide import Aslide

import torch
import torch.nn as nn
import torchvision
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, resnet18_baseline
import argparse
from my_utils.utils import print_network, collate_features
from my_utils.file_utils import save_hdf5
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import h5py
import openslide


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:7')


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    x, y = dataset[0]
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


def load_simclr_pretrained_model(model, simclr_save_path):
    # add mlp projection head
    dim_mlp = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    # load simclr pretrained model parameters
    simclr_saved = torch.load(simclr_save_path)
    state_dict = {}
    for key, value in simclr_saved['state_dict'].items():
        new_key = key.replace("backbone.", "")
        state_dict[new_key] = value
    model.load_state_dict(state_dict)
    print('load simclr pretrained model successfully.')
    model.fc = nn.Identity()

    return model


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_slide_dir', type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/HE/wsi/')
parser.add_argument('--slide_ext', type=str, default='.kfb')
parser.add_argument('--csv_path', type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/HE/unpair_patch/256/count.csv')
parser.add_argument('--data_h5_dir', type=str, default='/data2/yhhu/LLB/Data/前列腺癌数据/HE/unpair_patch/256/coord/')
parser.add_argument('--model', type=str, default='resnet50_1024')
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--simclr_save_path', type=str,
                    default='../SimCLR/runs/Aug27_19-37-19_resnet50_2048/checkpoint_0200.pth.tar')
args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    args.feat_dir = '/data3/yhhu/BreastCancer/pix2pixHD-master_du/results_patch2WSI/FFPE2HE1152_du_no_flip/test_100/FEATURES_' + \
                    (args.data_h5_dir).split('_')[-1] + '_by_' + args.model + '_nSLN'

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print('loading model checkpoint')
    print('device:{}'.format(device))
    if args.model == 'resnet18_256':
        model = resnet18_baseline(pretrained=True)
    elif args.model == 'resnet50_1024':
        model = resnet50_baseline(pretrained=True)
    elif args.model == 'resnet18_512':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Identity()
    elif args.model == 'resnet50_2048':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()
    elif args.model == 'simclr_resnet18_512':
        model = torchvision.models.resnet18(pretrained=False, num_classes=128)
        load_simclr_pretrained_model(model, args.simclr_save_path)
    elif args.model == 'simclr_resnet50_1024':
        model = resnet50_baseline(pretrained=False)
        model.fc = nn.Linear(1024, 128)
        load_simclr_pretrained_model(model, args.simclr_save_path)
    elif args.model == 'simclr_resnet50_2048':
        model = torchvision.models.resnet50(pretrained=False, num_classes=128)
        load_simclr_pretrained_model(model, args.simclr_save_path)
        model.fc = nn.Identity()
    model = model.to(device)

    # print_network(model)
    # if torch.cuda.device_count() > 1:
    # 	model = nn.DataParallel(model,device_ids=[0,1])

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()

        if not os.path.exists(slide_file_path):
            print('{} 文件不存在，无法处理')
            continue
        if not os.path.exists(h5_file_path):
            continue
        # wsi = openslide.open_slide(slide_file_path)
        wsi = Aslide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))

    print('\n\nSuccess!')
