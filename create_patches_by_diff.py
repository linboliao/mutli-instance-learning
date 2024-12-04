# encoding: utf-8
import openslide
import numpy as np
import cv2
import argparse
import os
import h5py
from PIL import Image, ImageDraw
import math
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]  # 每个块的形状
            maxshape = (None,) + data_shape[1:]  # 不限制个数
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def saveStitchesByCoords(img, patch_size, coord_save_path, stitch_save_path):
    # 读取坐标
    file = h5py.File(coord_save_path, mode='r')
    coords_list = file['coords'][:]
    half_size = patch_size // 2

    # 根据patch画斜线矩形框，生成stitch缩略图
    for i in range(len(coords_list)):
        [x, y] = coords_list[i]
        ImageDraw.Draw(img).rectangle([(x, y), (x + patch_size, y + patch_size)], fill=None, outline='black', width=20)
        ImageDraw.Draw(img).line([(x, y + half_size), (x + half_size, y + patch_size)], fill='black', width=20)
        ImageDraw.Draw(img).line([(x, y), (x + patch_size, y + patch_size)], fill='black', width=20)
        ImageDraw.Draw(img).line([(x + half_size, y), (x + patch_size, y + half_size)], fill='black', width=20)

    img.thumbnail((img.size[0] // 10, img.size[1] // 10))
    img = img.convert('RGB')
    img.save(stitch_save_path)


# 计算并得到符合条件（mask中每个patch，如果其中255的数量占比高于某一阈值）的坐标列表，然后对坐标和附带的属性信息进行保存
def saveCoordsByMask(mask, patch_size, coord_save_path, name, level_dim, level_downsamples, patch_level):
    # 注意h,w
    [h, w] = mask.shape
    num_h = math.floor(h / patch_size)
    num_w = math.floor(w / patch_size)

    results = []
    pixel_num = patch_size * patch_size
    for h_index in range(num_h):

        h_start = h_index * patch_size
        for w_index in range(num_w):

            w_start = w_index * patch_size
            patch = mask[h_start: h_start + patch_size, w_start: w_start + patch_size]
            if np.sum(patch > 0) / pixel_num >= 0.3:  # tissue占比超过0.3
                results.append([w_start, h_start])  # 保存时按[w,h]保存！！！

    results = np.array([result for result in results if result is not None])
    print('Extracted {} coordinates'.format(len(results)))
    # 保存坐标
    if len(results) > 0:
        asset_dict = {'coords': results}
        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
                'level_dim': level_dim[patch_level],
                'name': name,
                'save_path': coord_save_dir}
        attr_dict = {'coords': attr}
        save_hdf5(coord_save_path, asset_dict, attr_dict, mode='w')


# img_array H W C
# 将通道部分分布差异大的部分进行突出显示
# return H W mask数据
def getTissueMask(img_array, mask_save_path, threshold=10):
    if os.path.isfile(mask_save_path):
        # flags默认为1，读取彩色图像，为0读取灰度图, 为-1读取图像原通道数
        mask = cv2.imread(mask_save_path, flags=0) / 255
    else:
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.int8)
        pixel_max = np.max(img_array, axis=2)
        pixel_min = np.min(img_array, axis=2)
        difference = pixel_max - pixel_min
        index = np.where(difference > threshold)
        mask[index] = 1
        # cv2.imwrite(mask_save_path, mask * 255)

    return mask


# 可能原本数据有错误，这里对每层的下采样因子进行计算并判断
# return list(tuple(x,x))---> 每层下采样因子元组构成的列表
def assertLevelDownsamples(wsi):
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]

    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))  # 重复

    return level_downsamples


def seg_and_patch(directories, patch_size, patch_level):

    slides = sorted(os.listdir(directories['source']),reverse=False) # reverse默认是false，意思按升序排列
    #slides = slides[130:]
    slides_done = os.listdir(directories['stitch_save_dir']) # 得到的是已经处理的文件名列表
    print('\n{}/{} slides already done'.format(len(slides_done), len(slides))) 
    #abnormal_list =['B2023-18556.tif','B2022-6414.tif','B2021-02459.tif','B2021-17056.tif','B2020-12441.tif', 'B2020-9888.tif']
    slides = [slide for slide in slides if os.path.isfile(os.path.join(directories['source'], slide))] # 根据有序文件名列表得到文件的绝对路径列表

    total = len(slides)
    coord_num = [0]*len(slides)
    for i, slide in enumerate(slides):

        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        name = os.path.splitext(os.path.basename(slide))[0]  # 索引值

        save_paths = {'coord_save_path'   : os.path.join(directories['coord_save_dir'], name+'.h5'), 
                    'mask_save_path'      : os.path.join(directories['mask_save_dir'], name+'.jpg'),
                    'mask_thumb_save_path': os.path.join(directories['mask_thumb_save_dir'], name+'.jpg'),
                    'stitch_save_path'    : os.path.join(directories['stitch_save_dir'], name+'.jpg')} 

        # 根据coord和stitch文件选择跳过（coord后续环节使用，stitch保证coord的正确性）
        if os.path.isfile(save_paths['coord_save_path']) and os.path.isfile(save_paths['stitch_save_path']):
            print('{} already exist in destination location, skipped'.format(slide))

            file = h5py.File(save_paths['coord_save_path'], mode='r')
            coord_num[i] = len(file['coords'][:])
            csv = {"slide_id":slides, 'coord_num':coord_num}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(directories['save_dir'], 'process_list_autogen_2.csv'), index=False)

            continue
        else:
            print('processing {}'.format(slide))
        try:
            # 读取WSI
            full_path = os.path.join(directories['source'], slide)
            wsi = openslide.open_slide(full_path)
            level_downsamples = assertLevelDownsamples(wsi) # 获取真实且正确的下采样因子
            level_dim = wsi.level_dimensions
            # openslide直接read_region读level_dim[1]有问题，这里采取另外一种办法
            img = wsi.read_region((0, 0), patch_level, wsi.level_dimensions[patch_level])
            img = img.convert('RGB')
            # img.thumbnail(size=level_dim[patch_level])  # 利用原图按照指定大小生成缩略图
            # img转numpy 原先w,h 变为h,w 使用时注意。cv2读image也是如此！读出来就是numpy
            img_array = np.array(img)

            # 保存并提取wsi mask
            mask = getTissueMask(img_array, save_paths['mask_save_path'])
            # 根据mask切分patch, 保存坐标coords
            saveCoordsByMask(mask, patch_size, save_paths['coord_save_path'], name, level_dim, level_downsamples, patch_level)
            # 根据坐标coords，生成stitch,确保patch切分正确
            saveStitchesByCoords(img, patch_size, save_paths['coord_save_path'], save_paths['stitch_save_path'])
            
            csv = {"slide_id": slides, 'coord_num':coord_num}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(directories['save_dir'], 'process_list_autogen_2.csv'), index=False)
        except:
            print('暂时无法处理：{} 切片'.format(slide))

    print('\n\nSuccess!')



parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, default='/data3/yhhu/BreastCancer/非前哨淋巴结预测/postive_tif2',
                    help='path to folder containing raw wsi image files')
parser.add_argument('--patch_size', type=int, default=256,
                    help='patch_size')
parser.add_argument('--patch_level', type=int, default=0,
                    help='patch_level')
parser.add_argument('--save_dir', type=str, default='/data3/yhhu/BreastCancer/pix2pixHD-master_du/results_patch2WSI/FFPE2HE1152_du_no_flip/test_100/COORDS_RESULTS_diff_nSLN',
                    help='directory to save processed data')

if __name__ == '__main__':
    args = parser.parse_args()

    coord_save_dir = os.path.join(args.save_dir, 'patches_' + str(args.patch_size))
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    mask_thumb_save_dir = os.path.join(args.save_dir, 'masks_thumb')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches_' + str(args.patch_size))

    directories = {'source'             : args.source, 
				   'save_dir'           : args.save_dir,
				   'coord_save_dir'     : coord_save_dir, 
				   'mask_save_dir'      : mask_save_dir,
                   'mask_thumb_save_dir': mask_thumb_save_dir,
                   'stitch_save_dir'    : stitch_save_dir} 

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_and_patch(directories, args.patch_size, args.patch_level)
