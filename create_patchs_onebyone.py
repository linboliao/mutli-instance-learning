import openslide
import numpy as np
import argparse
import h5py
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import os
from skimage import io

import sys
sys.path.insert(0, r'/data1/duzhicheng/virtual_stain/CLAM/aslide_copy')
from aslide_copy.aslide import Aslide
def get_aslide_image(wsi, location, patch_level, dimension):
    (W,H) = dimension
    full_image = np.zeros((H,W,3), dtype=np.uint8)
    
    read_size = 20000
    (x0,y0) = location 
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

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
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

def saveStitchesByCoords(img, patch_size, save_paths):
        # 读取坐标
        file = h5py.File(save_paths['coord_save_path'], mode='r')
        coords_list = file['coords'][:]
        half_size = patch_size//2

        # 根据patch画斜线矩形框，生成stitch缩略图
        for i in range(len(coords_list)):
            [x,y] = coords_list[i]
            ImageDraw.Draw(img).rectangle([(x,y), (x+patch_size,y+patch_size)] , fill =None, outline ='black',width =20)
            ImageDraw.Draw(img).line([(x,y+half_size), (x+half_size,y+patch_size)] , fill ='black', width =20)
            ImageDraw.Draw(img).line([(x,y), (x+patch_size,y+patch_size)] , fill ='black', width =20)
            ImageDraw.Draw(img).line([(x+half_size,y), (x+patch_size,y+half_size)] , fill ='black', width =20)

        img.thumbnail((img.size[0]//50, img.size[1]//50))
        img = img.convert('RGB')
        img.save(save_paths['stitch_save_path'])

def saveCoordsAndMask(img, args, save_paths, level_dim, level_downsamples):
    # 注意h,w
    name = os.path.splitext(os.path.basename(save_paths['coord_save_path']))[0]
    patch_size  = args.patch_size
    patch_level = args.patch_level
    [w,h] = level_dim[patch_level]
    print(name, 'size', [w,h])

    num_h = int(h / patch_size)
    num_w = int(w / patch_size)
    pixel_num = patch_size*patch_size
    results= []
    mask = Image.new("L", [w,h], color=0)
    for h_index in range(num_h):
        h_start = h_index*patch_size
        for w_index in range(num_w):
            w_start = w_index*patch_size

            patch = img.crop((w_start,h_start, w_start+patch_size, h_start+patch_size))
            patch = np.array(patch)
            pixel_max = np.max(patch, axis=2)
            pixel_min = np.min(patch, axis=2)
            difference = pixel_max-pixel_min

            index1 = pixel_min < args.min_RGB
            index2 = (difference > args.min_RGB_diffs) & (difference < args.max_RGB_diffs)
            index  = index1 & index2
            
            if np.sum(index)/pixel_num >= args.foreg_ratio:   #tissue占比超过一定值
                results.append([w_start, h_start]) #保存时按[w,h]保存！！！

            patch_mask_array = np.zeros((patch_size,patch_size),dtype=np.uint8)
            patch_mask_array[index] = np.array(255).astype(np.uint8)
            mask.paste(Image.fromarray(patch_mask_array,mode="L"), (w_start, h_start))
    # mask.save(save_paths['mask_save_path'])
    mask.thumbnail((w//50, h//50))
    mask.save(save_paths['mask_thumb_save_path'])

    results = np.array([result for result in results if result is not None])
    print('Extracted {} coordinates'.format(len(results)))
    #保存坐标
    if len(results)>0:
        asset_dict = {'coords' : results}
        attr = {'patch_size' :            patch_size, # To be considered...
                'patch_level' :           patch_level,
                'downsample':             level_downsamples[patch_level],
                'downsampled_level_dim' : tuple(np.array(level_dim[patch_level])),
                'level_dim':              level_dim[patch_level],
                'name':                   name,
                'save_path':              coord_save_dir}
        attr_dict = { 'coords' : attr}
        save_hdf5(save_paths['coord_save_path'], asset_dict, attr_dict, mode='w')

    return len(results)

def assertLevelDownsamples(wsi):
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]
    
    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
    
    return level_downsamples

def seg_and_patch(directories, args):

    slides = sorted(os.listdir(directories['source']), reverse=False)

    slides_done = os.listdir(directories['stitch_save_dir'])
    print('\n{}/{} slides already done'.format(len(slides_done),len(slides)))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(directories['source'], slide))]

    total = len(slides)
    coord_num = [0]*len(slides)
    for i, slide in enumerate(slides):

        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))

        name = os.path.splitext(os.path.basename(slide))[0]
        save_paths = {'coord_save_path'   : os.path.join(directories['coord_save_dir'], name+'.h5'), 
                    'mask_save_path'      : os.path.join(directories['mask_save_dir'], name+'.png'),
                    'mask_thumb_save_path': os.path.join(directories['mask_thumb_save_dir'], name+'.png'),
                    'stitch_save_path'    : os.path.join(directories['stitch_save_dir'], name+'.jpg')} 

        # 根据coord和stitch文件选择跳过（coord后续环节使用，stitch保证coord的正确性）
        if os.path.isfile(save_paths['coord_save_path']) and os.path.isfile(save_paths['stitch_save_path']): 
            print('{} already exist in destination location, skipped'.format(slide))

            file = h5py.File(save_paths['coord_save_path'], mode='r')
            coord_num[i] = len(file['coords'][:])
            csv = {"slide_id":slides, 'coord_num':coord_num}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(directories['save_dir'], 'process_list_autogen{}.csv'.format(args.patch_size)), index=False)

            continue
        else:
            print('processing {}'.format(slide))

        # 读取WSI
        full_path = os.path.join(directories['source'], slide)
        
        # wsi = openslide.open_slide(full_path)
        # img = wsi.read_region((0,0), args.patch_level, wsi.level_dimensions[args.patch_level])
        # img = img.convert('RGB')
        
        wsi = Aslide(full_path) # read_region超过一定大小会Segmentation fault， 大概20000*20000
        img = get_aslide_image(wsi, (0,0), args.patch_level, wsi.level_dimensions[args.patch_level])
        level_downsamples = assertLevelDownsamples(wsi)
        
        cur_corrd_num = saveCoordsAndMask(img, args, save_paths, wsi.level_dimensions, level_downsamples)
        coord_num[i] = cur_corrd_num
        # 根据坐标coords，生成stitch,确保patch切分正确
        saveStitchesByCoords(img, args.patch_size, save_paths)

        csv = {"slide_id":slides, 'coord_num':coord_num}
        csv = pd.DataFrame(csv)
        csv.to_csv(os.path.join(directories['save_dir'], 'process_list_autogen{}.csv'.format(args.patch_size)), index=False)
    
    print('\n\nSuccess!')

#####
parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, default = '/data1/duzhicheng/FFPE-to-HE建模数据集/FFPE-to-HE建模数据集3/HE',help='path to folder containing raw wsi image files')
parser.add_argument('--save_dir', type = str, default = '/data1/duzhicheng/FFPE-to-HE建模数据集/FFPE-to-HE建模数据集3/HE_COORDS_RESULTS_diff_0.3_30', help='directory to save processed data')
parser.add_argument('--patch_size', type = int, default=1024,help='patch_size')
parser.add_argument('--patch_level', type = int, default=0,help='patch_level')
parser.add_argument('--min_RGB', type = int, default=230,help='threshold of min(RGB). bigger background, smaller foreground')
parser.add_argument('--min_RGB_diffs', type = int, default=30,help='')
parser.add_argument('--max_RGB_diffs', type = int, default=256,help='foreground RGB difference should be in [min, max].')
parser.add_argument('--foreg_ratio', type = float, default=0.3,help='threshold of foreground ratio. bigger save, smaller abandon')

if __name__ == '__main__':
    args = parser.parse_args()
    
    coord_save_dir = os.path.join(args.save_dir, 'patch_'+str(args.patch_size))
    mask_save_dir = os.path.join(args.save_dir, 'mask')
    mask_thumb_save_dir = os.path.join(args.save_dir, 'mask_thumb')
    stitch_save_dir = os.path.join(args.save_dir, 'stitch_'+str(args.patch_size))

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

    seg_and_patch(directories, args)
    
