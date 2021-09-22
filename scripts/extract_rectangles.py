
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:

import numpy as np
import os
from PIL import Image
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cityscapes')
parser.add_argument('--split', type=str, dest="split", default='train')
parser.add_argument('--region_size', type=int, dest="region_size", default=32)
FLAGS = parser.parse_args()

def get_boxes(crop_size, base_anchor_size):    
    anchor_stride = base_anchor_size
    
    grid_height = int(crop_size / anchor_stride)
    grid_width = int(crop_size / anchor_stride)
    
    scales = [1.0]
    aspect_ratios = [1.0]
    
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    scales_grid = np.reshape(scales_grid, [-1])
    aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
        
    ratio_sqrts = np.sqrt(aspect_ratios_grid)
    heights = scales_grid / ratio_sqrts * base_anchor_size
    widths = scales_grid * ratio_sqrts * base_anchor_size
      
    
    y_min = np.array(range(grid_height))
    y_min = y_min * anchor_stride
    x_min = np.array(range(grid_width))
    x_min = x_min * anchor_stride
    x_min, y_min = np.meshgrid(x_min, y_min)
    
    widths_grid, x_min_grid = np.meshgrid(widths, x_min)
    heights_grid, y_min_grid = np.meshgrid(heights, y_min)
    
    bbox_min = np.stack([y_min_grid, x_min_grid], axis=2)
    bbox_sizes = np.stack([heights_grid, widths_grid], axis=2)
    
    bbox_min = np.reshape(bbox_min, [-1, 2])
    bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
    
    #ymin, xmin, ymax, xmax
    bbox_corners = np.concatenate([bbox_min, bbox_min + bbox_sizes - 1], axis=1)
    bbox_corners = bbox_corners.astype(np.float32)
    return bbox_corners
  
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  image_array = np.array(image.getdata())
  if image_array.size == im_width * im_height:
     image_array = image_array.reshape((im_height, im_width, 1))
     image_array = np.repeat(image_array, 3, axis = 2)
  else:
     image_array = image_array.reshape((im_height, im_width, 3))

  return image_array.astype(np.uint8)      

if __name__ == '__main__':
    dataset_name = FLAGS.dataset_name
    
    if dataset_name == 'pascal_voc_seg':
      devkit_path = 'deeplab/datasets/pascal_voc_seg/VOCdevkit/'
      image_dir = devkit_path + 'VOC2012/JPEGImages'
      imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/train.txt'
      crop_size = 513
    elif dataset_name == 'cityscapes':
      devkit_path = './deeplab/datasets/cityscapes/'
      image_dir = devkit_path + 'leftImg8bit/'
      imageset_path = devkit_path + 'image_list/train.txt'  
      crop_size = 2049
      
    with open(imageset_path, 'r') as f:
        lines = f.readlines()
    image_list = [x.strip() for x in lines]
    
    image_name_path = {}
    if dataset_name == 'pascal_voc_seg':
       for image_name in image_list: 
           image_path = os.path.join(image_dir, image_name + '.jpg')
           image_name_path[image_name] = image_path   
    elif dataset_name == 'cityscapes':
        for image_name in image_list:
           parts = image_name.split("_")
           image_path = os.path.join(image_dir, 'train', parts[0], image_name + '_leftImg8bit.png')
           image_name_path[image_name] = image_path   
    
    count = 0
    
    base_anchor_size = FLAGS.region_size
    valid_idx_dir = os.path.join('rectangles', dataset_name, 'rs_%d'%base_anchor_size)
    
    if not os.path.exists(valid_idx_dir):
        os.makedirs(valid_idx_dir)
    
    bbox_corners = get_boxes(crop_size, base_anchor_size)
    
    region_num_per_image = bbox_corners.shape[0]
    
    for image_name, image_path in image_name_path.items():
        print('image {}'.format(count))
        image = Image.open(image_path)
        image_width, image_height = image.size
          
        boxes = np.zeros((region_num_per_image, 4), dtype = np.int16)       
        
        valid_idxes = []
        for i in range(region_num_per_image):
            ymin, xmin, ymax, xmax = bbox_corners[i, :]
            if ymin < image_height and xmin < image_width: #overlapping with image region
                ymax = np.clip(ymax, 0, image_height - 1)
                xmax = np.clip(xmax, 0, image_width - 1)
                
                boxes[i, :] = np.array([int(ymin), int(xmin), int(ymax), int(xmax)])
                
                valid_idxes.append(i)
       
        output_dict = {}        
        output_dict['boxes'] = boxes
        output_dict['valid_idxes'] = valid_idxes
          
        pickle.dump(output_dict, open(os.path.join(valid_idx_dir, image_name + '.pkl'), 'wb'))
        count += 1