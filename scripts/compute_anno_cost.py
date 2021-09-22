import os

import pickle
import numpy as np
import json

from PIL import Image
import scipy.ndimage
import skimage
import skimage.measure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, dest="dataset_name", default='cityscapes')
parser.add_argument('--region_size', type=int, dest="region_size", default=32)

FLAGS = parser.parse_args()

def is_region_boundary(x, y, label_map):
  h,w = label_map.shape
  
  l = label_map[y, x]
  left = [x-1, y]
  right = [x+1, y]
  up = [x, y-1]
  down = [x, y+1]
  
  for p in [left, right, up, down]:
    if 0<=p[0]<w and 0<=p[1]<h:
      if label_map[p[1], p[0]] != l:
         return True   
  return False  

def is_vert_intersect_point(x, y, label_map):
  h,w = label_map.shape
  
  l = label_map[y, x]
  left = [x-1, y]
  right = [x+1, y]
  up = [x, y-1]
  down = [x, y+1]
  
  for p in [down]:
    if 0<=p[0]<w and 0<=p[1]<h:
      if label_map[p[1], p[0]] != l:
         return True   
  return False 

def is_hori_intersect_point(x, y, label_map):
  h,w = label_map.shape
  
  l = label_map[y, x]
  left = [x-1, y]
  right = [x+1, y]
  up = [x, y-1]
  down = [x, y+1]
  
  for p in [right]:
    if 0<=p[0]<w and 0<=p[1]<h:
      if label_map[p[1], p[0]] != l:
         return True   
  return False 

        
def compute_rec_anno_cost(image_name_label_path, image_name_path, image_name_seglabel_path, crop_size):

    valid_idx_dir = './rectangles/{}/rs_{}/'.format(FLAGS.dataset_name, FLAGS.region_size)
  
    anno_cost_dir = './anno_cost/{}/rs_{}/'.format(FLAGS.dataset_name, FLAGS.region_size)
    if not os.path.exists(anno_cost_dir):
       os.makedirs(anno_cost_dir)
    
    n = 0  

    region_num_per_image = int(crop_size/FLAGS.region_size) * int(crop_size/FLAGS.region_size)
    total_cc = 0
    total_ci = 0
    total_cp = 0
    for image_name, label_path in image_name_label_path.items():
        print(image_name)
        image = Image.open(image_name_path[image_name])
        w, h = image.size
       
        rec_dic = pickle.load(open(os.path.join(valid_idx_dir, image_name + '.pkl'), 'rb'))
        region_boxes = rec_dic['boxes']
        valid_idxes = rec_dic['valid_idxes']
        anno_cost = np.zeros((region_num_per_image, ), dtype = np.float32)
        
        seglabel_path = image_name_seglabel_path[image_name]
        seg_array = np.array(Image.open(seglabel_path)) 
        
        labels = np.zeros((h,w), dtype = np.int32)
        
        c_c = 0
        c_i = 0
        c_p = 0
        for j in valid_idxes:
              ymin = region_boxes[j, 0]
              xmin = region_boxes[j, 1]
              ymax = region_boxes[j, 2] 
              xmax = region_boxes[j, 3] 
              labels[ymin:ymax+1, xmin:xmax+1] = j
              
              #c_c
              _, num = skimage.measure.label(seg_array[ymin:ymax+1, xmin:xmax+1], background=254, return_num=True, connectivity=2)
              anno_cost[j] += num
              c_c += num
              
              #c_i
              for x in [xmin, xmax]:
                 for y in range(ymin+1, ymax):
                   if is_vert_intersect_point(x, y, seg_array):
                      anno_cost[j] += 1
                      c_i += 1
                      
              for y in [ymin, ymax]:
                  for x in range(xmin+1, xmax):
                   if is_hori_intersect_point(x, y, seg_array):
                      anno_cost[j] += 1
                      c_i += 1
        
        #c_p   
        with open(label_path, 'r') as f:
            jsonText = f.read()
            anno = json.loads(jsonText)            
            for item in anno['objects']:
                polygon = item['polygon']
                for point in polygon:   
                                
                    x = int(point[0])
                    y = int(point[1])
                    
                    if x >= w or y >= h or x < 0 or y < 0:
                      continue
                    if is_region_boundary(x, y, labels):
                      continue
                    
                    anno_cost[labels[y, x]] += 1
                    c_p += 1  
        
        n += 1
        print('c_c is %d, c_i is %d, c_p is %d\n'%(c_c, c_i, c_p))
        total_cc += c_c
        total_ci += c_i
        total_cp += c_p
        pickle.dump(anno_cost, open(os.path.join(anno_cost_dir, image_name + '.pkl'), 'wb'))
        
    print('total_cc is %d, total_ci is %d, total_cp is %d\n'%(total_cc, total_ci, total_cp))
    time_log = './anno_cost_rec_{}_rs_{}.txt'.format(FLAGS.dataset_name, FLAGS.region_size)
  
    with open(time_log,"w") as f:
      f.write('total_cc is %d, total_ci is %d, total_cp is %d\n'%(total_cc, total_ci, total_cp))

def compute_img_anno_cost(image_name_label_path, image_name_seglabel_path):
    c_p = 0 
    c_c = 0
    for image_name, label_path in image_name_label_path.items():
        
        seg_label_path = image_name_seglabel_path[image_name]  
        seg_array = np.array(Image.open(seg_label_path))
        
        _, num = skimage.measure.label(seg_array, background=254, return_num=True, connectivity=2)

        c_c += num
        with open(label_path, 'r') as f:
            jsonText = f.read()
            anno = json.loads(jsonText)
            for item in anno['objects']:
                  polygon = item['polygon']
                  for point in polygon:
                    c_p += 1
    print(c_p)
    print(c_c)              
        
if __name__ == '__main__':
    if FLAGS.dataset_name == 'pascal_voc_seg':
      devkit_path = './deeplab/datasets/pascal_voc_seg/VOCdevkit/'
      image_dir = devkit_path + 'VOC2012/JPEGImages'
      label_dir = devkit_path + 'VOC2012/SegmentationJSON'
      imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/train.txt'
      semantic_segmentation_folder = devkit_path + 'VOC2012/SegmentationClassRaw'
      crop_size = 513
    elif FLAGS.dataset_name == 'cityscapes':
      gt_anno = 'gtFine'
      devkit_path = './deeplab/datasets/cityscapes/'
      image_dir = devkit_path + 'leftImg8bit/'
      label_dir = devkit_path + '%s/'%gt_anno
      imageset_path = devkit_path + 'image_list/train.txt'  
      crop_size = 2049

    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]
    
    image_name_path = {}
    image_name_label_path = {}
    image_name_seglabel_path = {}
    if FLAGS.dataset_name == 'pascal_voc_seg':
       for image_name in image_list: 
           image_path = os.path.join(image_dir, image_name + '.jpg')
           image_name_path[image_name] = image_path  
           label_path = os.path.join(label_dir, image_name + '.json')
           image_name_label_path[image_name] = label_path
           
           seg_label_path = os.path.join(semantic_segmentation_folder, image_name + '.png')
           image_name_seglabel_path[image_name] = seg_label_path
          
    elif FLAGS.dataset_name == 'cityscapes':
       for image_name in image_list:
          parts = image_name.split("_")
          image_path = os.path.join(image_dir, 'train', parts[0], image_name + '_leftImg8bit.png')
          image_name_path[image_name] = image_path 
              
          label_path = os.path.join(label_dir, 'train', parts[0], image_name + '_%s_polygons.json'%gt_anno)
          image_name_label_path[image_name] = label_path
          
          seglabel_path = os.path.join(label_dir, 'train', parts[0], image_name + '_%s_labelTrainIds.png'%gt_anno)
          image_name_seglabel_path[image_name] = seglabel_path
          
    
    #compute_img_anno_cost(image_name_label_path, image_name_seglabel_path)
    
    compute_rec_anno_cost(image_name_label_path, image_name_path, image_name_seglabel_path, crop_size)
   

