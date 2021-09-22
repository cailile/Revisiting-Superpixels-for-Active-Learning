# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:16:35 2019

@author: lile
"""
#%%
import os

project_dir = '/raid/scratch/i2r/caill/Projects/Revisiting-Superpixels-for-Active-Learning/'

#dataset_name='pascal_voc_seg' #c_p=157389, c_c=29516, c_p+c_c=186905
dataset_name='cityscapes' #c_p=4756857, c_c=338302, c_p+c_c=5095159

eval_size = {}
eval_size['pascal_voc_seg'] = [513, 513]
eval_size['cityscapes'] = [1025, 2049]

crop_size = {}
crop_size['pascal_voc_seg'] = eval_size['pascal_voc_seg'][1]
crop_size['cityscapes'] = eval_size['cityscapes'][1]

export_size = {}
export_size['pascal_voc_seg'] = eval_size['pascal_voc_seg']
export_size['cityscapes'] = [eval_size['cityscapes'][0]/ 2, eval_size['cityscapes'][1]/ 2]

num_class = {}
num_class['pascal_voc_seg'] = 21
num_class['cityscapes'] = 19

train_crop_size = {}
train_crop_size['pascal_voc_seg'] = 513
train_crop_size['cityscapes'] = 769

if dataset_name == 'pascal_voc_seg':
    train_batch_size = 12
    train_itr=30000
elif dataset_name == 'cityscapes':
    train_batch_size = 4
    train_itr=60000

dataset_to_base_learning_rate = {}
dataset_to_base_learning_rate['pascal_voc_seg'] = 0.007
dataset_to_base_learning_rate['cityscapes'] = 0.007

num_samples = {}
num_samples['pascal_voc_seg'] = 1464
num_samples['cityscapes'] = 2975

train_split='train'

fine_tune_batch_norm = True

bash_dir = 'bash_files'

if not os.path.exists(bash_dir):
  os.makedirs(bash_dir)

ngpu = 1

    
#v0: random baseline
#v1: entropy baseline

model_name_seg='xception_65'

sp_method = 'seeds'
is_bal = True

region_type = 'sp'
region_size = 32 
num_superpixels = 2048 

if region_type == 'rec':
  cost_type = 'cc'
  if cost_type == 'cc':
     is_uniq = False
  else:
     is_uniq = True
elif region_type == 'sp':
  cost_type = 'rc'
  is_uniq = True 

if dataset_name == 'cityscapes':
   base_k = 100000
   devkit_path = './deeplab/datasets/cityscapes/'
   image_folder = devkit_path + 'leftImg8bit'
   list_folder = devkit_path + 'image_list'  
   gt_anno = 'gtFine'
   semantic_segmentation_folder = devkit_path + gt_anno
   semantic_segmentation_folder_region = devkit_path + gt_anno + 'Region'
elif dataset_name == 'pascal_voc_seg':
   base_k = 10000
   devkit_path = './deeplab/datasets/pascal_voc_seg/VOCdevkit/'
   list_folder = devkit_path + 'VOC2012/ImageSets/Segmentation'      
   image_folder = devkit_path + 'VOC2012/JPEGImages'
   semantic_segmentation_folder = devkit_path + 'VOC2012/SegmentationClassRaw'
   semantic_segmentation_folder_region = devkit_path + 'VOC2012/SegmentationClassRawRegion'

num_batch = 5
k_array = tuple([base_k * i for i in range(1,num_batch+1)]) 
k_array = str(k_array).replace(',', '')

runs = [2] 
gpu_id = 5
for version in ['v1']:
  if region_type == 'rec':
    region_num_per_image = int(crop_size[dataset_name]/region_size) * int(crop_size[dataset_name]/region_size)
  elif region_type == 'sp':
    region_num_per_image = num_superpixels + 100    
    
  for run in runs:              
      random_ref_job =  '{}_{}_train_iter_{}_bn_{}_trainbs_{}_crop_{}_nr_{}_rt_{}_{}_ct_{}_uniq_{}_v0_run_{}'.\
                      format(dataset_name, model_name_seg, train_itr, fine_tune_batch_norm, train_batch_size,  train_crop_size[dataset_name], region_num_per_image, region_type, sp_method, cost_type, is_uniq, run)
      if version == 'v0':
          job_name= '{}_{}_train_iter_{}_bn_{}_trainbs_{}_crop_{}_nr_{}_rt_{}_{}_ct_{}_uniq_{}_v0_run_{}'.\
                      format(dataset_name, model_name_seg, train_itr, fine_tune_batch_norm, train_batch_size,  train_crop_size[dataset_name], region_num_per_image, region_type, sp_method, cost_type, is_uniq, run)
      elif version == 'v1':
          job_name='{}_{}_train_iter_{}_bn_{}_trainbs_{}_crop_{}_nr_{}_rt_{}_{}_ct_{}_uniq_{}_bal_{}_{}_run_{}'.\
            format(dataset_name, model_name_seg, train_itr, fine_tune_batch_norm, train_batch_size, train_crop_size[dataset_name],  region_num_per_image, region_type, sp_method, cost_type, is_uniq,
                  is_bal, version, run) 
                  
      with open(os.path.join(bash_dir, job_name + '.sh'), 'w') as f:               
        f.write('project_dir=%s\n'%project_dir)
        f.write('export PYTHONPATH=${project_dir}/deeplab:${project_dir}/slim:${project_dir}/deeplab/datasets:$PYTHONPATH\n')                       
        f.write('export CUDA_VISIBLE_DEVICES=%d\n'%gpu_id)
                    
        f.write('dataset_name={}\n'.format(dataset_name))
        f.write('model_name_seg={}\n'.format(model_name_seg))
      
        f.write('region_num_per_image={}\n'.format(region_num_per_image))
        f.write('region_size={}\n'.format(region_size))
        f.write('train_itr={}\n'.format(train_itr))

        f.write('num_batch={}\n'.format(num_batch))
        f.write('base_learning_rate={}\n'.format(dataset_to_base_learning_rate[dataset_name]))
                        
        f.write('train_split={}\n'.format(train_split))
        
        f.write('job_name={}\n'.format(job_name))
        f.write('random_ref_job={}\n'.format(random_ref_job))
      
        f.write('crop_size={}\n'.format(crop_size[dataset_name]))

        f.write('k_array={}\n'.format(k_array))
        
        f.write('region_idx_dir=./region_index/$job_name\n')
        f.write('mkdir -p $region_idx_dir\n') 
        
        f.write('reg_idx_dir_ref=./region_index/$random_ref_job\n')

        if region_type == 'rec':           
            f.write('anno_cost_dir=./anno_cost/$dataset_name/rs_%d/\n'%(region_size))   
            f.write('valid_idx_dir=./rectangles/$dataset_name/rs_$region_size\n')
        elif region_type == 'sp':
            f.write('anno_cost_dir=None\n')
            f.write('valid_idx_dir=./superpixels/$dataset_name/%s_%d/train/label\n'%(sp_method, num_superpixels))
            
        f.write('devkit_path=%s\n'%devkit_path)
        f.write('list_folder=%s\n'%list_folder)
    
        f.write('image_folder=%s\n'%image_folder)
        f.write('semantic_segmentation_folder=%s\n'%semantic_segmentation_folder) 
                                                              
        f.write('PATH_TO_INITIAL_CHECKPOINT=deeplab/models/$model_name_seg/model.ckpt\n')
    
        f.write('eval_data_dir=deeplab/datasets/%s/tfrecord\n'%dataset_name)
        
        f.write('mkdir -p ./accuracy_log\n') 
        f.write('accuracy_log=./accuracy_log/${job_name}.txt\n')
        
        f.write('mkdir -p ./batch_log\n') 
        f.write('batch_log=./batch_log/${job_name}.txt\n')
        
        f.write('mkdir -p ./logs\n') 
        
        f.write('if test -f $batch_log; then\n')
        f.write('   typeset -i start_batch=$(cat $batch_log)\n')
        f.write('   echo start batch log is $start_batch\n')
        f.write('   start_batch=$(( start_batch + 1 ))\n')
        f.write('else\n')
        f.write('   start_batch=0\n')
        f.write('fi\n')
        
       
        
        if version == 'v1':   
            f.write('if [ "$start_batch"  == 0 ]; then\n')
            f.write('    train_dir=./outputs/$job_name/batch_0\n')
            f.write('    mkdir -p $train_dir\n')
            f.write('    train_dir_ref=./outputs/$random_ref_job/batch_0\n')
            f.write('    if [ -d $train_dir_ref ]\n') 
            f.write('    then\n')
            f.write('      cp  $train_dir_ref/frozen_inference_graph.pb $train_dir\n')
            f.write('    fi\n')
                  
            f.write('    reg_idx_dir_ref=./region_index/$random_ref_job\n')
            f.write('    if [ -d $reg_idx_dir_ref ]\n') 
            f.write('    then\n')
            f.write('      cp  $reg_idx_dir_ref/batch_0.pkl $region_idx_dir\n')
            f.write('      cp  $reg_idx_dir_ref/batch_0_selected_idx.pkl $region_idx_dir\n')
            f.write('    fi\n')
            f.write('    start_batch=$(( start_batch + 1 ))\n')
            f.write('fi\n')
                
        f.write('echo start batch now is $start_batch\n')
        
        f.write('for ((batch_id=start_batch;batch_id<num_batch;batch_id++));\n')
        f.write(' do\n')
        f.write('    if [ "$batch_id"  == 0 ]; then\n')
        f.write('       k=${k_array[$batch_id]}\n')
        f.write('    elif [ "$batch_id"  -lt "$num_batch" ]; then\n')
        f.write('       k=$(( k_array[batch_id] - k_array[batch_id-1] ))\n')
        f.write('    fi\n')
            
        if version == 'v0':
            f.write('    python ./scripts/region_selection_using_random.py --region_idx_dir=$region_idx_dir --list_folder=$list_folder \\\n\
                            --region_num_per_image=$region_num_per_image --batch_id=$batch_id --k=$k --train_split=$train_split \\\n\
                            --valid_idx_dir=$valid_idx_dir --anno_cost_dir=$anno_cost_dir \\\n\
                            --cost_type=%s --seed=%d 2>&1 | tee ./logs/${job_name}_${batch_id}_random.log\n \\\\n\
                            \n'%(cost_type, run))                 
        elif version == 'v1':
            f.write('    region_uncert_dir=./region_uncertainty/$job_name/batch_$(( batch_id - 1 ))\n')
            f.write('    mkdir -p $region_uncert_dir\n')
                                
            if is_bal:
                f.write('    class_to_region_idx_path=./class_to_region_idx/$job_name/batch_$(( batch_id - 1 ))/ctr_idx.pkl\n')
              
            f.write('    python scripts/extract_model_predictions.py --dataset_name=$dataset_name --job_name=$job_name \\\n\
                          --batch_id=$(( batch_id - 1 )) --region_type=%s --region_num_per_image=%d --num_superpixels=%d --region_size=%d\\\n\
                          --is_bal=%s --sp_method=%s 2>&1 | tee ./logs/${job_name}_${batch_id}_model_pred.log\n  '%(region_type, region_num_per_image, num_superpixels, region_size, 
                                                                        is_bal, sp_method))

            f.write('    python scripts/region_selection_using_al.py \\\n\
                          --batch_id=$batch_id --list_folder=$list_folder  --region_uncert_dir=$region_uncert_dir \\\n\
                          --region_idx_dir=$region_idx_dir  --k=$k --region_num_per_image=$region_num_per_image --train_split=$train_split \\\n\
                          --region_size=$region_size --valid_idx_dir=$valid_idx_dir --anno_cost_dir=$anno_cost_dir --cost_type=%s \\\n\
                          --is_bal=%s --class_to_region_idx_path=$class_to_region_idx_path 2>&1 | tee ./logs/${job_name}_${batch_id}_al.log\n'%(cost_type, is_bal))
        
        f.write('    echo Generating batch training data...\n')
        f.write('    semantic_segmentation_folder_region=%s/$job_name/batch_$batch_id\n'%semantic_segmentation_folder_region)
        f.write('    mkdir -p $semantic_segmentation_folder_region\n')
        f.write('    tfrecord_dir=./deeplab/datasets/$dataset_name/tfrecord/$job_name/batch_$batch_id\n')
        f.write('    mkdir -p $tfrecord_dir\n')
        
        f.write('    python ./deeplab/build_data_active_sp.py \\\n\
                                --dataset_name=%s --list_folder=$list_folder --tfrecord_dir=$tfrecord_dir --image_folder=$image_folder \\\n\
                                --semantic_segmentation_folder=$semantic_segmentation_folder \\\n\
                                --semantic_segmentation_folder_region=$semantic_segmentation_folder_region \\\n\
                                --region_idx_dir=$region_idx_dir --valid_idx_dir=$valid_idx_dir --batch_id=$batch_id \\\n\
                                --region_type=%s --train_split=$train_split --is_uniq=%s  \\\n\
                                2>&1 | tee ./logs/${job_name}_${batch_id}_build_data.log \n'%(dataset_name, region_type, is_uniq))
        
        f.write('    train_dir=./outputs/$job_name/batch_$batch_id\n')
        f.write('    mkdir -p $train_dir\n')
        f.write('    echo Active training batch $batch_id ...\n')
        f.write('    python ./deeplab/train.py \\\n\
                        --logtostderr \\\n\
                        --training_number_of_steps=$train_itr \\\n\
                        --base_learning_rate=$base_learning_rate \\\n\
                        --num_clones=%d \\\n\
                        --train_split=$train_split \\\n\
                        --model_variant=$model_name_seg \\\n\
                        --train_crop_size=%d \\\n\
                        --train_crop_size=%d \\\n\
                        --atrous_rates=6 \\\n\
                        --atrous_rates=12 \\\n\
                        --atrous_rates=18 \\\n\
                        --output_stride=16 \\\n\
                        --decoder_output_stride=4 \\\n\
                        --train_batch_size=%d \\\n\
                        --dataset=$dataset_name \\\n\
                        --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \\\n\
                        --train_logdir=$train_dir \\\n\
                        --dataset_dir=$tfrecord_dir \\\n\
                        --fine_tune_batch_norm=%s \\\n\
                        2>&1 | tee ./logs/${job_name}_${batch_id}_train.log\n'%(ngpu, train_crop_size[dataset_name], train_crop_size[dataset_name],
                                                                  train_batch_size, fine_tune_batch_norm))
        
        f.write('    python ./deeplab/export_model.py \\\n\
                        --logtostderr \\\n\
                        --checkpoint_path=$train_dir/model.ckpt-$train_itr \\\n\
                        --export_path=$train_dir/frozen_inference_graph.pb \\\n\
                        --model_variant=$model_name_seg \\\n\
                        --atrous_rates=6 \\\n\
                        --atrous_rates=12 \\\n\
                        --atrous_rates=18 \\\n\
                        --output_stride=16 \\\n\
                        --decoder_output_stride=4 \\\n\
                        --num_classes=%d \\\n\
                        --crop_size=%d \\\n\
                        --crop_size=%d \\\n\
                        --inference_scales=1.0\n'%(num_class[dataset_name], export_size[dataset_name][0], export_size[dataset_name][1]))
                                       
        f.write('    python ./deeplab/eval_once.py \\\n\
                            --checkpoint_path=$train_dir/model.ckpt-$train_itr \\\n\
                            --dataset=%s \\\n\
                            --eval_logdir=$train_dir \\\n\
                            --dataset_dir=$eval_data_dir \\\n\
                            --model_variant=$model_name_seg \\\n\
                            --eval_crop_size=%d \\\n\
                            --eval_crop_size=%d \\\n\
                            --atrous_rates=6 \\\n\
                            --atrous_rates=12 \\\n\
                            --atrous_rates=18 \\\n\
                            --output_stride=16 \\\n\
                            --decoder_output_stride=4 \\\n\
                            --accuracy_log=$accuracy_log \\\n\
                            --batch_log=$batch_log \\\n\
                            --batch_id=$batch_id \n'%(dataset_name, eval_size[dataset_name][0], eval_size[dataset_name][1]))            
        
        f.write('    if [ ! -f $accuracy_log ] \n')  
        f.write('    then\n')
        f.write('       echo training $batch_id is not successful \n')
        f.write('       break \n')
        f.write('    else\n')
        f.write('       rm -r $tfrecord_dir\n')
        f.write('    fi\n')
        
        f.write('done\n')   
                
                