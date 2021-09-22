#%%
import pickle
import os

#%%

dataset_name = 'cityscapes'
#dataset_name = 'pascal_voc_seg'
if dataset_name == 'pascal_voc_seg':
      num_class = 21
elif dataset_name == 'cityscapes':
      num_class = 19

#job_name = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_2148_rt_sp_seeds_ct_rc_uniq_True_bal_True_v1_run_0'
job_name = 'cityscapes_xception_65_train_iter_60000_bn_True_random_False_trainbs_4_crop_769_nr_2148_rt_sp_seedsf_ct_rc_uniq_True_dorm_True_bal_True_uv_v1_lambdae_1.0_lambdaf_0.0_v1_server_True_run_0'
class_to_region_idx_dir = './class_to_region_idx/%s/batch_0'%job_name

#class_to_region_idx_path = class_to_region_idx_dir + '/%s_%d_0.pkl'%(sp_method, num_superpixels)
class_to_region_idx_path = class_to_region_idx_dir + '/ctr_idx.pkl'

class_to_region_idx = pickle.load(open(class_to_region_idx_path, 'rb'))

region_num_class_pseudo = []
for i in range(num_class):
    region_num_class_pseudo.append(len(class_to_region_idx[i]))