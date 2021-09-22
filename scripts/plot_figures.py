# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:13:39 2019

@author: liled
"""
#%%
import os
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict 
import matplotlib.ticker as ticker
import pickle

#%% 
colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:purple', 
          'tab:red', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:pink']
markers = ['o', 'D', 'h', 's', '<','X', '^', '2', 'x', '.', ',', 'v', '<', '>']
def read_accu_file(job_name_dic, num_batch, runs_dic, accu_log_dir, insert_random = True, keyword = 'mIOU'):
           
   accu_dic = OrderedDict()
   accu_dic_mean = OrderedDict()
   accu_dic_std = OrderedDict()
   for method in job_name_dic:
       accu_dic[method] = OrderedDict()

   for method in job_name_dic:
      for run in runs_dic[method]:
         accu_dic[method][run] = np.zeros((num_batch, ))

         if os.path.exists(os.path.join(accu_log_dir, job_name_dic[method] + '.txt')):
             accu_file_name = os.path.join(accu_log_dir, job_name_dic[method] + '.txt')  
         elif os.path.exists(os.path.join(accu_log_dir, job_name_dic[method] + '_run_{}.txt'.format(run))): 
             accu_file_name = os.path.join(accu_log_dir, job_name_dic[method] + '_run_{}.txt'.format(run))  
     
         with open(accu_file_name, 'r') as f:
             lines = f.readlines()
             n = 0
             for x in lines:
                if n >= num_batch: break
                if keyword in x.strip().split(' '):
                    accu_dic[method][run][n] = float(x.strip().split(':')[1]) * 100
                    n += 1
      concat_array = np.vstack([accu_dic[method][run]] for run in runs_dic[method])
      accu_dic_mean[method] = np.mean(concat_array, axis = 0)
      accu_dic_std[method] = np.std(concat_array, axis = 0)

   for method in job_name_dic:
       # if insert_random and 'Random' not in method:
         if insert_random and method != 'Random':
           if accu_dic_mean[method].size == num_batch:
              accu_dic_mean[method] = np.delete(accu_dic_mean[method], -1)
              accu_dic_std[method] = np.delete(accu_dic_std[method], -1)
           accu_dic_mean[method] = np.insert(accu_dic_mean[method], 0, accu_dic_mean['Random'][0])
           accu_dic_std[method] = np.insert(accu_dic_std[method], 0, accu_dic_std['Random'][0])
   return accu_dic_mean, accu_dic_std


def plot_al_curve(accu_dic_mean, accu_dic_std, colors_dic, markders_dic, line_dic, figsize = [7, 7], 
                  skip_random=True, full_miou=True, leg_size = 16, lab_size=18, with_perc=True, tick_size=16, markersize=1):
    plt.figure('al curve', figsize = figsize) #w,h
    
    x_data = np.arange(num_batch) + 1
    for method, accu in accu_dic_mean.items():
        if skip_random:
            if  'Random' in method:
                continue
       
        plt.errorbar(x_data, accu, accu_dic_std[method], label = method, color = colors_dic[method], marker = markders_dic[method], linestyle = line_dic[method], markersize=markersize)
        #plt.plot(x_data, accu, label = method, color = colors_dic[method], marker = markders_dic[method], linestyle = line_dic[method])
        
        #plt.fill_between(x_data, accu-accu_dic_std[method], accu+accu_dic_std[method], alpha=0.25, facecolor=colors_dic[method])

    
    xticks = []
    for x in x_data:
        xt = dataset_stat.base_k * x //1000
        xticks.append('%dk'%xt)
    plt.xticks(x_data, xticks, fontsize=tick_size)

    plt.yticks(fontsize=tick_size)
    #plt.ylim(33,77)

    if full_miou:
        if dataset_name == 'cityscapes':
            plt.axhline(dataset_stat.full_data_miou, label = '95% Fully-Supervised', linestyle = '--', color = 'tab:gray')
        elif dataset_name == 'pascal_voc_seg':
            plt.axhline(dataset_stat.full_data_miou, label = '95% Fully-Supervised', linestyle = '--', color = 'tab:gray')
    
    plt.legend(loc=0, fontsize=leg_size)
    plt.xlabel('Amount of Clicks', fontsize=lab_size)
    plt.ylabel('mIoU(%)', fontsize=lab_size) 
    plt.grid()

class Dataset_stat(object):
    pass

def get_dataset_stat(dataset_name):
    dataset_stat = Dataset_stat()
    if dataset_name == 'cityscapes':   
        dataset_stat.region_size = 128
        dataset_stat.full_data_miou = 76.48*0.95
        dataset_stat.train_iter = 60000
        dataset_stat.runs = [0]
        dataset_stat.sp_num = 2048
        dataset_stat.reg_num = dataset_stat.sp_num + 100
        dataset_stat.base_k = 100000
    elif dataset_name == 'pascal_voc_seg':
        dataset_stat.region_size = 32
        dataset_stat.full_data_miou = 77.80*0.95
        dataset_stat.train_iter = 9150
        dataset_stat.runs = [0,1,2]
        dataset_stat.sp_num = 200
        dataset_stat.reg_num = dataset_stat.sp_num + 100
        dataset_stat.base_k = 10000
    return dataset_stat

accu_log_dir = '/Users/cailile/Projects/Revisiting-Superpixels-for-Active-Learning/accuracy_log/'
num_batch = 5
#%% selection strategy
dataset_name = 'cityscapes'
#dataset_name = 'pascal_voc_seg'
dataset_stat = get_dataset_stat(dataset_name)   

job_name_dic = OrderedDict()
if dataset_name == 'cityscapes':
 
  
  job_name_dic['Sp+Do+Random'] = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_2148_rt_sp_seeds_ct_rc_uniq_True_v0_run_2'
  job_name_dic['Sp+Do+Uncertainty'] = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_2148_rt_sp_seeds_ct_rc_uniq_True_bal_False_v1_run_2'
  job_name_dic['Sp+Do+ClassBal'] = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_2148_rt_sp_seeds_ct_rc_uniq_True_bal_True_v1_run_2'
  
  job_name_dic['Rec+Pr+Random'] = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_4096_rt_rec_seeds_ct_cc_uniq_False_v0_run_0'
  job_name_dic['Rec+Pr+Uncertainty'] = 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_4096_rt_rec_seeds_ct_cc_uniq_False_bal_False_v1_run_0'
  
  runs_dic = {}
  for method in job_name_dic:
      runs_dic[method] = [0]
  title = '%s_al_curve_2048.pdf'%dataset_name

line_dic = {}
colors_dic = {}
markers_dic = {}
i = 0
for method in job_name_dic.keys():
    
    if 'Sp+Do' in method:
       colors_dic[method] = 'tab:orange'  
    elif 'Rec+Do' in method:  
        colors_dic[method] = 'tab:green'
    elif 'Rec+Pr' in method:
        colors_dic[method] = 'tab:blue'
    
    if 'Random' in method:
       line_dic[method] = 'dotted'
    elif 'Uncertainty' in method:
         line_dic[method] = 'dashed'
    elif 'ClassBal' in method:
         line_dic[method] = 'solid'
    markers_dic[method] = markers[i]
    i += 1

num_batch = 5
accu_dic_mean, accu_dic_std = read_accu_file(job_name_dic, num_batch, runs_dic, accu_log_dir, insert_random=False)  

for reg in ['Sp', 'Rec']:
   for method in ['Uncertainty', 'ClassBal']:
      for cost_type in ['Pr', 'Do']:
        job = reg + '+' + cost_type + '+' + method  
        job_random = reg + '+' + cost_type + '+' + 'Random'  
        if job not in job_name_dic.keys(): continue
        if accu_dic_mean[job].size == num_batch:
            accu_dic_mean[job] = np.delete(accu_dic_mean[job], -1)
            accu_dic_std[job] = np.delete(accu_dic_std[job], -1)
        accu_dic_mean[job] = np.insert(accu_dic_mean[job], 0, accu_dic_mean[job_random][0])
        accu_dic_std[job] = np.insert(accu_dic_std[job], 0, accu_dic_std[job_random][0])

plot_al_curve(accu_dic_mean, accu_dic_std, colors_dic, markers_dic, line_dic, figsize = [7, 7], skip_random=False, full_miou=True, markersize=7)