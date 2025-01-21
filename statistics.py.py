# Databricks notebook source
# VA NAII Project CARDINAL Code - Please discuss with Raffi Hagopian and CARDINAL GROUP before reusing code for other projects 
# 3/7/24

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd 
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix

def calculate_acc_ppv_npv_sensi_speci(study_ids, predictions, truth, threshold):
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0
  id_dict = {'TP':[], 'TN': [], 'FP': [], 'FN': []}
  for study_id, pred_val, true_val in zip(study_ids, predictions, truth):
      if pred_val >= threshold and true_val >= threshold:
          true_pos += 1
          id_dict['TP'].append(study_id)
      elif pred_val < threshold and true_val < threshold:
          true_neg += 1
          id_dict['TN'].append(study_id)
      elif pred_val >= threshold and true_val < threshold:
          false_pos += 1
          id_dict['FP'].append(study_id)
      else:
          false_neg += 1
          id_dict['FN'].append(study_id)
  acc = (true_pos + true_neg)/len(predictions)
  ppv = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
  npv = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
  sens = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
  speci = true_neg / (true_neg + false_pos) if (true_neg + false_pos) >0 else 0
  return acc, ppv, npv, sens, speci, id_dict 

  
# Input: Subgroups, IDs, Preds, Truths should be lists of equal length
# Subgroups should be categorial variables
# Output: pandas table where each row is a subgroup and columns will be ()
def stats_by_subgroup(subgroups, study_ids, predictions, truth, threshold):
    return 0
  
def calculate_correlation(predictions, truth):
  print('# of data points:', len(predictions))
  print('Mean predicted:', np.mean(predictions))
  print('Mean ground truth:', np.mean(truth))
  corr, _ = pearsonr(truth, predictions)
  print('Pearsons correlation: %.3f' % corr)
  mse = mean_squared_error(truth, predictions)
  mae = mean_absolute_error(truth, predictions)
  print('Mean Sqaured Error: %.3f' % mse)
  print('Mean Absolute Error: %.3f' % mae)

def cac_confusion_matrix(predictions, truth, row_title='Truth', column_title='Predictions'):
    bins = [-np.inf, 0, 10, 100, 400, 1000, np.inf]
    bin_labels = ['0','1-10', '11-100', '101-400', '401-1000', '>1000']
    truth_binned = pd.cut(truth, bins, labels=bin_labels)
    preds_binned = pd.cut(predictions, bins, labels=bin_labels)
    conf_matrix = confusion_matrix(truth_binned, preds_binned, labels=bin_labels)
    row_names = pd.MultiIndex.from_tuples([(row_title, bin) for bin in bin_labels])
    col_names = pd.MultiIndex.from_tuples([(column_title, bin) for bin in bin_labels])
    conf_df = pd.DataFrame(conf_matrix, index=row_names, columns=col_names)
    #conf_df['Total'] = conf_df.sum(axis=1)
    #conf_df.loc['Total'] = conf_df.sum()
    return conf_df

def plot_scatter_log(predictions, truth, title='Predicted vs. Ground truth', xtitle='Truth (log10 scale)', ytitle='Predictions (log10 scale)'):
    plt.scatter(np.log10([i+1 for i in truth]), np.log10([i+1 for i in predictions]), facecolors='none', edgecolors='blue')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axhline(y = 2, color = 'r', linestyle = '-') 
    plt.axvline(x = 2, color = 'r', linestyle = '-') 
    plt.title(title)
    plt.show()
  
