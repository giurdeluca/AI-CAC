# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import random
import pydicom
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai 
from monai.networks.nets import SwinUNETR
 
from filter_series import * 
from dataset_generator_inference import * 
from processing import *
from visualization import *
from config import load_config

config = load_config('config.json')

NUM_WORKERS = config.num_workers 
BATCH_SIZE = config.batch_size
RESAMPLE_IMAGE_SIZE = config.resample_image_size
RESAMPLE_SHAPE = config.resample_shape
ZOOM_FACTORS = config.zoom_factors

SAVE_MASKS = config.save_masks # Save PNGs of AI-CAC segmentations
VISUALIZE_RESULTS = config.visualize_results # Display segmentation masks during inference

DICOM_ROOT_DIR = config.dicom_root_dir
MODEL_CHECKPOINT_FILE = config.model_checkpoint_file
SCORE_FILE = config.score_file
MASK_FOLDER = config.mask_folder

score_data = []
dicom_df = create_dicom_df(DICOM_ROOT_DIR)
print('DCM DF Created')
one_series_per_study_df = filter_dicom_df(dicom_df)
print('DCM DF Filtered')
one_series_per_study_df.to_csv(os.path.join(DICOM_ROOT_DIR, 'dicom_input_one_series.csv'), index=False)

study_files = {}
for index, row in one_series_per_study_df.iterrows():
  study = row['StudyName']
  file_path = row['DICOMFilePath'] 
  axial_cord = float(row['AxialPosition'])
  if study not in study_files:
    study_files[study] = [(file_path, axial_cord)]
  else:
    study_files[study].append((file_path, axial_cord))

study_ids = []
study_paths = []
study_labels = []

for study in study_files.keys():
  study_ids.append(study)
  study_paths.append(study_files[study])
  study_labels.append(-1) #None for now 

input_volume_data = CTChestDataset_nongated(study_ids, study_paths, study_labels, new_shape=RESAMPLE_SHAPE, zoom_factors=ZOOM_FACTORS) # transform was unused, took out
input_loader = DataLoader(input_volume_data, batch_size=1, shuffle=False, num_workers=NUM_WORKERS) #Batch 1 volume at a time for volume loader, will batch slices within volume later 

model = SwinUNETR(
    spatial_dims=2,
    img_size=RESAMPLE_IMAGE_SIZE, 
    in_channels=1,
    out_channels=1,
    feature_size= 96,
    use_checkpoint=True,
    drop_rate=0.2,
)

model = nn.DataParallel(model)

checkpoint = torch.load(MODEL_CHECKPOINT_FILE) 
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

i=0
model.eval()
with torch.no_grad():
    for study_id, inputs, targets, hu_vols, vox_dims in input_loader:
        i += 1
        study_id = study_id[0] # size one batch, first id is only id
        inputs = inputs.to(device)
        pred_vol = torch.zeros(inputs.shape, dtype=torch.float, device=device)
        num_slices = inputs.shape[4]

        for start_idx in range(0, num_slices, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_slices)
            batch = inputs[..., start_idx:end_idx]
            batch = batch.squeeze(0).permute(3,0,1,2)
            batch_out = model(batch.float()) #might need to move batch dimension
            batch_out = batch_out.unsqueeze(0).permute(0,2,3,4,1)
            pred_vol[..., start_idx:end_idx] = batch_out

        pred_cacs = compute_agatston_for_batch(inputs.cpu(), pred_vol.cpu(), vox_dims)
        
        print(i, study_id, pred_cacs[0]) 
        row = {'StudyName': study_id, 'AI-CAC': pred_cacs[0]}
        score_data.append(row)

        if SAVE_MASKS:
            save_vol_masks(inputs.cpu().squeeze(), pred_vol.cpu().squeeze(), os.path.join(MASK_FOLDER, study_id))
        if VISUALIZE_RESULTS: 
            draw_first_positive(inputs.cpu(), pred_vol.cpu(), pred_vol.cpu(),0)

score_df = pd.DataFrame(score_data)
study_metadata = one_series_per_study_df.groupby('StudyName').first().reset_index()
score_df_complete = score_df.merge(study_metadata, how='left', on='StudyName')
score_df_complete.to_csv(config.score_file, index=False)
score_df_complete.to_csv(SCORE_FILE, index=False)
