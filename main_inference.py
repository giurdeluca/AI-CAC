# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import csv
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

# TO DO: ensure this works if gpu_device is a list!

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
#SCORE_FILE = config.score_file
#FILTERED_FILE = config.filtered_file
#MASK_FOLDER = config.mask_folder
GPU_DEVICE = config.gpu_device
OUTPUT_DIR = config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCORE_FILE = os.path.join(OUTPUT_DIR, 'scores.csv')
FILTERED_FILE = os.path.join(OUTPUT_DIR, 'one_series_per_study_df.csv')

#dicom_df = create_series_df_parallel(DICOM_ROOT_DIR)
#print('DCM DF Created')

# Check if processed file already exists

if os.path.exists(FILTERED_FILE):
    # Load existing file
    one_series_per_study_df = pd.read_csv(FILTERED_FILE)
    print('Loaded existing one_series file')
else:
    # Create new file
    print('Creating new one_series file')
    one_series_per_study_df = optimized_dicom_processing(DICOM_ROOT_DIR, max_workers=4, force_sequential=True)
    one_series_per_study_df.to_csv(FILTERED_FILE, index=False)
    print('DCM DF Filtered')


# Check what's already been processed
processed_studies = set()
if os.path.exists(SCORE_FILE):
    existing_df = pd.read_csv(SCORE_FILE)
    processed_studies = set(existing_df['StudyName'].tolist())
    file_mode = 'a'  # append mode
    write_header = False
else:
    file_mode = 'w'  # write mode
    write_header = True
# Filter out already processed studies
one_series_per_study_df = one_series_per_study_df[~one_series_per_study_df['StudyName'].isin(processed_studies)]
one_series_per_study_df = one_series_per_study_df.reset_index(drop=True)
print(f"Number of studies to process: {len(one_series_per_study_df)}")

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

device = torch.device(f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu")
print(device)

model = SwinUNETR(
    spatial_dims=2,
    img_size=RESAMPLE_IMAGE_SIZE, 
    in_channels=1,
    out_channels=1,
    feature_size= 96,
    use_checkpoint=True,
    drop_rate=0.2,
)

model = nn.DataParallel(model, device_ids=[GPU_DEVICE])

checkpoint = torch.load(MODEL_CHECKPOINT_FILE, map_location=device) 
model.load_state_dict(checkpoint['model_state_dict'])


model.to(device)

# Prepare metadata for efficient merging
study_metadata = one_series_per_study_df.groupby('StudyName').first().reset_index()
metadata_dict = study_metadata.set_index('StudyName').to_dict('index')

# Define final column structure (StudyName, AI-CAC, then all metadata columns)
final_columns = ['StudyName', 'AI-CAC'] + [col for col in study_metadata.columns if col != 'StudyName']

# Initialize CSV file with headers
# Conditional header writing
if write_header:
    with open(SCORE_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_columns)
        writer.writeheader()

with open(SCORE_FILE, 'a', newline='') as csvfile: # Always append from here
  writer = csv.DictWriter(csvfile, fieldnames=final_columns)
  i=0
  model.eval()
  with torch.no_grad():
      for study_id, inputs, targets, hu_vols, vox_dims in input_loader:
        i += 1
        try:
          assert inputs.shape[4] > 0, "No slices in volume"
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
          # Create row with AI-CAC result
          row = {'StudyName': study_id, 'AI-CAC': pred_cacs[0]}
          # Merge with metadata
          if study_id in metadata_dict:
              row.update(metadata_dict[study_id])
          # Write immediately
          writer.writerow(row)

          if SAVE_MASKS:
              save_vol_masks(inputs.cpu().squeeze(), pred_vol.cpu().squeeze(), os.path.join(OUTPUT_DIR, study_id))
          if VISUALIZE_RESULTS: 
              draw_first_positive(inputs.cpu(), pred_vol.cpu(), pred_vol.cpu(),0)

        except (IndexError, AssertionError, RuntimeError) as e:
            # Handle expected processing errors
            error_msg = f"Processing failed for {study_id}: {str(e)}"
            print(error_msg)
            # Write failure row and continue
        except Exception as e:
            # Log unexpected errors but continue
            error_msg = f"Unexpected error for {study_id}: {str(e)}"
            print(error_msg)
            # Could also write to separate error log