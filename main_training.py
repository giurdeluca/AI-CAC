# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import random
import pydicom
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import monai 
from monai.losses import FocalLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import RandAffined

from processing import * 
from visualization import * 
from dataset_generator_training import * 
from filter_series import * 

# Configuration Parameters 
VISUALIZE_DURING_TRAINING = False
SAVE_CHECKPOINT_DURING_TRAINING = True
LOAD_PRIOR_CHECKPOINT = False #pickup training from last checkpoint 

NUM_WORKERS = 12 
BATCH_SIZE = 16 
POS_SLICE_UPSAMPLE_FACTOR = 1 # Multiple by which slices containing calcium are upsampled during training
RESAMPLE_IMG_SIZE = (512, 512)
RESAMPLE_SHAPE = (512, 512) 
ZOOM_FACTORS = (1, 1)
LEARNING_RATE = 1e-3
DECAY_WEIGHT = 1e-5
FOCAL_ALPHA = 0.6 
DROPOUT_RATE = 0.2
SCHEDULER_STEP_SIZE = 10 # 20 Number of epochs before dropping learning rate
SCHEDULER_GAMMA = 0.1
INTERVAL_TO_ADD_FALSE_POSITIVE_SLICES = 5
# Validation set takes longer than training sets since it has positive and negative CAC slices. 
#   Don't check Validation set with each epoch to save computation for training. 
#   Also use Modulus+1 offset so that it get checked before add FP slices epochs
INTERVAL_TO_CHECK_VALIDATION_SET_RESULTS = 5  
PRINT_INTERVAL = 10
VISUALIZE_INTERVAL = 40

LOAD_CHECKPOINT_FILE = ''
SAVE_NAME = 'temp_model'
SAVE_CHECKPOINT_FILE = '/path/'+SAVE_NAME+'.pth'

comment = f'==== Dynamic Model Training ====\nBatch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Decay: {DECAY_WEIGHT}, Focal alpha: {FOCAL_ALPHA}, Positive Slice upsample factor: {POS_SLICE_UPSAMPLE_FACTOR}, save: {SAVE_CHECKPOINT_FILE}'
print(comment)

""" SEE IF TRAINING PUBLIC CODE WORKS ON THIS SUBSET OF TRAINING DATA """ 
sample_df = pd.read_csv('/path/to/sample_of_dicom_paths_and_dcm_tags.csv')
sample_labels_dir = '/path/to/niftis.nii.gz' # StudyName.nii.gz
sample_df['StudyName'] = sample_df['study_ID']
sample_df['DICOMFilePath'] = sample_df['file_path'].str.replace('dbfs:/', '/dbfs/')
sample_df['AxialPosition'] = sample_df['ImagePositionPatient'].apply(lambda coord: coord.split(', ')[-1].replace(']', '')) # Take Axial position out of [#, #, #]
sample_df[['StudyName', 'study_ID', 'file_path', 'DICOMFilePath', 'AxialPosition']]
studies = list(sample_df['StudyName'].unique())
random.seed(42)
random.shuffle(studies)
split_index = int(len(studies) * 0.8)
train_studies = studies[:split_index]
valid_studies = studies[split_index:]
train_df = sample_df[sample_df['StudyName'].isin(train_studies)]
valid_df = sample_df[sample_df['StudyName'].isin(valid_studies)]
""" END LOAD TRIAL OF PUBLIC TRAIN DATASET """ 

# Training and Validation Dataset should already be filtered to one series per study - the series that has been already manually segmented for CAC
#train_dicom_dir = '' 
#train_df = create_dicom_df(train_dicom_dir) 
#valid_dicom_dir = ''
#valid_df = create_dicom_df(valid_dicom_dir)
train_seg_dir = sample_labels_dir
valid_seg_dir = sample_labels_dir
 
study_files = {}
train_ids = [] 
valid_ids = [] 

for seg_dir, dicom_df, id_list in [(train_seg_dir, train_df, train_ids), (valid_seg_dir, valid_df, valid_ids)]:  
    for index, row in dicom_df.iterrows():
        study_id = row['StudyName']
        file_path = row['DICOMFilePath']
        axial_cord = float(row['AxialPosition'])
        slice_tuple = (file_path, axial_cord)
        nifti_file = os.path.join(seg_dir, study_id + '.nii.gz')
        if not os.path.exists(nifti_file):
            print(f"Missing Segmentation .nii.gz file: {nifti_file}")
            continue 
        if study_id not in study_files:
            study_files[study_id] = {'seg_dir': seg_dir, 'slices':[slice_tuple]}
            id_list.append(study_id)
        else:
            study_files[study_id]['slices'].append(slice_tuple)

train_volume_data = CTChestDataset_dynamic(study_files, train_ids, RESAMPLE_SHAPE, ZOOM_FACTORS, pos_slice_only=True, pos_slice_upsample_factor=POS_SLICE_UPSAMPLE_FACTOR)
valid_volume_data = CTChestDataset_dynamic(study_files, valid_ids, RESAMPLE_SHAPE, ZOOM_FACTORS, pos_slice_only=False, pos_slice_upsample_factor=POS_SLICE_UPSAMPLE_FACTOR)

print('train study_list length: %s' % len(train_ids))
print('valid study_list length: %s' % len(valid_ids))
print('train batchloader length: %s' % len(train_volume_data))
print('valid batchloader length: %s' % len(valid_volume_data))

train_loader = DataLoader(train_volume_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_volume_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = SwinUNETR(
    spatial_dims=2, 
    img_size=RESAMPLE_IMG_SIZE, 
    in_channels=1,
    out_channels=1,
    feature_size=96,
    use_checkpoint=True,
    drop_rate=DROPOUT_RATE,
)

model = nn.DataParallel(model)

if LOAD_PRIOR_CHECKPOINT:
    checkpoint = torch.load(LOAD_CHECKPOINT_FILE) 
    model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = FocalLoss(alpha=FOCAL_ALPHA)
criterion_score_val = nn.MSELoss() 

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_WEIGHT)
scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

num_epochs = 2000
start_epoch = 1 

if LOAD_PRIOR_CHECKPOINT:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    start_epoch = checkpoint['epoch']

dice_metric_sample_avg = torchmetrics.Dice(average='samples', threshold=0, zero_division=1) # average='samples' Average dice per each batch 2D sample instead of computing DICE over full batch 3D
dice_metric_global = torchmetrics.Dice(threshold=0, zero_division=1) # monai one is calculated per batch item, other is full batch, set dice to 1 if negative seg TP+FP+FN all zero

rand_affine = RandAffined(
    keys=["image", "label", "hu"],
    mode=("bilinear", "nearest", "bilinear"),
    prob=0.5,
    spatial_size=None, 
    translate_range=(50, 50), 
    rotate_range=(np.pi/18, np.pi/18),
    scale_range=(0.1, 0.1),
    padding_mode="border",
)
rand_affine.set_random_state(seed=123)

print('trainable_params: %s\n' % sum(p.numel() for p in model.parameters() if p.requires_grad))

add_negatives_step_counter = 0 
for epoch in range(start_epoch, num_epochs):
    if epoch % INTERVAL_TO_ADD_FALSE_POSITIVE_SLICES == 0:  
        add_negatives_step_counter += 1
        # Reset back to just positive slices after 1st set of FP add back epochs as initial epochs will have lots of False positive non-CAC containing slices after first add back 
        if add_negatives_step_counter == 2: 
            print('reseting training dataset to just CAC containing slices')
            train_volume_data = CTChestDataset_dynamic(study_files, train_ids, RESAMPLE_SHAPE, ZOOM_FACTORS, pos_slice_only=True, pos_slice_upsample_factor=POS_SLICE_UPSAMPLE_FACTOR)
        print('updating training dataset with false positive negative (no true cac) slices')
        train_volume_data.add_incorrect_negative_slices(model, 32)
        train_loader = DataLoader(train_volume_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_dice_sample_avg = 0.0
    i = 0 
    for inputs, targets, hu_vols, vox_dims in train_loader:
        transformed = rand_affine({"image":inputs.squeeze(1), "label":targets.squeeze(1), "hu": hu_vols.squeeze(1)}) #() to (1) 
        inputs, targets, hu_vols = transformed["image"].unsqueeze(1), transformed["label"].unsqueeze(1), transformed["hu"].unsqueeze(1)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        batch_dice = dice_metric_global(outputs.cpu(),targets.cpu()>0).item()
        train_dice += batch_dice
        batch_dice_sample_avg = dice_metric_sample_avg(outputs.cpu(),targets.cpu()>0).item()
        train_dice_sample_avg += batch_dice_sample_avg
        
        i += 1
        if i % PRINT_INTERVAL == 0: 
            true_pix_count = torch.sum(targets>0)
            pred_pix_count = torch.sum(outputs>0)
            pix_dif = pred_pix_count - true_pix_count
            print(f'Epoch [{epoch}/{num_epochs}], Train Step [{i}/{len(train_loader)}], Loss: {loss.item():.9f}, Global batch DCE: {batch_dice:.2f}, Sample avg DCE: {batch_dice_sample_avg:.2f}, True Pix: {true_pix_count}, Pred Pix: {pred_pix_count}, Pix Dif:{pix_dif}')

        if VISUALIZE_DURING_TRAINING and i % VISUALIZE_INTERVAL == 0:
            draw_first_positive_2d_in_batch(inputs.cpu(), outputs.cpu(), targets.cpu())
 
    train_loss /= i
    train_dice /= i
    train_dice_sample_avg /= i 

    val_loss = 0.0
    val_dice = 0.0
    val_dice_sample_avg = 0.0
    j = 0 
    with torch.no_grad():
        if (epoch + 1) % INTERVAL_TO_CHECK_VALIDATION_SET_RESULTS == 0: # Add offset one so that validation set is checked before adding incorrect slices steps if both intervals have the same value
            for inputs, targets, hu_vols, vox_dims in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                inputs = inputs.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            
                batch_dice = dice_metric_global(outputs.cpu(),targets.cpu()>0).item()
                val_dice += batch_dice
                batch_dice_sample_avg = dice_metric_sample_avg(outputs.cpu(),targets.cpu()>0).item()
                val_dice_sample_avg += batch_dice_sample_avg
            
                j += 1
                if j % PRINT_INTERVAL == 0: 
                    true_pix_count = torch.sum(targets>0)
                    pred_pix_count = torch.sum(outputs>0)
                    pix_dif = pred_pix_count - true_pix_count
                    print(f'Epoch [{epoch}/{num_epochs}], Val Step [{j}/{len(valid_loader)}], Loss: {loss.item():.9f}, Global batch DCE: {batch_dice:.2f}, Sample avg DCE: {batch_dice_sample_avg:.2f}, True Pix: {true_pix_count}, Pred Pix: {pred_pix_count}, Pix Dif:{pix_dif}')

                if VISUALIZE_DURING_TRAINING and j % VISUALIZE_INTERVAL == 0:
                    draw_first_positive_2d_in_batch(inputs.cpu(), outputs.cpu(), targets.cpu())

            val_loss /= j
            val_dice /= j
            val_dice_sample_avg /= j 
    
    scheduler.step()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if SAVE_CHECKPOINT_DURING_TRAINING:
        torch.save(checkpoint, SAVE_CHECKPOINT_FILE)
        
    print('')
    print(f'EPOCH DONE --- LR: {scheduler.get_last_lr()[0]}----')
    if (epoch + 1) % INTERVAL_TO_CHECK_VALIDATION_SET_RESULTS == 0: # Epoch on which Validation Set is checked 
        print(f'----Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.9f}, Val Loss: {val_loss:.9f}, LR: {scheduler.get_last_lr()[0]}----')
        print(f'----Train batch global DCE: {train_dice:.2f}, Val batch global DCE: {val_dice:.2f}, Train sample avg DCE: {train_dice_sample_avg:.2f}, Val avg sample DCE: {val_dice_sample_avg:.2f}----')
    else: # Epoch on which validation set is not checked to save computation 
        print(f'----Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.9f}, LR: {scheduler.get_last_lr()[0]}----')
        print(f'----Train batch global DCE: {train_dice:.2f}, Train sample avg DCE: {train_dice_sample_avg:.2f} ----')
