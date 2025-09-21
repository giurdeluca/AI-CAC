# AI-CAC Project Code - Optimized Version
# Creator: Raffi Hagopian MD

import os
import re
import ast
from pathlib import Path
import pydicom 
import pandas as pd 
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from collections import defaultdict
import time

# Extract selected attributes using pydicom 
def extract_dicom_attributes(dicom_file):
    attributes = {
        'Modality' : None,
        'SliceThickness' : None,
        'SeriesDescription' : None,
        'StudyDescription' : None,
        'KVP' : None,
        'ConvolutionKernel' : None,
        'ImageOrientationPatient' : None,
        'ImageType': None,
        'ContrastBolusAgent': None, 
        'BodyPartExamined': None, 
        'AcquisitionTime' : None, 
        'SeriesInstanceUID': None, 
        'ImagePositionPatient': None, # Used to get slice axial position 
    }
    try: 
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        for attr in attributes.keys():
            attributes[attr] = getattr(ds, attr, None)
    except Exception as e:
        print(f"Error reading DICOM file {dicom_file}: {e}")
    return attributes 
def create_series_df_sequential(series_representatives, batch_size=10000):
    """Sequential processing for very large datasets with memory management"""
    print("  Using sequential processing with batching...")
    
    total = len(series_representatives)
    all_data = []
    
    # Process in batches to manage memory
    series_items = list(series_representatives.items())
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = series_items[batch_start:batch_end]
        
        print(f"    Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"({batch_start}-{batch_end}/{total})")
        
        batch_data = []
        for i, (series_key, info) in enumerate(batch_items):
            if i % 1000 == 0 and i > 0:
                print(f"      Progress: {i}/{len(batch_items)} in current batch")
            
            # Read DICOM metadata from representative file only
            dicom_attr = extract_dicom_attributes(info['representative_file'])
            
            row = {
                'StudyName': info['study_name'],
                'SeriesKey': series_key,
                'RepresentativeFile': info['representative_file'],
                'SeriesFolder': info['series_folder'],
                'SeriesFolderName': info['series_folder_name'],
                'SliceCount': info['slice_count'],
                **dicom_attr
            }
            batch_data.append(row)
        
        all_data.extend(batch_data)
        print(f"      Completed batch: {len(batch_data)} series processed")
    
    df = pd.DataFrame(all_data)
    print(f"  Created series DataFrame with {len(df)} rows (one per series)")
    return df

def identify_series_from_paths(root_dir):
    """Identify unique series without reading DICOM files - much faster!"""
    print("Step 1: Identifying series from directory structure...")
    start_time = time.time()
    
    series_representatives = {}  # series_key -> info
    series_file_counts = defaultdict(int)
    study_series_map = defaultdict(set)  # study -> set of series_keys
    
    root_path = Path(root_dir)
    total_files = 0
    
    for dicom_file in root_path.rglob("*.dcm"):
        total_files += 1
        if total_files % 50000 == 0:
            print(f"  Processed {total_files} files...")
        
        # Extract study name from path
        path_parts = dicom_file.parts
        subject_name = None
        session_name = None
        
        for part in path_parts:
            if subject_name is None and part.startswith(('sub-', 'subject')):
                subject_name = part
            elif part.startswith('ses-'):
                session_name = part
                break
        
        if subject_name is None:
            continue
            
        study_name = f"{subject_name}_{session_name}" if session_name else subject_name
        
        # Create series identifier from path structure
        # Most DICOM datasets organize files by series in folders
        series_folder = dicom_file.parent
        series_key = f"{study_name}||{series_folder.name}||{str(series_folder)}"
        
        # Count files per series
        series_file_counts[series_key] += 1
        study_series_map[study_name].add(series_key)
        
        # Keep first file as representative
        if series_key not in series_representatives:
            series_representatives[series_key] = {
                'study_name': study_name,
                'representative_file': str(dicom_file),
                'series_folder': str(series_folder),
                'series_folder_name': series_folder.name
            }
    
    elapsed = time.time() - start_time
    print(f"  Found {len(series_representatives)} unique series from {total_files} files")
    print(f"  Across {len(study_series_map)} studies in {elapsed:.1f} seconds")
    
    # Add slice counts to representatives
    for series_key, info in series_representatives.items():
        info['slice_count'] = series_file_counts[series_key]
    
    return series_representatives, study_series_map

def create_series_df_parallel(series_representatives, max_workers=4):
    """Create DataFrame with one row per series using parallel processing"""
    print(f"Step 2: Reading metadata from {len(series_representatives)} representative files...")
    
    def process_series_chunk(series_items):
        chunk_data = []
        for series_key, info in series_items:
            # Read DICOM metadata from representative file only
            dicom_attr = extract_dicom_attributes(info['representative_file'])
            
            row = {
                'StudyName': info['study_name'],
                'SeriesKey': series_key,
                'RepresentativeFile': info['representative_file'],
                'SeriesFolder': info['series_folder'],
                'SeriesFolderName': info['series_folder_name'],
                'SliceCount': info['slice_count'],
                **dicom_attr
            }
            chunk_data.append(row)
        return chunk_data
    
    # Split into chunks for parallel processing
    series_items = list(series_representatives.items())
    chunk_size = max(1, len(series_items) // (max_workers * 2))
    chunks = [series_items[i:i + chunk_size] for i in range(0, len(series_items), chunk_size)]
    
    all_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_series_chunk, chunk): chunk for chunk in chunks}
        
        for i, future in enumerate(as_completed(future_to_chunk)):
            chunk_data = future.result()
            all_data.extend(chunk_data)
            print(f"  Processed chunk {i+1}/{len(chunks)}")
    
    df = pd.DataFrame(all_data)
    print(f"  Created series DataFrame with {len(df)} rows (one per series)")
    return df

def filter_series_df(series_df):
    """Apply filtering at series level - much faster than slice level"""
    
    def is_axial(orientation_list):
        if orientation_list is None or str(orientation_list) in ['None', 'nan', '']:
            return True
        else:
            try:
                if isinstance(orientation_list, str):
                    orientation_list = ast.literal_eval(orientation_list)
                int_list = [int(float(x)) for x in orientation_list]
                return int_list == [1,0,0,0,1,0] #Axial orientation
            except:
                return True  # Default to True if can't parse
    
    def keep_series(study_desc, series_desc, body_part): 
        if not study_desc or not series_desc or str(study_desc) in ['None', 'nan', ''] or str(series_desc) in ['None', 'nan', '']:
            return False 
            
        study_desc = str(study_desc).lower()
        series_desc = str(series_desc).lower()
        body_part = str(body_part).lower()
        
        filter_terms = ['head', 'brain', 'skull', 'sinus', 'maxillofacial', 'neck', 'spine', 
                       'sternum', 'bone', 'abdomen', 'abd', 'adrenal', 'liver', 'kidney', 
                       'colon', 'pelvis', 'femoral', 'leg', 'extremity', 'abscess', 'needle', 
                       'drain', 'circleofwillis', 'tavr', 'mip', 'arterial', 'venous', 
                       'delay', 'runoff', 'enhanced']
        
        for term in filter_terms:
            if term in study_desc or term in series_desc or term in body_part:
                return False
                
        pattern = r'(?i)(?<!\S)A/P(?!\S)'
        if re.search(pattern, series_desc):
            return False
        return True 
    
    def select_best_series_per_study(group):
        """Select the best series for each study"""
        series_desc = group['SeriesDescription'].astype(str).str.lower()
        body_part = group['BodyPartExamined'].astype(str).str.lower()
        
        # Priority order for series selection
        if any('calc' in s for s in series_desc):
            return group[series_desc.str.contains('calc')].iloc[0]
        if any('cacs' in s for s in series_desc):
            return group[series_desc.str.contains('cacs')].iloc[0]
        if any('ca ' in s for s in series_desc):
            return group[series_desc.str.contains('ca ')].iloc[0]
        if any('cac' in s for s in series_desc):
            return group[series_desc.str.contains('cac')].iloc[0]
        if any('calcium' in s for s in body_part):
            return group[body_part.str.contains('calcium')].iloc[0]
        if any('ca ' in s for s in body_part):
            return group[body_part.str.contains('ca ')].iloc[0]
        if any('heart' in s for s in body_part):
            return group[body_part.str.contains('heart')].iloc[0]
        if any('card' in s for s in series_desc):
            return group[series_desc.str.contains('card')].iloc[0]
        if any('lung' in s for s in series_desc):
            return group[series_desc.str.contains('lung')].iloc[0]
        if any('chest' in s for s in body_part):
            return group[body_part.str.contains('chest')].iloc[0]
        else:
            return group.iloc[0]

    print("Step 3: Filtering series...")
    original_count = len(series_df)
    
    # Apply all filters
    print(f"  Starting with {original_count} series")
    
    # Filter 1: Modality
    series_df = series_df[series_df['Modality'] == 'CT']
    print(f"  After CT filter: {len(series_df)} series")
    
    # Filter 2: Slice thickness
    series_df['SliceThickness'] = pd.to_numeric(series_df['SliceThickness'], errors='coerce')
    thickness_mask = (series_df['SliceThickness'] >= 2.5) & (series_df['SliceThickness'] <= 5.0)
    series_df = series_df[thickness_mask]
    print(f"  After thickness filter (2.5-5.0mm): {len(series_df)} series")
    
    # Filter 3: Minimum slice count
    series_df = series_df[series_df['SliceCount'] > 15]
    print(f"  After slice count filter (>15): {len(series_df)} series")
    
    # Filter 4: Contrast
    contrast_mask = (series_df['ContrastBolusAgent'].isna()) | (series_df['ContrastBolusAgent'].astype(str) == 'None')
    series_df = series_df[contrast_mask]
    print(f"  After contrast filter: {len(series_df)} series")
    
    # Filter 5: Orientation
    orientation_mask = series_df['ImageOrientationPatient'].apply(is_axial)
    series_df = series_df[orientation_mask]
    print(f"  After orientation filter: {len(series_df)} series")
    
    # Filter 6: Anatomy terms
    anatomy_mask = series_df.apply(lambda row: keep_series(
        row['StudyDescription'], 
        row['SeriesDescription'], 
        row['BodyPartExamined']
    ), axis=1)
    series_df = series_df[anatomy_mask]
    print(f"  After anatomy filter: {len(series_df)} series")
    
    if len(series_df) == 0:
        print("  WARNING: No series passed all filters!")
        return pd.DataFrame()
    
    # Filter 7: Select best series per study
    print("  Selecting best series per study...")
    selected_series = series_df.groupby('StudyName').apply(select_best_series_per_study).reset_index(drop=True)
    print(f"  Final selection: {len(selected_series)} series (one per study)")
    
    return selected_series

def expand_to_slice_level(selected_series):
    """Expand selected series back to individual slice level"""
    print("Step 4: Expanding selected series to slice level...")
    
    all_slice_data = []
    
    for _, series_row in selected_series.iterrows():
        study_name = series_row['StudyName']
        series_folder = Path(series_row['SeriesFolder'])
        
        print(f"  Processing series: {study_name} - {series_row['SeriesDescription']}")
        
        # Get all DICOM files in this series folder
        dicom_files = list(series_folder.glob("*.dcm"))
        
        for dicom_file in dicom_files:
            try:
                # Read only the ImagePositionPatient for axial position
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True,
                                   specific_tags=['ImagePositionPatient'])
                position = getattr(ds, 'ImagePositionPatient', [0,0,0])
                axial_pos = float(position[2]) if len(position) > 2 else 0.0
            except:
                axial_pos = 0.0
            
            # Create row with series metadata + slice-specific info
            slice_row = {
                'StudyName': study_name,
                'DICOMFilePath': str(dicom_file),
                'AxialPosition': axial_pos,
                # Copy all series-level metadata
                'SeriesDescription': series_row['SeriesDescription'],
                'StudyDescription': series_row['StudyDescription'],
                'Modality': series_row['Modality'],
                'SliceThickness': series_row['SliceThickness'],
                'KVP': series_row['KVP'],
                'ConvolutionKernel': series_row['ConvolutionKernel'],
                'ImageOrientationPatient': series_row['ImageOrientationPatient'],
                'ImageType': series_row['ImageType'],
                'ContrastBolusAgent': series_row['ContrastBolusAgent'],
                'BodyPartExamined': series_row['BodyPartExamined'],
                'AcquisitionTime': series_row['AcquisitionTime'],
                'SeriesInstanceUID': series_row['SeriesInstanceUID'],
                'ImagePositionPatient': str([0, 0, axial_pos]),  # Simplified
            }
            all_slice_data.append(slice_row)
    
    final_df = pd.DataFrame(all_slice_data)
    print(f"  Expanded to {len(final_df)} slices from {final_df['StudyName'].nunique()} studies")
    
    return final_df

def optimized_dicom_processing(root_dir, max_workers=4, force_sequential=False):
    """Complete optimized workflow - series-level processing first"""
    print("="*60)
    print("OPTIMIZED DICOM PROCESSING")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Identify series from paths (fast)
    series_representatives, study_series_map = identify_series_from_paths(root_dir)
    
    # Step 2: Create series-level DataFrame (small, manageable)
    if force_sequential or len(series_representatives) > 50000:
        series_df = create_series_df_sequential(series_representatives)
    else:
        try:
            series_df = create_series_df_parallel(series_representatives, max_workers)
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to sequential processing...")
            series_df = create_series_df_sequential(series_representatives)
    
    # Step 3: Filter at series level (fast)
    selected_series = filter_series_df(series_df)
    
    if len(selected_series) == 0:
        print("No series passed filtering criteria!")
        return pd.DataFrame()
    
    # Step 4: Expand only selected series to slice level
    final_df = expand_to_slice_level(selected_series)
    
    # Step 5: Final cleanup
    final_df = final_df.astype(str)  # Convert to string as in original
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"PROCESSING COMPLETE in {total_time/60:.1f} minutes")
    print(f"Final result: {len(final_df)} slices from {final_df['StudyName'].nunique()} studies")
    print("="*60)
    
    return final_df