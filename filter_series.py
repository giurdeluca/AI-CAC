# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import re
import ast 
import pydicom 
import pandas as pd 
import matplotlib.pyplot as plt

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
  
# Create Pandas dataframe of StudyName, DICOMFilePath, and DICOM Attributes 
def create_dicom_df(root_dir):
    data = []
    for study_name in os.listdir(root_dir):
        study_path = os.path.join(root_dir, study_name)
        if os.path.isdir(study_path):
            for parent_path, _, files in os.walk(study_path):
                for file_name in files:
                    if file_name.lower().endswith('.dcm'):
                        dicom_path = os.path.join(parent_path, file_name)
                        dicom_attr = extract_dicom_attributes(dicom_path)
                        row = {
                            'StudyName': study_name,
                            'DICOMFilePath': dicom_path, 
                            **dicom_attr,
                        }
                        data.append(row)
    df = pd.DataFrame(data)
    print(f"Created initial DICOM dataframe with {len(df)} files from {df['StudyName'].nunique()} studies")
    return df


# Filter to keep only one series per study
# Criteria:
#  1) At least 15 slices
#  2) Non-contrast 
#  3) Axial in Orientation
#  4) No Study, Series, or Body Part descriptors containing non-chest anatomy words [head, skull, brain, ... etc]
#  5) Of the remaining series, select the first series in order of the following descriptors [calc, casc, cac, ca]
#  6) If none of the above words match series/body part descriptors, select the first remaining series in the dataframe
 
def filter_dicom_df(dicom_df):

    def is_axial(orientation_list):
        if orientation_list == None or orientation_list == 'None':
            return True
        else:
            try:
                orientation_list = ast.literal_eval(orientation_list)
                int_list = [int(x) for x in orientation_list]
                return int_list == [1,0,0,0,1,0] #Axial orientation
            except:
                print(f"Warning: Could not parse orientation: {orientation_list}")
                return True  # Default to True if can't parse
    
    def keep_series(study_desc, series_desc, body_part): 
        if study_desc == 'None' or series_desc == 'None' or study_desc == '' or series_desc == '':
            return False 
        study_desc = study_desc.lower()
        series_desc = series_desc.lower()
        body_part = body_part.lower()
        filter_terms = ['head', 'brain', 'skull', 'sinus', 'maxillofacial', 'neck', 'spine', 'sternum', 'bone', 'abdomen', 'abd', 'adrenal', 'liver', 'kidney', 'colon', 'pelvis', 'femoral', 'leg', 'extremity', 'abscess', 'needle', 'drain', 'circleofwillis', 'tavr', 'mip', 'arterial', 'venous', 'delay', 'runoff', 'enhanced']
        
        for term in filter_terms:
            if term in study_desc or term in series_desc or term in body_part:
                return False
        pattern = r'(?i)(?<!\S)A/P(?!\S)' #remove any 'A/P' for abdomen/pelvis if no preceeding or following non-whitespace chacerters 
        if re.search(pattern, series_desc):
            return False
        return True 
    
    def filter_row(row):
        study_desc = row['StudyDescription']
        orient = row['ImageOrientationPatient']
        series_desc = row['SeriesDescription']
        body_part = row['BodyPartExamined']
        contrast = row['ContrastBolusAgent']
        
        if contrast != 'None': 
            print(f"  ❌ Rejected (contrast): Study='{row['StudyName']}', Series='{series_desc}', Contrast='{contrast}'")
            return False
        if not is_axial(orient):
            print(f"  ❌ Rejected (orientation): Study='{row['StudyName']}', Series='{series_desc}', Orientation='{orient}'")
            return False
        if not keep_series(study_desc, series_desc, body_part):
            print(f"  ❌ Rejected (anatomy filter): Study='{row['StudyName']}', Series='{series_desc}', Body='{body_part}'")
            return False
        
        print(f"  ✅ Accepted: Study='{row['StudyName']}', Series='{series_desc}', Slices={row['size']}")
        return True 
  
    def select_row(group):
        series_desc = group['SeriesDescription'].str.lower() # set to lower case
        body_part = group['BodyPartExamined'].str.lower()
        
        print(f"    Selecting from {len(group)} series for study '{group['StudyName'].iloc[0]}':")
        for idx, row in group.iterrows():
            print(f"      - '{row['SeriesDescription']}' ({row['size']} slices)")
        
        # Return the first series for this study that satisfies this search
        # Initially searching over calcium terms, then heart anatomy, then lung, then chest, then if none matched, pick first series left
        if any('calc' in s for s in series_desc):
            selected = group[series_desc.str.contains('calc')].iloc[0]
            print(f"    🎯 Selected (calc): '{selected['SeriesDescription']}'")
            return selected
        if any('cacs' in s for s in series_desc):
            selected = group[series_desc.str.contains('cacs')].iloc[0]
            print(f"    🎯 Selected (cacs): '{selected['SeriesDescription']}'")
            return selected
        if any('ca ' in s for s in series_desc):
            selected = group[series_desc.str.contains('ca ')].iloc[0]
            print(f"    🎯 Selected (ca): '{selected['SeriesDescription']}'")
            return selected
        if any('cac' in s for s in series_desc):
            selected = group[series_desc.str.contains('cac')].iloc[0]
            print(f"    🎯 Selected (cac): '{selected['SeriesDescription']}'")
            return selected
        if any('calcium' in s for s in body_part):
            selected = group[body_part.str.contains('calcium')].iloc[0]
            print(f"    🎯 Selected (calcium body): '{selected['SeriesDescription']}'")
            return selected
        if any('ca ' in s for s in body_part):
            selected = group[body_part.str.contains('ca ')].iloc[0]
            print(f"    🎯 Selected (ca body): '{selected['SeriesDescription']}'")
            return selected
        if any('heart' in s for s in body_part):
            selected = group[body_part.str.contains('heart')].iloc[0]
            print(f"    🎯 Selected (heart): '{selected['SeriesDescription']}'")
            return selected
        if any('card' in s for s in series_desc):
            selected = group[series_desc.str.contains('card')].iloc[0]
            print(f"    🎯 Selected (card): '{selected['SeriesDescription']}'")
            return selected
        if any('lung' in s for s in series_desc):
            selected = group[series_desc.str.contains('lung')].iloc[0]
            print(f"    🎯 Selected (lung): '{selected['SeriesDescription']}'")
            return selected
        if any('chest' in s for s in body_part):
            selected = group[body_part.str.contains('chest')].iloc[0]
            print(f"    🎯 Selected (chest): '{selected['SeriesDescription']}'")
            return selected
        else:
            selected = group.iloc[0]
            print(f"    🎯 Selected (first): '{selected['SeriesDescription']}'")
            return selected

    # If a series is repeated within a study (similar series specific description, settings tags etc), keep the one that was acquired later in time (more likely higher quality)
    def keep_latest_series_if_repeated(dicom_df):
        dicom_df['AcquisitionTime'] = pd.to_numeric(dicom_df['AcquisitionTime'], errors='coerce')
        dicom_df['AcquisitionTime'] = dicom_df['AcquisitionTime'].fillna(0) # set NA to 0
        max_time_per_pair = dicom_df.groupby(['StudyName','SeriesInstanceUID'])['AcquisitionTime'].max().reset_index() # Get the latest Study/SeriesID pair timestamp 
        latest_pairs = max_time_per_pair.loc[max_time_per_pair.groupby('StudyName')['AcquisitionTime'].idxmax()] # For each study, keep the series with the latest timestamp
        result_df = pd.merge(dicom_df, latest_pairs[['StudyName','SeriesInstanceUID']], on=['StudyName', 'SeriesInstanceUID'])
        return result_df

    print("="*60)
    print("STARTING DICOM FILTERING PROCESS")
    print("="*60)

    series_specific_columns = ['StudyName','StudyDescription','SeriesDescription', 'SliceThickness', 'ImageType', 'ConvolutionKernel', 'ImageOrientationPatient', 'KVP', 'ContrastBolusAgent', 'BodyPartExamined']

    # Step 1: Filter by modality
    print(f"\nStep 1: Filtering by modality")
    print(f"  Before: {len(dicom_df)} files")
    print(f"  Modality distribution: {dicom_df['Modality'].value_counts().to_dict()}")
    dicom_df = dicom_df[dicom_df['Modality'] == 'CT']
    print(f"  After CT filter: {len(dicom_df)} files")

    # Step 2: Analyze and filter by slice thickness
    print(f"\nStep 2: Analyzing slice thickness")
    dicom_df['SliceThickness'] = pd.to_numeric(dicom_df['SliceThickness'], errors='coerce')
    
    # Log slice thickness distribution
    print("  Slice thickness analysis:")
    thickness_counts = dicom_df['SliceThickness'].value_counts(dropna=False).sort_index()
    for thickness, count in thickness_counts.items():
        print(f"    {thickness}mm: {count} files")
    
    print(f"  Slice thickness statistics:")
    print(f"    Min: {dicom_df['SliceThickness'].min()}")
    print(f"    Max: {dicom_df['SliceThickness'].max()}")
    print(f"    Mean: {dicom_df['SliceThickness'].mean():.2f}")
    print(f"    Null values: {dicom_df['SliceThickness'].isna().sum()}")
    
    # Apply slice thickness filter
    before_thickness = len(dicom_df)
    thickness_mask = (dicom_df['SliceThickness'] >= 2.5) & (dicom_df['SliceThickness'] <= 5.0)
    dicom_df = dicom_df[thickness_mask]
    print(f"  After thickness filter (2.5-5.0mm): {len(dicom_df)} files")
    print(f"  Removed {before_thickness - len(dicom_df)} files due to slice thickness")
    
    if dicom_df.shape[0] == 0:
        print('❌ CRITICAL: No images satisfy the slice thickness requirement')
        print('   Consider adjusting the slice thickness range or check your DICOM files')
        return pd.DataFrame()  # Return empty dataframe
    
    # Step 3: Group by series
    print(f"\nStep 3: Grouping by series")
    dicom_df.fillna('None', inplace=True) # Replace NA with string 'None'
    dicom_df = dicom_df.astype(str) # allows for grouping in certain columns 
    group_df = dicom_df.groupby(series_specific_columns, as_index=False).size() # Group by these columns to get one row per series in the dataframe 
    print(f"  Found {len(group_df)} unique series across {group_df['StudyName'].nunique()} studies")
    
    # Log series info
    for study in group_df['StudyName'].unique():
        study_series = group_df[group_df['StudyName'] == study]
        print(f"    Study '{study}': {len(study_series)} series")
        for _, row in study_series.iterrows():
            print(f"      - '{row['SeriesDescription']}' ({row['size']} slices)")

    # Step 4: Filter by minimum slice count
    print(f"\nStep 4: Filtering by minimum slice count (>15)")
    before_slice_count = len(group_df)
    group_df = group_df[group_df['size'] > 15] # Only keep series with at least 15 slices
    print(f"  Before: {before_slice_count} series")
    print(f"  After: {len(group_df)} series")
    print(f"  Removed {before_slice_count - len(group_df)} series with ≤15 slices")

    if len(group_df) == 0:
        print('❌ CRITICAL: No series have more than 15 slices')
        return pd.DataFrame()

    # Step 5: Apply quality filters (contrast, orientation, anatomy)
    print(f"\nStep 5: Applying quality filters (contrast, orientation, anatomy)")
    print("  Testing each series:")
    
    filter_results = []
    for index, row in group_df.iterrows():
        result = filter_row(row)
        filter_results.append(result)
    
    filt_df = group_df[filter_results]
    print(f"  Results: {len(filt_df)} series passed out of {len(group_df)} total")
    
    if len(filt_df) == 0:
        print('❌ CRITICAL: No series passed the quality filters')
        print('   Consider adjusting filter criteria for your dataset')
        return pd.DataFrame()

    # Step 6: Select best series per study
    print(f"\nStep 6: Selecting best series per study")
    select_df = filt_df.groupby('StudyName').apply(select_row).reset_index(drop=True)
    print(f"  Final selection: {len(select_df)} series (one per study)")

    # Step 7: Expand back to slice level
    print(f"\nStep 7: Expanding back to slice level")
    one_series_per_study_df = pd.merge(dicom_df, select_df, on=series_specific_columns)
    print(f"  Expanded to {len(one_series_per_study_df)} individual slices")

    # Step 8: Handle repeated series
    print(f"\nStep 8: Handling repeated series (keeping latest)")
    one_series_per_study_df = keep_latest_series_if_repeated(one_series_per_study_df)
    print(f"  Final result: {len(one_series_per_study_df)} slices from {one_series_per_study_df['StudyName'].nunique()} studies")

    # Step 9: Add axial position
    print(f"\nStep 9: Adding axial position information")
    one_series_per_study_df['AxialPosition'] = one_series_per_study_df['ImagePositionPatient'].apply(lambda coord: coord.split(', ')[-1].replace(']', ''))
    one_series_per_study_df = one_series_per_study_df.astype(str)

    print("="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    
    # Final summary
    for study in one_series_per_study_df['StudyName'].unique():
        study_data = one_series_per_study_df[one_series_per_study_df['StudyName'] == study]
        series_desc = study_data['SeriesDescription'].iloc[0]
        print(f"✅ Study '{study}': {len(study_data)} slices from series '{series_desc}'")

    return one_series_per_study_df