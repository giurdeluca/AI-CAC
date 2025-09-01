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
            orientation_list = ast.literal_eval(orientation_list)
            int_list = [int(x) for x in orientation_list]
            return int_list == [1,0,0,0,1,0] #Axial orientation
    
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
        if row['ContrastBolusAgent'] != 'None': 
            return False
        if not is_axial(orient):
            return False
        if not keep_series(study_desc, series_desc, body_part):
            return False
        return True 
  
    def select_row(group):
        series_desc = group['SeriesDescription'].str.lower() # set to lower case
        body_part = group['BodyPartExamined'].str.lower()
        # Return the first series for this study that satisfies this search
        # Initially searching over calcium terms, then heart anatomy, then lung, then chest, then if none matched, pick first series left
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

    # If a series is repeated within a study (similar series specific description, settings tags etc), keep the one that was acquired later in time (more likely higher quality)
    def keep_latest_series_if_repeated(dicom_df):
        dicom_df['AcquisitionTime'] = pd.to_numeric(dicom_df['AcquisitionTime'], errors='coerce')
        dicom_df['AcquisitionTime'] = dicom_df['AcquisitionTime'].fillna(0) # set NA to 0
        max_time_per_pair = dicom_df.groupby(['StudyName','SeriesInstanceUID'])['AcquisitionTime'].max().reset_index() # Get the latest Study/SeriesID pair timestamp 
        latest_pairs = max_time_per_pair.loc[max_time_per_pair.groupby('StudyName')['AcquisitionTime'].idxmax()] # For each study, keep the series with the latest timestamp
        result_df = pd.merge(dicom_df, latest_pairs[['StudyName','SeriesInstanceUID']], on=['StudyName', 'SeriesInstanceUID'])
        return result_df

    series_specific_columns = ['StudyName','StudyDescription','SeriesDescription', 'SliceThickness', 'ImageType', 'ConvolutionKernel', 'ImageOrientationPatient', 'KVP', 'ContrastBolusAgent', 'BodyPartExamined']

    dicom_df = dicom_df[dicom_df['Modality'] == 'CT']
    dicom_df['SliceThickness'] =  pd.to_numeric(dicom_df['SliceThickness'], errors='coerce')
    dicom_df = dicom_df[(dicom_df['SliceThickness'] >= 2.5) & (dicom_df['SliceThickness'] <= 5)] # Keep DICOMs with SliceTHickness between 2.5 and 5
    dicom_df.fillna('None', inplace = True) # Replace NA with string 'None'
    dicom_df = dicom_df.astype(str) # allows for grouping in certain columns 
    group_df = dicom_df.groupby(series_specific_columns, as_index = False).size() # Group by these columns to get one row per series in the dataframe 
    group_df = group_df[group_df['size'] > 15] # Only keep series with at least 15 slices

    #debug
    print(f"group_df shape before filtering: {group_df.shape}")
    print(f"group_df columns: {group_df.columns.tolist()}")
    print("Sample rows from group_df:")
    print(group_df.head())
    filter_results = []
    for index, row in group_df.iterrows():
        result = filter_row(row)
        filter_results.append(result)
        if not result:
            print(f"Row {index} rejected:")
            print(f"  StudyDescription: {row['StudyDescription']}")
            print(f"  SeriesDescription: {row['SeriesDescription']}")
            print(f"  BodyPartExamined: {row['BodyPartExamined']}")
            print(f"  ContrastBolusAgent: {row['ContrastBolusAgent']}")
            print(f"  ImageOrientationPatient: {row['ImageOrientationPatient']}")
            print()

    print(f"Filter results: {sum(filter_results)} passed out of {len(filter_results)} total")
    #
    filt_df = group_df[[filter_row(row) for index,row in group_df.iterrows()]] # Filter out series with contrast, non-axial slices, non-chest body part
    select_df = filt_df.groupby('StudyName').apply(select_row).reset_index(drop=True) # Select series based on search term priorties calcium terms, cardiac terms, lung, chest etc
    one_series_per_study_df = pd.merge(dicom_df, select_df, on = series_specific_columns) # Expand selected series dataframe to now have rows for each slice (still only one series per study selected)

    one_series_per_study_df = keep_latest_series_if_repeated(one_series_per_study_df) # If series specific settings were repeated twice, keep the one with the latest timestamp 
    one_series_per_study_df['AxialPosition'] = one_series_per_study_df['ImagePositionPatient'].apply(lambda coord: coord.split(', ')[-1].replace(']', '')) # Take Axial position out of [#, #, #]
    one_series_per_study_df = one_series_per_study_df.astype(str)
    return one_series_per_study_df

