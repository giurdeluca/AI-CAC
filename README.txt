About:

The AI-CAC is a deep learning model that segments and scores coronary artery calcium (CAC) on routine non-gated, non-contrast chest CT scans. The model has been developed and internally validated in the U.S. Department of Veterans Affairs (VA) and benchmarked against paired gated CAC studies. 

This repository contains: 

  •	Trained model weights 
  •	Inference code
  •	Training code 

Requirements:

Install dependencies with: ‘pip install -r requirements.txt’  

Instructions for Inference (running AI-CAC scoring non-gated, non-contrast chest CTs): 

  1)	Create a folder for each CT chest scan study and place DICOM files from that study within the folder. The folder name will be used as the name for that study. 
  2)	Modify the following hardcoded variables in main_inference.py:
      a.	DICOM_ROOT_DIR – set to the path of the parent folder that contains the non-gated study subfolders as described above. 
      b.	MODEL_CHECKPOINT_FILE - set to the path of the model weights [available for download as a binary file in this repository].
      c.	SCORE_FILE – set this to the path where the final CSV table containing the study-level AI-CAC generated calcium scores will be saved.
  3)	OPTIONAL:
      a.	VISUALIZE_RESULTS – Flag whether to display segmentation masks during inference (default = False, setting to True will slow inference). 
      b.	SAVE_MASKS – Flag whether to save AI-CAC segmentations into PNG files (default = False, setting to True will slow inference). 
      c.	MASK_FOLDER – Directory in which to save PNG masks.
  4)	Running main_inference.py will now generate the AI-CAC scores for your studies. 

The code will select a single non-contrast chest series per study that is most suitable for our CAC model using DICOM metadata. The script internally creates a metadata table across all the suitable imaging series, where each row represents a single DICOM file from a selected series, and has the following columns: "StudyName", "DICOMFilePath", and "AxialPosition." This table will be used by the inference code to run the model on each slice/DICOM file from the series and aggregate the results into a CAC score.

Citation: 

Please cite our NEJM AI paper: doi.org/10.1056/AIoa2400937
