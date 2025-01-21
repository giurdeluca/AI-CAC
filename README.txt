Instructions for AI-CAC model inference on Non-gated Chest CT scans: 

Please create a folder for each CT chest scan study and place DICOM files from that study within the folder. The folder name will be used as the name for that study. The code will take a root directory consisting of multiple folders from multiple studies for inference. The code will select a single non-contrast chest series per study that is most suitable for our CAC model. The software will create a table for reach dicom file from acceptable series and place them into a table with the following columns "StudyName", "DICOMFilePath", "". This table will be used by the inference code to run the model on each slice from the series and aggregate the results into a CAC score.

 Input: Specify root folder containing DICOM study subfolders
 Output: Specify name/location of CSV file to be saved with AI generated CAC scores. 
 Optional: Save PNG masks (alternative options coming soon)



Package Requirements:
    Databricks Runtime 11.3 LTS ML Environment g4dn.12xLarge
    Python 3.9.19 (main, Apr  6 2024, 17:57:55) 
    pip 21.2.4 
Python Packages: 
    itk                                5.4.0
    matplotlib                         3.9.4
    monai                              1.4.0
    monailabel                         0.8.5
    nibabel                            5.3.2
    numpy                              1.26.4
    opencv-python                      4.11.0.86
    pandas                             1.3.4
    pillow                             11.1.0
    pydicom                            2.4.4
    python-gdcm                        3.0.24.1
    scikit-learn                       0.24.2
    scipy                              1.13.1
    SimpleITK                          2.4.1
    torch                              1.12.1+cu113
    torchmetrics                       1.5.2
    torchvision                        0.13.1+cu113


Public Init GPU script 
#!/bin/bash
pip install itk==5.4.0 matplotlib==3.9.4 monai[all]==1.4.0 monailabel==0.8.5 nibabel==5.3.2 
pip install numpy==1.26.4 opencv-python==4.11.0.86 pandas==1.3.4 pillow==11.1.0 pydicom==2.4.4 
pip install python-gdcm==3.0.24.1 scikit-learn==0.24.2 scipy==1.13.1 SimpleITK==2.4.1 
pip install torch==1.12.1+cu113 torchmetrics==1.5.2 torchvision==0.13.1+cu113 

