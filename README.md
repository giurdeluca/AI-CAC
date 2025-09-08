# Personal documentation
> Check [README_original](README_original.md) for further documentation and citation!

## Installation
Clone the repo
```shell
git clone https://github.com/Raffi-Hagopian/AI-CAC.git
```

Create the environment (python version was specified [here](https://github.com/Raffi-Hagopian/AI-CAC/issues/3#issuecomment-2986416848))
```shell
conda create -n <env_name> python=3.9.19
```

Install required packages (pytorch installed separately as it is an old version)
```shell
cd AI-CAC
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Requirements

Install dependencies with: `'pip install -r requirements.txt'`

## Instructions for Inference (running AI-CAC scoring on non-gated, non-contrast chest CTs)

1. Create a folder for each CT chest scan study and place DICOM files from that study within the folder. The folder name will be used as the name for that study.
2. Create a config.json file in the project directory.
REQUIRED fields in the config:
   * `gpu_device`- The ID of the CUDA GPU that will be used for inference. Accepts only a single id for now!
   * `dicom_root_dir` – Set to the path of the parent folder that contains the non-gated study subfolders as described above.
   * `model_checkpoint_file` – Set to the path of the model weights (download: [va_non_gated_ai_cac_model.pth](https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth)).
   * `output_dir` – Set to the path where the final CSV table containing the study-level AI-CAC generated calcium scores (scores.csv) and eventual maps will be saved.
OPTIONAL:
   * `save_masks` – Flag whether to save AI-CAC segmentations into PNG files (`default=False`; setting to `True` will slow inference).
   * `visualize_results` – Flag whether to display segmentation masks during inference (`default=False`; setting to `True` will slow inference).


EXAMPLE config.json:
```JSON
{
  "processing": {
    "gpu_device": 2,
    "num_workers": 12,
    "batch_size": 16,
    "resample_image_size": [512, 512],
    "resample_shape": [512, 512, 64],
    "zoom_factors": [1, 1, 1]
  },
  "output": {
    "save_masks": false,
    "visualize_results": false
  },
  "paths": {
    "dicom_root_dir": "/path/to/dicom/root/dir",
    "model_checkpoint_file": "/path/to/model/checkpoint",
    "output_dir":"/path/to/output/dir"
  }
}
```
4. Run `main_inference.py` to generate the AI-CAC scores for your studies.

The code will select a single non-contrast chest series per study that is most suitable for our CAC model using DICOM metadata. The script internally creates a metadata table across all the suitable imaging series, where each row represents a single DICOM file from a selected series, and has the following columns: `StudyName, DICOMFilePath, AxialPosition`. This table will be used by the inference code to run the model on each slice/DICOM file from the series and aggregate the results into a CAC score.

## Run docker image
1. Build the docker image
```shell
docker build -t aicac .
```

2. Run image
```shell
docker run --gpus all --shm-size=8g -v /local/path/to/input/folder:/app/data -v /local/path/to/output/folder:/app/output -v /home/deluca/test-data/config.json:/app/config.json:ro -v /local/path/to/model_checkpoint:/app/model_checkpoint aicac
```
