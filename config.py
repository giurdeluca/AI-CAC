# config.py
# Configuration management for AI-CAC Project

import json
import sys
from pathlib import Path

class Config:
    """Configuration loader for AI-CAC processing pipeline."""
    
    # Default values
    DEFAULT_GPU_DEVICE = 0
    DEFAULT_NUM_WORKERS = 12
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_RESAMPLE_IMAGE_SIZE = (512, 512)
    DEFAULT_RESAMPLE_SHAPE = (512, 512, 64)
    DEFAULT_ZOOM_FACTORS = (1, 1, 1)
    DEFAULT_SAVE_MASKS = False
    DEFAULT_VISUALIZE_RESULTS = False
    
    # Required path keys
    REQUIRED_PATHS = ['dicom_root_dir', 'model_checkpoint_file', 'score_file', 'mask_folder', 'filtered_file']    
    def __init__(self, config_path):
        """Initialize configuration from JSON file."""
        self.config_data = self._load_json(config_path)
        #self._validate_required_paths()
        self._set_processing_defaults()
        self._set_output_defaults()
        self._set_paths()
        
    def _load_json(self, config_path):
        """Load and parse JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _validate_required_paths(self):
        """Validate that all required path configurations are present."""
        paths_section = self.config_data.get('paths', {})
        missing_paths = []
        
        for required_path in self.REQUIRED_PATHS:
            if required_path not in paths_section:
                missing_paths.append(required_path)
        
        if missing_paths:
            print(f"Error: Missing required path configurations: {missing_paths}")
            sys.exit(1)
    
    def _set_processing_defaults(self):
        """Set processing parameters with defaults if not specified."""
        processing = self.config_data.get('processing', {})
        self.gpu_device = processing.get('gpu_device', self.DEFAULT_GPU_DEVICE)
        self.num_workers = processing.get('num_workers', self.DEFAULT_NUM_WORKERS)
        self.batch_size = processing.get('batch_size', self.DEFAULT_BATCH_SIZE)
        
        # Convert lists back to tuples for image processing
        self.resample_image_size = tuple(processing.get('resample_image_size', self.DEFAULT_RESAMPLE_IMAGE_SIZE))
        self.resample_shape = tuple(processing.get('resample_shape', self.DEFAULT_RESAMPLE_SHAPE))
        self.zoom_factors = tuple(processing.get('zoom_factors', self.DEFAULT_ZOOM_FACTORS))
    
    def _set_output_defaults(self):
        """Set output parameters with defaults if not specified."""
        output = self.config_data.get('output', {})
        
        self.save_masks = output.get('save_masks', self.DEFAULT_SAVE_MASKS)
        self.visualize_results = output.get('visualize_results', self.DEFAULT_VISUALIZE_RESULTS)
    
    def _set_paths(self):
        """Set path configurations (already validated as required)."""
        paths = self.config_data['paths']
        
        self.dicom_root_dir = paths['dicom_root_dir']
        self.model_checkpoint_file = paths['model_checkpoint_file']
        self.output_dir = paths['output_dir']
        #self.score_file = paths['score_file']
        #self.filtered_file = paths['filtered_file']  # Add this line
        #self.mask_folder = paths['mask_folder']

def load_config(config_path='config.json'):
    """Convenience function to load configuration."""
    return Config(config_path)