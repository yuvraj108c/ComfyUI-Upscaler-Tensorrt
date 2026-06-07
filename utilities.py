import requests
from tqdm import tqdm
import logging
import sys
import os
import json

class ColoredLogger:
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'RESET': '\033[0m'
    }

    LEVEL_COLORS = {
        'DEBUG': COLORS['BLUE'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['MAGENTA']
    }

    def __init__(self, name="MY-APP"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.app_name = name
        
        # Prevent message propagation to parent loggers
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Custom formatter class to handle colored components
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                # Color the level name according to severity
                level_color = ColoredLogger.LEVEL_COLORS.get(record.levelname, '')
                colored_levelname = f"{level_color}{record.levelname}{ColoredLogger.COLORS['RESET']}"
                
                # Color the logger name in blue
                colored_name = f"{ColoredLogger.COLORS['BLUE']}{record.name}{ColoredLogger.COLORS['RESET']}"
                
                # Set the colored components
                record.levelname = colored_levelname
                record.name = colored_name
                
                return super().format(record)
        
        # Create formatter with the new format
        formatter = ColoredFormatter('[%(name)s|%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)


    def debug(self, message):
        self.logger.debug(f"{self.COLORS['BLUE']}{message}{self.COLORS['RESET']}")

    def info(self, message):
        self.logger.info(f"{self.COLORS['GREEN']}{message}{self.COLORS['RESET']}")

    def warning(self, message):
        self.logger.warning(f"{self.COLORS['YELLOW']}{message}{self.COLORS['RESET']}")

    def error(self, message):
        self.logger.error(f"{self.COLORS['RED']}{message}{self.COLORS['RESET']}")

    def critical(self, message):
        self.logger.critical(f"{self.COLORS['MAGENTA']}{message}{self.COLORS['RESET']}")

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

def download_file(url, save_path):
    """
    Download a file from URL with progress bar
    
    Args:
        url (str): URL of the file to download
        save_path (str): Path to save the file as
    """
    GREEN = '\033[92m'
    RESET = '\033[0m'
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        colour='green',
        bar_format=f'{GREEN}{{l_bar}}{{bar}}{RESET}{GREEN}{{r_bar}}{RESET}' 
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def get_final_resolutions(width, height, resize_to, scale=4):
    final_width = None
    final_height = None
    aspect_ratio = float(width/height)

    match resize_to:
        case "HD":
            final_width = 1280
            final_height = 720
        case "FHD":
            final_width = 1920
            final_height = 1080
        case "2k":
            final_width = 2560
            final_height = 1440
        case "4k":
            final_width = 3840
            final_height = 2160
        case "none":
            final_width = width*scale
            final_height = height*scale

        case _:
            resize_factor = float(resize_to.split('x')[0])
            final_width = width*resize_factor
            final_height = height*resize_factor

    if aspect_ratio == 1.0:
        final_width = final_height

    if aspect_ratio < 1.0 and resize_to not in ("none", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"):
        temp = final_width
        final_width = final_height
        final_height = temp

    return (int(final_width), int(final_height)) # must be whole numbers

def load_node_config(config_filename="load_upscaler_config.json"):
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)

    default_config = {
        "models": {
            "4x-UltraSharp": {"path": "models/4x-UltraSharp.engine", "scale": 4}
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision"
        }
    }
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config: {config_filename}")
        return config
    except Exception as e:
        logger.warning(f"Config load failed: {e}, using fallback")
        return default_config

LOAD_UPSCALER_NODE_CONFIG = load_node_config()

def get_model_scale(model_name):
    for m in LOAD_UPSCALER_NODE_CONFIG.get("models", []):
        if m["name"] == model_name:
            return m["scale"]
    return 4