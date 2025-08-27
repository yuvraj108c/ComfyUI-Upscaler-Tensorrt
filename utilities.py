import requests
from tqdm import tqdm
import logging
import sys

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

def get_final_resolutions(width, height, resize_to):
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
            final_width = width*4
            final_height = height*4
        case "2x":
            final_width = width*2
            final_height = height*2
        case "3x":
            final_width = width*3
            final_height = height*3

    if aspect_ratio == 1.0:
        final_width = final_height

    if aspect_ratio < 1.0 and resize_to not in ("none", "2x", "3x"):
        temp = final_width
        final_width = final_height
        final_height = temp

    return (final_width, final_height)