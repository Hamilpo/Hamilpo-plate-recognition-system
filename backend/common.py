"""
Funciones comunes para el backend
"""

import logging
from datetime import datetime

def setup_logging():
    """Configura el sistema de logging"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def get_timestamp() -> str:
    """
    Retorna un timestamp en formato legible
    
    Returns:
        String con timestamp
    """
    
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")