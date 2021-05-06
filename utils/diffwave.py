import sys
import json
import os

PACKAGE_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(PACKAGE_PATH, 'tf-diffwave'))

from model import DiffWave
from model.config import Config

sys.path.pop()
from config import load_state

def pretrained_diffwave() -> DiffWave:
    """Load pretrained diffwave.
    Returns:
        loaded diffwave.
    """
    with open(os.path.join(PACKAGE_PATH, 'pretrained/diffwave/l1.json')) as f:
        config = Config()
        load_state(config, json.load(f)['model'])

    diffwave = DiffWave(config)
    diffwave.restore(
        os.path.join(PACKAGE_PATH, 'pretrained/diffwave/l1/l1_1000000.ckpt-1')
    ).expect_partial()
    return diffwave
