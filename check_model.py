import sys
sys.path.insert(0, 'src')
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.load('src/unifolm_vla/config/training/unifolm_vla_agibot_train.yaml')
cfg.framework.qwenvl.base_vlm = '/root/gpufree-data/unifolm-weights/unifolm-vlm-base'

print('Loading model...')
model = Unifolm_VLA(cfg)
print('Model structure:')
for name, param in model.named_parameters():
    print(name)
