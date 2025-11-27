from torchmetrics.functional.audio import signal_noise_ratio as snr

def get_loss_func():
    def loss_func(output, target):
        snr_val = - snr(output['waveform'], target['waveform']).mean()
        loss_dict = {
            'loss': snr_val, # main loss, for back propagation
        }
        return loss_dict
    return loss_func

if __name__ == '__main__':
    # cd ../../..
    
    import os, sys; os.chdir('../../..'); sys.path.append(os.getcwd())
    import importlib
    def initialize_config(module_cfg):
        module = importlib.import_module(module_cfg["module"])
        if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
        return getattr(module, module_cfg["main"])()
    
    config = {
        "module": 'src.training.loss.snr',
        'main': 'get_loss_func',
    }
    lossfunc = initialize_config(config)
    
    import torch
    output = {'waveform': torch.rand(3, 3, 1, 480000)}
    target = {'waveform': torch.rand(3, 3, 1, 480000)}
    loss = lossfunc(output, target)
    for k,v in loss.items(): print(f'{k}: {v}')

