from torchmetrics.functional.audio import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)

def get_loss_func():
    def loss_func(output, target):
        mask = output['source_mask'] # [bs, n_source]
        # snr(output['waveform'], target['waveform'])*mask[:, :, None]
        snr_val = - snr(output['waveform'], target['waveform'])
        snr_val = snr_val*mask[:, :, None]
        snr_val = snr_val.mean()
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
        "module": 'src.training.loss.snr_withmask',
        'main': 'get_loss_func',
    }
    lossfunc = initialize_config(config)
    
    import torch
    output = {'waveform': torch.rand(3, 3, 1, 480000), 'output_mask': torch.tensor([[1., 0., 0.], [0., 0., 1.], [1., 0., 1.]])}
    target = {'waveform': torch.rand(3, 3, 1, 480000)}
    loss = lossfunc(output, target)
    for k,v in loss.items(): print(f'{k}: {v}')

