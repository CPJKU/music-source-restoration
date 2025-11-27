from torchmetrics.functional.audio import signal_noise_ratio as snr


def get_metric_func():
    def metric_func(output, target):
        snr_val = snr(output['waveform'], target['waveform']).mean()
        metric_dict = {
            'snr': snr_val,
        }
        return metric_dict
    return metric_func