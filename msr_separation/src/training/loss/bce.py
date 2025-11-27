import torch
import torch.nn.functional as F

def get_loss_func():
    def loss_func(output, target):
        target_prob = target['probabilities']
        output_prob = output['probabilities']
        with torch.cuda.amp.autocast(enabled=False):
            loss_val = F.binary_cross_entropy(output_prob.float(), target_prob.float(), reduction='mean')

        loss_dict = {
            'loss': loss_val,
        }
        return loss_dict
    return loss_func