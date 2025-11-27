import torch


def _weighted_mean(x: torch.Tensor, w: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute weighted mean along dim with numerical stability.
    Shapes must be broadcastable between x and w.
    """
    weighted_sum = (x * w).sum(dim=dim, keepdim=True)
    weight_total = w.sum(dim=dim, keepdim=True).clamp_min(eps)
    return weighted_sum / weight_total


def _masked_si_snr(pred: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor, masking_p: float = 0.8, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute SI-SNR over masked frames.

    Args:
        pred: [B, S, C, T]
        target: [B, S, C, T]
        frame_mask: [B, S, 1, T] with 1 for valid frames, 0 for masked-out
        masking_p: float, the likelihood to apply masking

    Returns:
        si_snr: [B, S]
    """
    # Apply masking
    if masking_p < 1.0:
        # Generate random values per batch, and set mask to all-ones if random > masking_p
        rand_vals = torch.rand(frame_mask.shape[0], device=frame_mask.device)
        reset_mask = rand_vals > masking_p
        if reset_mask.any():
            frame_mask[reset_mask] = 1.0

    # zero-mean per-channel using masked mean
    pred_mean = _weighted_mean(pred, frame_mask, dim=-1, eps=eps)
    target_mean = _weighted_mean(target, frame_mask, dim=-1, eps=eps)
    pred_zm = pred - pred_mean
    target_zm = target - target_mean

    # Compute projection of pred onto target using masked inner products
    # Numerator: <pred_zm, target_zm>
    # Denominator: ||target_zm||^2
    dot_num = (pred_zm * target_zm * frame_mask).sum(dim=(-2, -1), keepdim=True)  # sum over C,T
    dot_den = (target_zm * target_zm * frame_mask).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    s_target = dot_num / dot_den * target_zm

    e_noise = pred_zm - s_target

    # Energy terms with mask
    s_target_energy = (s_target * s_target * frame_mask).sum(dim=(-2, -1))  # [B, S]
    e_noise_energy = (e_noise * e_noise * frame_mask).sum(dim=(-2, -1)).clamp_min(eps)

    si_snr = 10.0 * torch.log10(s_target_energy.clamp_min(eps) / e_noise_energy)
    return si_snr


def get_loss_func(masking_p: float = 0.8):
    """
    Factory for masked SI-SNR loss.

    Expects:
        output['waveform']: [B, S, C, T]
        target['waveform']: [B, S, C, T]
        target.get('source_mask'): [B, S] in {0,1} (optional)
        target.get('frame_mask'): [B, S, 1, T] in {0,1} (optional)
    """

    def loss_func(output, target):
        pred = output['waveform']  # [B, S, C, T]
        gt = target['waveform']    # [B, S, C, T]

        B, S, C, T = pred.shape

        device = pred.device
        dtype = pred.dtype

        # Source mask: [B, S]
        source_mask = target.get('source_mask', None)
        if source_mask is None:
            source_mask = torch.ones((B, S), dtype=dtype, device=device)
        else:
            source_mask = source_mask.to(device=device, dtype=dtype)

        # Frame mask per source: [B, S, 1, T]
        frame_mask = target.get('frame_mask', None)
        if frame_mask is None:
            frame_mask = torch.ones((B, S, 1, T), dtype=dtype, device=device)
        else:
            frame_mask = frame_mask.to(device=device, dtype=dtype)

        si_snr_vals = _masked_si_snr(pred, gt, frame_mask, masking_p)  # [B, S]

        # Average only over present sources (and implicitly valid frames via frame_mask)
        weighted = si_snr_vals * source_mask
        denom = source_mask.sum().clamp_min(1.0)
        avg_si_snr = weighted.sum() / denom

        loss = -avg_si_snr
        return {"loss": loss}

    return loss_func


