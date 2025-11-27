import numpy as np
import torch
from numba import jit, prange
from torch.autograd import Function

@jit(nopython=True, parallel=True, fastmath=True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.full((B, N + 2, M + 2), np.inf, dtype=D.dtype)
  R[:, 0, 0] = 0.0
  diag_max = N + M
  for k in prange(B):
    for diag in range(2, diag_max + 1):
      row_start = max(1, diag - M)
      row_end = min(N, diag - 1)
      if row_end < row_start:
        continue
      for i in range(row_start, row_end + 1):
        j = diag - i
        if j < 1 or j > M:
          continue
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = r0
        if r1 > rmax:
          rmax = r1
        if r2 > rmax:
          rmax = r2
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  return R

@jit(nopython=True, parallel=True, fastmath=True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
  E = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in prange(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.tensor(gamma, device=dev, dtype=dtype)
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R_np = compute_softdtw(D_, g_)
    R = torch.from_numpy(R_np).to(device=dev, dtype=dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E_np = compute_softdtw_backward(D_, R_, g_)
    E = torch.from_numpy(E_np).to(device=dev, dtype=dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma = gamma
    self.func_dtw = _SoftDTW.apply

  def calc_distance_matrix(self, x, y):
    dist = torch.cdist(x, y, p=2)
    return dist.pow(2)

  def forward(self, x, y):
    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) < 3:
      x = x.unsqueeze(0)
      y = y.unsqueeze(0)
      squeeze = True
    if self.normalize:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      D_xx = self.calc_distance_matrix(x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      D_yy = self.calc_distance_matrix(y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      result = out_xy - 0.5 * (out_xx + out_yy)
    else:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      result = out_xy
    return result.squeeze(0) if squeeze else result
