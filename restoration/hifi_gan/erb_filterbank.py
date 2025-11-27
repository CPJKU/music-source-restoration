# ---------- ERB filterbank (design-time, numpy) ----------

import numpy as np


class FilterBank(object):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        self.leny = leny
        self.fs = fs
        self.N = N
        self.low_lim = low_lim
        self.high_lim, self.freqs, self.nfreqs = self.check_limits(leny, fs, high_lim)

    def check_limits(self, leny, fs, high_lim):
        if np.remainder(leny, 2) == 0:
            nfreqs = leny / 2
            max_freq = fs / 2
        else:
            nfreqs = (leny - 1) / 2
            max_freq = fs * (leny - 1) / 2 / leny
        freqs = np.linspace(0, max_freq, int(nfreqs) + 1)
        if high_lim > fs / 2:
            high_lim = max_freq
        return high_lim, freqs, int(nfreqs)

class EqualRectangularBandwidth(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(EqualRectangularBandwidth, self).__init__(leny, fs, N, low_lim, high_lim)
        erb_low = self.freq2erb(self.low_lim)
        erb_high = self.freq2erb(self.high_lim)
        erb_lims = np.linspace(erb_low, erb_high, self.N + 2)
        self.cutoffs = self.erb2freq(erb_lims)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def freq2erb(self, freq_Hz):
        return 9.265 * np.log(1 + np.divide(freq_Hz, 24.7 * 9.265))

    def erb2freq(self, n_erb):
        return 24.7 * 9.265 * (np.exp(np.divide(n_erb, 9.265)) - 1)

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = np.zeros([nfreqs + 1, N])
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]
            l_ind = np.min(np.where(freqs > l_k))
            h_ind = np.max(np.where(freqs < h_k))
            avg = (self.freq2erb(l_k) + self.freq2erb(h_k)) / 2
            rnge = self.freq2erb(h_k) - self.freq2erb(l_k)
            cos_filts[l_ind:h_ind + 1, k] = np.cos(
                (self.freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * np.pi
            )
        filters = np.zeros([nfreqs + 1, N + 2])
        filters[:, 1:N + 1] = cos_filts
        h_ind = np.max(np.where(freqs < cutoffs[1]))
        filters[:h_ind + 1, 0] = np.sqrt(1 - np.power(filters[:h_ind + 1, 1], 2))
        l_ind = np.min(np.where(freqs > cutoffs[N]))
        filters[l_ind:nfreqs + 1, N + 1] = np.sqrt(
            1 - np.power(filters[l_ind:nfreqs + 1, N], 2)
        )
        return filters

def make_erb_filterbank_for_stft(
    sr: int,
    n_fft: int,
    n_filters: int = 128,
    low_lim: float = 50.0,
    high_lim: float | None = None,
) -> np.ndarray:
    """
    Returns ERB filterbank aligned with STFT bins:
      fbanks: (n_filters, n_fft//2 + 1)
    """
    if high_lim is None:
        high_lim = sr / 2.0

    erb_fb = EqualRectangularBandwidth(
        leny=n_fft,
        fs=sr,
        N=n_filters,
        low_lim=low_lim,
        high_lim=high_lim,
    )
    # erb_fb.filters: (n_freqs+1, N+2); band-pass filters are 1..N
    fbanks_np = erb_fb.filters[:, 1:-1].T   # (N, n_freqs+1)
    return fbanks_np.astype(np.float32)
