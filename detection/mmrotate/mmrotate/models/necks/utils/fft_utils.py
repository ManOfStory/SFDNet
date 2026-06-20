import torch

def fft2d(x):
    fft = torch.fft.fft2(x, norm='ortho')
    mag = torch.abs(fft)
    pha = torch.angle(fft)
    return mag, pha

def fft2_shift(x, norm='ortho'):
    dtype = x.dtype
    x = x.float()
    with torch.cuda.amp.autocast(enabled=False):
        X = torch.fft.fft2(x, norm=norm)
        X = torch.fft.fftshift(X, dim=(-2, -1))
    magnitude = torch.abs(X)
    phase = torch.angle(X)

    magnitude = magnitude.to(dtype)
    phase = phase.to(dtype)
    return magnitude, phase

def ifft2d(mag, pha):
    real = mag * torch.cos(pha)
    imag = mag * torch.sin(pha)
    fft = torch.complex(real, imag)
    x = torch.fft.ifft2(fft, norm='ortho').real
    return x

def ifft2_shift(magnitude, phase, norm='ortho'):
    """
    magnitude, phase: [B, C, H, W]
    return: reconstruct real tensor [B, C, H, W]
    """
    dtype = magnitude.dtype
    magnitude = magnitude.float()
    phase = phase.float()
    X = magnitude * torch.exp(1j * phase)
    X = torch.fft.ifftshift(X, dim=(-2, -1))
    x_recon = torch.fft.ifft2(X, norm=norm)
    x_recon = x_recon.to(dtype)
    return x_recon.real