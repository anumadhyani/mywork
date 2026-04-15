import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.restoration import denoise_tv_chambolle


FEATURE_NAMES = [
    "noise_mean",
    "noise_std",
    "noise_variance",
    "low_freq_energy",
    "high_freq_energy",
    "freq_ratio",
    "gradient_mean",
    "gradient_std",
    "local_var_mean",
    "local_var_std",
    "edge_density",
]


def extract_noise_features(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {name: np.nan for name in FEATURE_NAMES}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = denoise_tv_chambolle(gray, weight=0.1)
    noise_residual = gray - denoised

    features: dict[str, float] = {}
    features["noise_mean"] = float(np.mean(noise_residual))
    features["noise_std"] = float(np.std(noise_residual))
    features["noise_variance"] = float(np.var(noise_residual))

    fft = fft2(gray)
    fft_shift = fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)

    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2

    low_freq = magnitude_spectrum[
        center_h - h // 8 : center_h + h // 8,
        center_w - w // 8 : center_w + w // 8,
    ]
    features["low_freq_energy"] = float(np.mean(low_freq))

    high_freq_mask = np.ones_like(magnitude_spectrum)
    high_freq_mask[
        center_h - h // 4 : center_h + h // 4,
        center_w - w // 4 : center_w + w // 4,
    ] = 0
    high_freq = magnitude_spectrum * high_freq_mask
    high_freq_energy = float(np.mean(high_freq[high_freq > 0])) if np.any(high_freq > 0) else 0.0
    features["high_freq_energy"] = high_freq_energy

    features["freq_ratio"] = float(high_freq_energy / (features["low_freq_energy"] + 1e-10))

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    features["gradient_mean"] = float(np.mean(gradient_magnitude))
    features["gradient_std"] = float(np.std(gradient_magnitude))

    kernel_size = 5
    local_means = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    local_sq_means = cv2.blur((gray.astype(float)) ** 2, (kernel_size, kernel_size))
    local_variance = local_sq_means - local_means**2

    features["local_var_mean"] = float(np.mean(local_variance))
    features["local_var_std"] = float(np.std(local_variance))

    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = float(np.sum(edges > 0) / edges.size)

    return features


def extract_all_features(image_path: str) -> np.ndarray:
    features = extract_noise_features(image_path)
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
