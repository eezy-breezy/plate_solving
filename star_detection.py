from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass, find_objects
import matplotlib.pyplot as plt


def read_fits(fits_name):

    #load in image data from fits file
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)

    return data

def subtract_gradient(data, sigma=100, show=True):

    print('subtracting gradients...')
    background = gaussian_filter(data, sigma)
    background_subtracted = data - 0.8*background
    print('gradient subtraction finished.')

    if show:
        fig, axes = plt.subplots(1,2, figsize=(12,12))
        axes[0].imshow(background_subtracted, cmap='gray', origin='lower', vmin=np.percentile(background_subtracted, 5), vmax=np.percentile(background_subtracted, 99))
        axes[0].set_title('Background_subtracted')

        axes[1].imshow(background, cmap='gray', origin='lower', vmin=np.percentile(background, 5), vmax=np.percentile(background, 99))
        axes[1].set_title('Background')

        plt.tight_layout()
        plt.show()

    return background_subtracted, background

def estimate_fwhm_from_image(data, threshold=5.0, sigma=2.0):
    median = np.median(data)
    std = np.std(data)

    smoothed = gaussian_filter(data, sigma)
    mask = smoothed > (median + threshold * std)

    labeled, num_features = label(mask)

    if num_features == 0:
        print("no features detected")
        return None

    sizes = np.bincount(labeled.ravel())[1:]
    sizes = sizes[sizes > 1]    #skip single pixels

    if len(sizes) == 0:
        print("no valid blobs found for FWHM estimation")
        return None

    median_area = np.median(sizes)

    fwhm_pixels = 2 * np.sqrt(median_area / np.pi)

    return fwhm_pixels

def detect_stars(data, threshold=5.0, sigma=2.0, min_size=None, max_size=None):
    median = np.median(data)
    std = np.std(data)

    smoothed = gaussian_filter(data, sigma=sigma)

    threshold_level = median + threshold * std
    mask = smoothed > threshold_level

    labeled, num_features = label(mask)

    if num_features == 0:
        return np.empty((0, 2))

    sizes = np.bincount(labeled.ravel())[1:]  # skip label 0
    labels = np.arange(1, num_features + 1)

    if min_size is None:
        min_size = 2
    if max_size is None:
        max_size = 9999999  # effectively no upper limit if not set

    keep = (sizes >= min_size) & (sizes <= max_size)
    keep_labels = labels[keep]

    if len(keep_labels) == 0:
        return np.empty((0, 2))

    mask_keep = np.isin(labeled, keep_labels)

    relabeled, new_num = label(mask_keep)

    centroids = center_of_mass(mask_keep, relabeled, range(1, new_num + 1))
    centroids = np.array(centroids)

    xy_coords = centroids[:, ::-1]  # (x, y) not (row, col)

    return xy_coords

def plot_stars(data, detected_stars):

    plt.figure(figsize=(12,12))
    plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax = np.percentile(data, 99))
    plt.scatter(detected_stars[:,0], detected_stars[:,1], edgecolor='blue', facecolor='none', s=60, label='Detected Stars')

    plt.title('Star Detection Overlay')
    plt.xlabel('X-pixel')
    plt.ylabel('Y-pixel')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # example
    fits_path='fits_data/bodes_cigar.fit'

    data = read_fits(fits_path)

    background_subtracted, _ = subtract_gradient(data, show=False)

    # Estimate FWHM from image
    fwhm_pixels = estimate_fwhm_from_image(background_subtracted, threshold=4.5, sigma=1.5)

    if fwhm_pixels is None:
        print("FWHM estimation failed. Using default size filters.")
        min_size, max_size = 2, 100
    else:
        # Compute size thresholds from FWHM
        min_size = max(2, int(0.5 * fwhm_pixels ** 2))
        max_size = int(10 * fwhm_pixels ** 2)


    star_coords = detect_stars(background_subtracted, threshold=1, min_size=min_size, max_size=max_size)
    print(star_coords)

    plot_stars(background_subtracted, star_coords)