from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass, find_objects
import matplotlib.pyplot as plt
from itertools import combinations


def read_fits(fits_path):

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

def detect_stars(data, threshold=5.0, sigma=2.0, min_size=2, max_size=200):
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

    # Estimate FWHM from image
    fwhm_pixels = estimate_fwhm_from_image(data, threshold=4.5, sigma=1.5)

    if fwhm_pixels is None:
        print("FWHM estimation failed. Using default size filters.")
    else:
        # Compute size thresholds from FWHM
        min_size = max(2, int(0.5 * fwhm_pixels ** 2))
        max_size = int(10 * fwhm_pixels ** 2)

    keep = (sizes >= min_size) & (sizes <= max_size)
    keep_labels = labels[keep]

    if len(keep_labels) == 0:
        return np.empty((0, 2))

    mask_keep = np.isin(labeled, keep_labels)

    relabeled, new_num = label(mask_keep)

    centroids = center_of_mass(mask_keep, relabeled, range(1, new_num + 1))
    centroids = np.array(centroids)

    xy_coords = centroids[:, ::-1]  # (x, y) not (row, col)

    # Compute brightness (sum of pixel values within each star region)
    brightness = []
    for label_id in range(1, new_num + 1):
        region_mask = (relabeled == label_id)
        total_flux = data[region_mask].sum()
        brightness.append(total_flux)

    brightness = np.array(brightness)

    result = np.column_stack((xy_coords, brightness))  # (x, y, brightness)

    return result

def plot_stars(data, detected_stars):

    plt.figure(figsize=(12,8))
    plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax = np.percentile(data, 99))
    plt.scatter(detected_stars[:,0], detected_stars[:,1], edgecolor='blue', facecolor='none', s=60, label='Detected Stars')

    plt.title('Star Detection Overlay')
    plt.xlabel('X-pixel')
    plt.ylabel('Y-pixel')
    plt.legend()
    plt.show()

def generate_quads_from_image(star_coords, max_stars=30, max_base_length=np.inf):

    if star_coords.shape[0] < 4:
        return np.empty((0,4)), np.empty((0,4), dtype=int)

    # sort by descending brightness
    sorted_indices = np.argsort(-star_coords[:,2])
    selected = star_coords[sorted_indices[:max_stars], :2]

    quads_invariant=[]
    quad_indices=[]

    for (i,j,k,l) in combinations(range(len(selected)), 4):
        idx = [i,j,k,l]
        pts = selected[idx]

        dists = {}
        for m, n in combinations(range(4), 2):
            d = np.linalg.norm(pts[m] - pts[n])
            dists[(m,n)] = d

        (m1, m2), base_length = max(dists.items(), key=lambda x: x[1])

        if base_length > max_base_length:
            continue    #skip large quads

        p1, p2 = pts[m1], pts[m2]
        delta = p2 - p1
        theta = -np.arctan2(delta[1], delta[0])
        scale = 1/np.linalg.norm(delta)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]) * scale

        transformed = (pts - p1) @ R.T

        rest_indices = [n for n in range(4) if n not in (m1, m2)]
        xy = transformed[rest_indices]

        xy = xy[np.lexsort((xy[:,1], xy[:,0]))]

        (x1, y1), (x2, y2) = xy

        quads_invariant.append([x1, y1, x2, y2])
        quad_indices.append([sorted_indices[i] for i in [m1, m2, rest_indices[0], rest_indices[1]]])

    quads_invariant = np.array(quads_invariant, dtype=np.float32)
    quad_indices = np.array(quad_indices, dtype=np.int32)

    return quads_invariant, quad_indices

if __name__ == '__main__':

    # example
    fits_path='fits_data/soul_nebula.fits'

    data = read_fits(fits_path)

    background_subtracted, _ = subtract_gradient(data, show=False)

    star_coords = detect_stars(background_subtracted, threshold=5, sigma=2)
    print(star_coords)

    plot_stars(background_subtracted, star_coords)

    # After star detection:
    image_quads, quad_star_indices = generate_quads_from_image(
        star_coords,
        max_stars=40,
        max_base_length=np.inf  # Optional filter on max base length in pixels
    )

    print(f"Generated {len(image_quads)} quads from image")