import numpy as np
from scipy.spatial import cKDTree
import star_detection
from catalog_parsing import radec_to_tangent_plane
import matplotlib.pyplot as plt


# Update this based on your optical setup or calculate from FITS header
DEGREES_PER_PIXEL = 0.0003  # ≈1.08 arcsec/pixel

def build_tilewise_quad_kdtrees(tile_folder):
    import os
    from glob import glob

    tile_files = glob(os.path.join(tile_folder, 'quads_ra*_dec*.npz'))

    trees = []
    invariants_list = []
    indices_list = []
    tile_keys_list = []

    print(f"Found {len(tile_files)} tile files. Building KD-trees...")

    for filename in tile_files:
        data = np.load(filename)
        tree = cKDTree(data['invariants'])

        trees.append(tree)
        invariants_list.append(data['invariants'])
        indices_list.append(data['indices'])
        tile_keys_list.append(data['tile_keys'])

    print(f"Finished building {len(trees)} KD-trees.")
    return trees, invariants_list, indices_list, tile_keys_list


def compute_similarity_transform(catalog_pts, image_pts):
    vec_cat = catalog_pts[1] - catalog_pts[0]
    vec_img = image_pts[1] - image_pts[0]

    norm_cat = np.linalg.norm(vec_cat)
    norm_img = np.linalg.norm(vec_img)

    if norm_cat < 1e-8 or norm_img < 1e-8:
        raise ValueError('Degenerate quad — baseline too small')

    scale = norm_img / norm_cat
    theta = np.arctan2(vec_img[1], vec_img[0]) - np.arctan2(vec_cat[1], vec_cat[0])

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    translation = image_pts[0] - (catalog_pts[0] @ R.T) * scale
    return R, translation, scale


def apply_transform(catalog_xy, R, translation, scale):
    return (catalog_xy @ R.T) * scale + translation


def verify_pose(R, translation, scale, catalog_xy_deg, image_xy,
                tolerance=5.0, degrees_per_pixel=DEGREES_PER_PIXEL):
    catalog_xy = catalog_xy_deg / degrees_per_pixel  # Convert to pixels
    projected_catalog = apply_transform(catalog_xy, R, translation, scale)

    tree = cKDTree(image_xy)
    matches = tree.query_ball_point(projected_catalog, r=tolerance)
    return sum(1 for m in matches if m)


def plot_pose_solution(image_data, image_star_coords, catalog_radec,
                        R, translation, scale, ra_center, dec_center,
                        degrees_per_pixel=DEGREES_PER_PIXEL):
    cat_x, cat_y = radec_to_tangent_plane(
        catalog_radec[:, 0], catalog_radec[:, 1],
        ra_center_deg=ra_center, dec_center_deg=dec_center
    )
    cat_x /= degrees_per_pixel
    cat_y /= degrees_per_pixel
    catalog_xy = np.column_stack((cat_x, cat_y))

    catalog_projected = apply_transform(catalog_xy, R, translation, scale)

    try:
        vmin, vmax = ZScaleInterval().get_limits(image_data)
        if vmin == vmax:
            raise ValueError
    except Exception:
        vmin = np.percentile(image_data, 5)
        vmax = np.percentile(image_data, 99)

    plt.figure(figsize=(12, 8))
    plt.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Pose Solution: Image vs Catalog')

    plt.scatter(image_star_coords[:, 0], image_star_coords[:, 1],
                facecolors='none', edgecolors='cyan', s=60, label='Detected Stars')

    plt.scatter(catalog_projected[:, 0], catalog_projected[:, 1],
                color='red', marker='x', s=50, label='Catalog (Projected)')

    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.legend(loc='upper right')
    plt.show()


def match_and_pose_solve(image_quads, image_stars,
                          kdtrees, invariants_list, indices_list, tile_keys_list,
                          catalog_stars_radec, quad_star_indices,
                          match_threshold=10, tolerance=5.0, tol_invariant=0.005,
                          debug_plot=False):
    print('Searching catalog for matches...')
    image_xy = image_stars[:, :2]

    for img_idx, query_vec in enumerate(image_quads):
        for tree, invariants, indices, tile_keys in zip(
                kdtrees, invariants_list, indices_list, tile_keys_list):

            result_indices = tree.query_ball_point(query_vec, r=tol_invariant)

            for idx in result_indices:
                cat_star_indices = indices[idx]
                cat_quad_radec = catalog_stars_radec[cat_star_indices, :2]

                ra_center = np.mean(cat_quad_radec[:, 0])
                dec_center = np.mean(cat_quad_radec[:, 1])

                cat_quad_x, cat_quad_y = radec_to_tangent_plane(
                    cat_quad_radec[:, 0], cat_quad_radec[:, 1],
                    ra_center_deg=ra_center, dec_center_deg=dec_center
                )
                cat_quad_x /= DEGREES_PER_PIXEL
                cat_quad_y /= DEGREES_PER_PIXEL
                catalog_pts = np.column_stack((cat_quad_x, cat_quad_y))

                img_star_indices = quad_star_indices[img_idx]
                image_pts = image_stars[img_star_indices, :2]

                try:
                    R, translation, scale = compute_similarity_transform(catalog_pts[:2], image_pts[:2])
                except Exception as e:
                    print(f"Pose computation failed: {e}")
                    continue

                if debug_plot:
                    cat_quad_transformed = apply_transform(catalog_pts, R, translation, scale)
                    plt.figure(figsize=(6, 6))
                    plt.scatter(image_pts[:, 0], image_pts[:, 1], color='cyan', label='Image Quad')
                    plt.scatter(cat_quad_transformed[:, 0], cat_quad_transformed[:, 1],
                                color='red', marker='x', label='Catalog Quad (Transformed)')
                    plt.title('Quad Alignment Check')
                    plt.xlabel('X pixel')
                    plt.ylabel('Y pixel')
                    plt.legend()
                    plt.gca().invert_yaxis()
                    plt.show()

                cat_full_x, cat_full_y = radec_to_tangent_plane(
                    catalog_stars_radec[:, 0], catalog_stars_radec[:, 1],
                    ra_center_deg=ra_center, dec_center_deg=dec_center
                )
                cat_full_xy = np.column_stack((cat_full_x, cat_full_y))

                n_matches = verify_pose(R, translation, scale, cat_full_xy, image_xy,
                                        tolerance=tolerance, degrees_per_pixel=DEGREES_PER_PIXEL)

                print(f"Quad {img_idx} → Tile {tile_keys[idx]} → Matches: {n_matches}")
                print(f"Pose: scale={scale}, R=\n{R}, translation={translation}")

                if n_matches >= match_threshold:
                    print('Pose solution found.')
                    return {
                        'R': R,
                        'translation': translation,
                        'scale': scale,
                        'n_matches': n_matches,
                        'img_quad_idx': img_idx,
                        'cat_quad_idx': idx,
                        'tile_bin': tuple(tile_keys[idx]),
                        'ra_center': ra_center,
                        'dec_center': dec_center
                    }

    print('No valid pose found.')
    return None


# Main runner
if __name__ == '__main__':
    quad_folder = 'tycho2_star_catalog/quad_tiles'

    print("Loading catalog quads...")
    trees, invariants_list, indices_list, tile_keys_list = build_tilewise_quad_kdtrees(quad_folder)

    print("Preparing image...")
    fits_path = 'fits_data/soul_nebula.fits'
    data = star_detection.read_fits(fits_path)
    background_subtracted, _ = star_detection.subtract_gradient(data, show=False)

    star_coords = star_detection.detect_stars(background_subtracted, threshold=5, sigma=2)
    print(f"Detected {len(star_coords)} stars.")

    image_quads, quad_star_indices = star_detection.generate_quads_from_image(
        star_coords, max_stars=30, max_base_length=np.inf
    )
    print(f"Generated {len(image_quads)} image quads.")

    print("Loading catalog...")
    from catalog_parsing import load_catalog
    catalog_file = 'tycho2_star_catalog/catalog.dat'
    catalog_stars, catalog_mags = load_catalog(catalog_file)

    result = match_and_pose_solve(
        image_quads=image_quads,
        image_stars=star_coords,
        kdtrees=trees,
        invariants_list=invariants_list,
        indices_list=indices_list,
        tile_keys_list=tile_keys_list,
        catalog_stars_radec=catalog_stars,
        quad_star_indices=quad_star_indices,
        match_threshold=10,
        tolerance=5.0,
        tol_invariant=0.005,
        debug_plot=False
    )

    if result:
        print("\nPose solution found:")
        for key, value in result.items():
            print(f"{key}: {value}")

        plot_pose_solution(
            image_data=background_subtracted,
            image_star_coords=star_coords,
            catalog_radec=catalog_stars,
            R=result['R'],
            translation=result['translation'],
            scale=result['scale'],
            ra_center=result['ra_center'],
            dec_center=result['dec_center'],
            degrees_per_pixel=DEGREES_PER_PIXEL
        )
    else:
        print("\nNo valid pose found.")
