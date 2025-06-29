import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import os


def parse_catalog_line(line):
    """
    Parse a single catalog line to extract RA, Dec, and Magnitude.
    """
    parts = line.strip().split('|')

    ra = parts[2].strip()
    dec = parts[3].strip()
    mag = parts[17].strip() if len(parts) > 17 else ''

    if ra == '' or dec == '':
        return None

    try:
        ra = float(ra)
        dec = float(dec)
        mag = float(mag) if mag != '' else np.inf  # If magnitude missing, set to very faint

    except ValueError:
        print('Value error in parsing catalog')
        return None

    return ra, dec, mag


def load_catalog(catalog_file):
    """
    Load the catalog into numpy arrays of (RA, Dec) and Magnitude.
    """
    stars = []
    mags = []

    with open(catalog_file, 'r') as f:
        for line in f:
            result = parse_catalog_line(line)
            if result:
                ra, dec, mag = result
                stars.append((ra, dec))
                mags.append(mag)

    stars = np.array(stars)  # (N, 2)
    mags = np.array(mags)    # (N,)

    return stars, mags


def radec_to_tangent_plane(ra_deg, dec_deg, ra_center_deg, dec_center_deg):
    """
    Project RA/Dec (degrees) onto a tangent plane centered at (ra_center, dec_center).

    Returns (x, y) in degrees on the tangent plane.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra_center_deg)
    dec0 = np.deg2rad(dec_center_deg)

    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)

    x = (np.cos(dec) * np.sin(ra - ra0)) / cos_c
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c

    # Convert radians to degrees
    x_deg = np.rad2deg(x)
    y_deg = np.rad2deg(y)

    return x_deg, y_deg


def filter_catalog_by_magnitude(catalog_array, magnitude_array, mag_limit):
    """
    Filter the catalog based on a magnitude limit.
    """
    return catalog_array[magnitude_array <= mag_limit]


def assign_tile(ra, dec, ra_bin_size=5, dec_bin_size=5):
    """
    Assign a star to an RA/Dec tile.
    RA wraps at 360 degrees.
    Dec runs from -90 to +90 degrees.
    """
    ra_bin = int(np.floor((ra % 360) / ra_bin_size))
    dec_bin = int(np.floor((dec + 90) / dec_bin_size))
    return (ra_bin, dec_bin)

def tile_catalog(catalog_array, ra_bin_size=5, dec_bin_size=5):
    """
    Tile catalog stars into RA/Dec bins
    """
    tiles = defaultdict(list)

    for idx, (ra,dec) in enumerate(catalog_array):
        bin_key = assign_tile(ra, dec, ra_bin_size, dec_bin_size)
        tiles[bin_key].append(idx)

    return tiles

def generate_quads_from_tile(stars_in_tile, mags_in_tile, max_stars=40, max_base_length=np.inf):
    """
    Generate quads from stars within a tile
    """

    if len(stars_in_tile) < 4:
        return []  # Not enough stars to form quads

        # Select brightest max_stars
    brightest_indices = np.argsort(mags_in_tile)[:max_stars]
    points = stars_in_tile[brightest_indices]

    quads = []

    from itertools import combinations

    for (i, j, k, l) in combinations(range(len(points)), 4):
        idx = [i, j, k, l]
        pts = points[idx]

        # Compute pairwise distances
        dists = {}
        for m, n in combinations(range(4), 2):
            d = np.linalg.norm(pts[m] - pts[n])
            dists[(m, n)] = d

        (m1, m2), base_length = max(dists.items(), key=lambda x: x[1])

        if base_length > max_base_length:
            continue  # Skip large quads

        # Transform: move m1 to (0, 0) and m2 to (1, 0)
        p1, p2 = pts[m1], pts[m2]
        delta = p2 - p1
        theta = -np.arctan2(delta[1], delta[0])
        scale = 1 / np.linalg.norm(delta)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]) * scale

        transformed = (pts - p1) @ R.T

        rest_indices = [n for n in range(4) if n not in (m1, m2)]
        xy = transformed[rest_indices]

        xy = xy[np.lexsort((xy[:, 1], xy[:, 0]))]  # sort for canonical order

        (x1, y1), (x2, y2) = xy

        quad = (idx[m1], idx[m2], idx[rest_indices[0]], idx[rest_indices[1]],
                x1, y1, x2, y2)

        quads.append(quad)

    return quads

def save_tile_quad_index(output_folder, tile_key, quads):
    """
    Save teh quads for a single tile for later lookup
    """
    ra_bin, dec_bin = tile_key

    if not quads:
        print(f"No quads for {tile_key}")
        return


    invariants = np.array([[q[4], q[5], q[6], q[7]] for q in quads], dtype=np.float32)
    indices = np.array([[q[0], q[1], q[2], q[3]] for q in quads], dtype=np.int32)
    tile_keys = np.tile(np.array(tile_key, dtype=np.int32), (len(quads), 1))

    filename = f'{output_folder}/quads_ra{ra_bin}_dec{dec_bin}.npz'
    np.savez_compressed(
        filename,
        invariants=invariants,
        indices=indices,
        tile_keys=tile_keys
    )
    print(f"Saved {len(quads)} quads to {filename}")

def load_tile_quad_index(filename):
    data = np.load(filename)
    invariants = data['invariants']
    indices = data['indices']
    tile_keys = data['tile_keys']
    print(f"Loaded {len(invariants)} quads from {filename}")
    return invariants, indices, tile_keys


if __name__ == '__main__':
    #example
    catalog_file = 'tycho2_star_catalog/catalog.dat'
    stars, mags = load_catalog(catalog_file)

    print(f"Loaded {len(stars)} stars")

    tiles = tile_catalog(stars, ra_bin_size=5, dec_bin_size=5)

    output_folder = 'tycho2_star_catalog/quad_tiles'
    os.makedirs(output_folder, exist_ok=True)

    #Generate quads for all tiles (only need to run this once).
    flag_file = 'tycho2_star_catalog/quad_tiles/finished.txt'   #file indicating all quad files have been generated
    if not os.path.exists(flag_file):
        print('Beginning quad computation (Should only have to run once to generate quad files)...')
        for tile_key, star_indices in tiles.items():
            tile_stars = stars[star_indices]
            tile_mags = mags[star_indices]

            # Calculate tile center in RA/Dec
            ra_bin, dec_bin = tile_key
            ra_center = (ra_bin + 0.5) * 5  # assuming ra_bin_size = 5
            dec_center = (dec_bin + 0.5) * 5 - 90  # assuming dec_bin_size = 5

            # Project RA/Dec to tangent plane (XY)
            x, y = radec_to_tangent_plane(
                tile_stars[:, 0], tile_stars[:, 1],
                ra_center, dec_center
            )
            tile_xy = np.column_stack((x, y))  # shape (N, 2)

            # Generate quads using XY projected positions
            tile_quads = generate_quads_from_tile(
                tile_xy, tile_mags, max_stars=30, max_base_length=np.inf
            )

            print(f"Tile {tile_key} → {len(tile_quads)} quads")

            save_tile_quad_index(output_folder, tile_key, tile_quads)
        # Create flag file indicating finished
        with open(flag_file, 'w') as f:
            f.write('Output generation complete.\n')

        print('All quads processed and saved')
    else:
        print('Quad files already generated')
