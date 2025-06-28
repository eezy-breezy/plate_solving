import numpy as np


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


def radec_to_cartesian(ra_deg, dec_deg):
    """
    Convert RA/Dec in degrees to 3D Cartesian unit vector (x, y, z).
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    return np.column_stack((x, y, z))


def filter_catalog_by_magnitude(catalog_array, magnitude_array, mag_limit):
    """
    Filter the catalog based on a magnitude limit.
    """
    return catalog_array[magnitude_array <= mag_limit]


if __name__ == '__main__':
    #example
    catalog_file = 'tycho2_star_catalog/catalog.dat'
    stars, mags = load_catalog(catalog_file)

    print(f"Loaded {len(stars)} stars")

    mag_limit=12.0
    filtered_stars = filter_catalog_by_magnitude(stars, mags, mag_limit=mag_limit)
    print(f"{len(filtered_stars)} stars brighter than magnitude {mag_limit}")