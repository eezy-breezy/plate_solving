import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree

def generate_quads(points, max_base_length=None):
    """
    Generate quads from a set of 2D points
    """
    quads = []

    for (i,j,k,l) in combinations(range(len(points)),4):

        idx = [i,j,k,l]
        pts = points[idx]

        #compute pairwise distances
        dists = {}
        for m, n in combinations(range(4), 2):
            d = np.linalg.norm(pts[m] - pts[n])
            dists[(m,n)] = d

        #determine longest side
        (m1, m2), base_length = max(dists.items(), key=lambda x: x[1])

        if max_base_length is not None and base_length > max_base_length:
            continue

        p1 = pts[m1]
        p2 = pts[m2]
        delta = p2 - p1
        theta = -np.arctan2(delta[1], delta[0])
        scale = 1 / np.linalg.norm(delta)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]) * scale

        transformed = (pts - p1) @ R.T

        # select the other two points
        rest_indices = [n for n in range(4) if n not in (m1, m2)]
        xy = transformed[rest_indices]

        # sort for canonical order
        xy = xy[np.lexsort((xy[:,1], xy[:,0]))] # sort by x then y

        (x1, y1), (x2, y2) = xy

        quad = (idx[m1], idx[m2], idx[rest_indices[0]], idx[rest_indices[1]], x1, y1, x2, y2)

        quads.append(quad)

    return quads

def build_quad_kdtree(quads):

    invariants = np.array([[q[4], q[5], q[6], q[7]] for q in quads])
    index_pairs = [(q[0], q[1], q[2], q[3]) for q in quads]

    tree = cKDTree(invariants)

    return tree, invariants, index_pairs