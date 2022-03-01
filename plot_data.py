"""
This script plots the data for four days in multiple formats.
Raw, clustered, and tracked clustered data is plotted for each of these days.
The resulting figure is stored at `prototype.png`.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from cluster_tracker import GridTracker, CentroidTracker, PairwiseTracker

eps = 4/6371
minpts = 30
n = 4
zoom = False

lat = xr.open_dataset('multi_days_lat.nc3').latitude
lon = xr.open_dataset('multi_days_lon.nc3').longitude

sdb = DBSCAN(eps=eps, min_samples=minpts, metric='haversine')

pair_tracker = PairwiseTracker(eps*80, 0.9, algorithm='kd_tree')
cent_tracker = CentroidTracker(eps*400)
grid_tracker = GridTracker([(3.78, 34.69), (-44.93, -32.002)], (1300, 600), 0.7)

fig = plt.figure(figsize=(3*n, 15), constrained_layout=True)
subs = fig.subfigures(5, 1)
axs = list(map(lambda x: x.subplots(1, n), subs))

subs[0].suptitle('Raw Data')
subs[1].suptitle('DBSCAN Clusters')
subs[2].suptitle('Tracked DBSCAN Clusters (Pairwise-based)')
subs[3].suptitle('Tracked DBSCAN Clusters (Centroid-based)')
subs[4].suptitle('Tracked DBSCAN Clusters (Grid-based)')

pair_colors = dict()
cent_colors = dict()
grid_colors = dict()

stats_points = []
stats_clusters = []

for i in tqdm(range(n)):
    a = lon.isel(ts=i+4).dropna('no')
    b = lat.isel(ts=i+4).dropna('no')
    X = np.vstack((a, b)).T
    sdb.fit(np.radians(X))
    pair_diffs = pair_tracker.next(X, sdb.labels_)
    cent_diffs = cent_tracker.next(X, sdb.labels_)
    grid_diffs = grid_tracker.next(X, sdb.labels_)

    y = b[sdb.labels_ != -1]
    x = a[sdb.labels_ != -1]
    labels = np.asarray(sdb.labels_)[sdb.labels_ != -1]
    num_labels = len(set(labels))

    stats_clusters.append(num_labels)
    stats_points.append(len(X))

    pair_cols = labels.copy()
    cent_cols = labels.copy()
    grid_cols = labels.copy()
    pair_new_colors = dict()
    cent_new_colors = dict()
    grid_new_colors = dict()
    for new, old in pair_diffs:
        temp = pair_colors.get(old, old)
        pair_new_colors[new] = temp
    for new, old in cent_diffs:
        temp = cent_colors.get(old, old)
        cent_new_colors[new] = temp
    for new, old in grid_diffs:
        temp = grid_colors.get(old, old)
        grid_new_colors[new] = temp

    pair_colors = pair_new_colors
    cent_colors = cent_new_colors
    grid_colors = grid_new_colors

    for idx, elem in pair_colors.items():
        pair_cols[labels == idx] = elem
    for idx, elem in cent_colors.items():
        cent_cols[labels == idx] = elem
    for idx, elem in grid_colors.items():
        grid_cols[labels == idx] = elem

    factor = 1
    if zoom:  # Zoom
        factor = 20
        for ax in axs:
            ax[i].axis([15, 17, -37, -35])

    axs[0][i].set_title(f'Day {i}')

    axs[0][i].scatter(lon, lat, s=0.000005*factor, c='grey')

    # Lazy way of dynamically choosing the correct axis scale for all plots
    axs[1][i].scatter(lon, lat, s=0.0001*factor, c='white')
    axs[2][i].scatter(lon, lat, s=0.0001*factor, c='white')
    axs[3][i].scatter(lon, lat, s=0.0001*factor, c='white')
    axs[4][i].scatter(lon, lat, s=0.0001*factor, c='white')
    axs[1][i].scatter(x, y, s=0.001*factor, c=labels, cmap='tab20')
    axs[2][i].scatter(x, y, s=0.001*factor, c=pair_cols, cmap='tab20')
    axs[3][i].scatter(x, y, s=0.001*factor, c=cent_cols, cmap='tab20')
    axs[4][i].scatter(x, y, s=0.001*factor, c=grid_cols, cmap='tab20')

plt.savefig('prototype.png')
