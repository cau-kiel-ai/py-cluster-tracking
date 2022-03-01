"""
This script evaluates the implemented tracking algorithms against a single data set.
Both accuracy and execution times are measured and saved to `results.csv`.
"""

import logging
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN

from cluster_tracker import GridTracker, CentroidTracker, PairwiseTracker

lat = xr.open_dataset('single_day_lat.nc3').latitude.isel(ts=0)
lon = xr.open_dataset('single_day_lon.nc3').longitude.isel(ts=0)

eps = 10/6371
minpts = 30

df = pd.DataFrame(columns=['name', 'accuracy', 'time'])

logging.info('Start')

db = DBSCAN(eps=eps, min_samples=minpts)

tracker = {
    'Centroid-based': CentroidTracker(eps/2),
    'Grid-based ($\\epsilon/10000$)': GridTracker([(3.78, 34.69), (-44.93, -32.002)], (1300000, 600000), 0.9),
    'Grid-based ($\\epsilon$)': GridTracker([(3.78, 34.69), (-44.93, -32.002)], (130, 60), 0.9),
    'Grid-based ($\\epsilon/100$)': GridTracker([(3.78, 34.69), (-44.93, -32.002)], (1300, 600), 0.9),
    'Grid-based ($\\epsilon/1000$)': GridTracker([(3.78, 34.69), (-44.93, -32.002)], (130000, 60000), 0.9),
    'Pairwise Distance (KD-Tree; $\\epsilon$)': PairwiseTracker(eps/1, 0.9, algorithm='kd_tree'),
    'Pairwise Distance (Brute; $\\epsilon$)': PairwiseTracker(eps/1, 0.9, algorithm='brute'),
    'Pairwise Distance (KD-Tree; $\\epsilon/10$)': PairwiseTracker(eps/10, 0.9, algorithm='kd_tree'),
    'Pairwise Distance (Brute; $\\epsilon/10$)': PairwiseTracker(eps/10, 0.9, algorithm='brute'),
}

X = np.vstack((lon, lat)).T
X = np.radians(X)
db.fit(X)

for name, tracker in tracker.items():
    tracker.next(X, db.labels_)
    for i in range(5):
        times = []
        start = timer()
        diffs = tracker.next(X, db.labels_)
        end = timer()
        num_correct = len(list(filter(lambda x: x[0] == x[1], diffs)))
        times.append(timedelta(seconds=end-start).total_seconds())
        df = df.append(
            {"name": name,
             "accuracy": num_correct/len(diffs),
             "time": timedelta(seconds=end-start).total_seconds()
             }, ignore_index=True
        )

df.to_csv('results.csv', index=False)
logging.info('STOP')
