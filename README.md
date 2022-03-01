# Python Cluster Tracking

This project gives an implementation of a cluster tracking algorithm used a perform stable clustering throughout a temporal domain.
An implementation of three variations is given in `cluster_tracker.py`. 

## Evaluation

We evaluated the different implementations and measured both accuracy and execution times of each approach.
The code used for the evaluation is stored in `delta.py`.

The data used for the evaluation (`multi_days_{lat,lon}.nc3` and `single_day_{lat,lon}.nc3` by Rene Schubert) are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Visualization

The file `plot_data.py` contains the code to plot all data for four consecutive days.
This includes the raw data, as well as the clusters and tracked clusters.