"""
Package initialisation for the modeling module.  This module contains the
data loading utilities, dataset preparation logic, model architecture and
training/inference scripts for the Tech Challenge LSTM project.

The module has the following submodules:

* ``dataio`` – helpers for loading data from Azure Blob Storage or the local
  filesystem and uploading artefacts back to Blob Storage.
* ``dataset`` – functions to transform raw DataFrames into supervised
  learning datasets, including sliding window generation and feature scaling.
* ``model`` – the definition of the LSTM architecture used for predicting
  future closing prices.
* ``train`` – a script that orchestrates the end‑to‑end training pipeline,
  from loading data and engineering features to training the model and
  evaluating/saving artefacts.
* ``predict`` – a simple inference script that loads the trained model and
  scaler, prepares the latest data and returns a prediction.

This layout promotes separation of concerns and makes it easy to swap out
components (e.g. experimenting with different models or feature sets) in a
structured manner.
"""
