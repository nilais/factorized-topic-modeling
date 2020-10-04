# Factorized topic modeling

This repository contains the code associated with our contribution at the ML4Jets anomaly detection workshop in January of 2020.

## Repository structure
```
.
│   README.md
│   quark_gluon.ipynb
│   anomaly_LHCO.ipynb
│
└───src
│   │   loader.py
│   │   solver.py
│   │   util.py
│   
└───data
│   │   {move source file here}.h5
│
└───figs
```

We have provided sample notebooks for the anomaly detection and quark-gluon discrimination tasks. Due to the large size of the original data files, we are unable to upload them to Github. The anomaly detection dataset can be downloaded from the following [link](https://zenodo.org/record/3547722), and we will release the quark-gluon discrimination dataset soon. Within the source directory, we provide utilities for loading both datasets from their source files, which must be downloaded separately and placed in the `data` folder.
