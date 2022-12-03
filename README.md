# ST-Curriculum Dropout in Spatial-Temporal Graph Modeling
<p align="center">
<img src="img/framework.png" width="100%" height="50%">
</p>

## Descriptions
Source code of the AAAI'23: ST-Curriculum Dropout in Spatial-Temporal Graph Modeling

## Requirements
* `Python==3.8`
* `pytorch==1.7.1`
* `torch-summary (>= 1.4.5)`
you will get some error if you installed torchsummary, see the details at https://pypi.org/project/torch-summary/. please uninstall torchsummary and run pip install torch-summary to install the new one.


#### Running
  `python main.py`

#### Dataset
We provide sample data under data/.

The project structure is organized as follows:
```
├── data
│   └── METRLA 
│       ├── metr-la.h5    # signal observation
│       ├── W_metrla.csv  # adj maxtrix
├── img
│   └── framework.png # image of model framework
├── models
│   ├──  STGCN.py # STGCN framework
│   ├──  Param.py # hyper parameter 
├── save
├── main.py
├── README.md
└── utils
    ├── Metrics.py # evaluation metrics 
    ├── Utils.py  
```
