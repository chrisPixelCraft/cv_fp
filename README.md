# cvfp_DL

## Door State Monitoring using Hybrid CNN-RNN Model

## Overview

This project uses a hybrid CNN-RNN model to monitor the state of doors in public transit systems. The model combines ResNet-50 for feature extraction and LSTM for temporal sequence modeling.

## Directory Structure

```plaintext
cvfp_vivotek/
├── data/
│   ├── raw_videos/
│   ├── test_videos/
│   ├── features/
│   ├── features_test/
│   ├── frames/
│   ├── frames_test/
│   ├── labels/
├── src/
│   ├── models/model_epoch_49.pth
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── train.py
│   ├── algorithm_template.py
│   ├── evaluate.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Environment building: run the following scripts

1. Create a virtual environment and activate it:

```bash
conda create -n cvfp_vivotek python==3.6.13
conda activate cvfp_vivotek
```

2. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

- note: it is recommended to use gpu machine, please commend out pytorch and torchvision in requirements.txt, and install the corrusponding cuda version.

## Usage

### Testing
> please put testing video under *data/test_videos*

1. Extract frames from the video. This will generate video frames under *data/frames_test*:

```bash
cd src
python data_preparation.py
```


2. Extract features from the frames. This will generate features for model under *data/features_test*:

```bash
python feature_extraction.py
```

3. Predict door states from test videos:
```bash
python algorithm_template.py
```

After that, the *output.json* file will be generated under *src/*

## Training
> please put training video under *data/raw_videos*, and *labels.json* under *data/labels*

1. Extract frames from the video:

```bash
cd src
python data_preparation.py --training
```

2. Extract features from the frames:

```bash
python feature_extraction.py --training
```

3. Training

```bash
python train.py
```
