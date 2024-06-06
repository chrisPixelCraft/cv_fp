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

## Usage

1. Extract frames from the video:

```bash
python src/data_preparation.py
```

2. Extract features from the frames:

```bash
python src/feature_extraction.py
```

3. Train the model:

```bash
python src/train.py
```

4. Predict door states from test videos, generate output.json (hasn't finished)

```bash
python src/algorithm_template.py
```

5. Evaluate the score (hasn't finished)

```bash
python src/evaluate.py
```
