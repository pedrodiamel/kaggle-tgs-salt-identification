
# Kaggle: TGS Salt Identification Challenge
Segment salt deposits beneath the Earth's surface

## Solution (branch)

- draka *** (place 210)
- durotan
- garona

## Results

| Branch   | Name     | CV      | LB (Public)    | LB (Private)   | Description               |
|---------:|---------:|:-------:|:--------------:|:--------------:|:--------------------------|
| classic  | unet     |         |                |                |                           |
|          | unetrest | 0.8134  | 0.840          | 0.868          | pre-train resnet135 + tta |
|          |          |         |                |                |                           |

- top10%: 210 place
- https://www.kaggle.com/c/tgs-salt-identification-challenge/leaderboard

## Installation

    $git clone https://github.com/pedrodiamel/pytorchvision.git
    $cd pytorchvision
    $python setup.py install
    $pip install -r installation.txt

## Download Kaggle dataset
    
    # loader dataset 
    kaggle competitions download -c tgs-salt-identification-challenge

## Visualize result with Visdom

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/


## How use

### Step 1: Create dataset

    #(1) kaggle dataset
    ./run_createdataset.sh 

### Step 2: Train

    ./train.sh
    
### Step 3: Submission

    ./submission.sh

## URLs


## Others

