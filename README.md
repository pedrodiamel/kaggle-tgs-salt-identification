
# Kaggle: TGS Salt Identification Challenge
Segment salt deposits beneath the Earth's surface


## Results

| Branch   | Name     | CV      | LB    | Description               |
|---------:|---------:|:-------:|:-----:|:--------------------------|
| classic  | unet     |         |       |                           |
|          | unetrest | 0.8134  | 0.782 | pre-train resnet135       |
|          |          |         |       |                           |
| depth    |          |         |       |                           |
|          |          |         |       |                           |
|          |          |         |       |                           |
|          |          |         |       |                           |



## Installation
       
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

