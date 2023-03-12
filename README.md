# EEG-VAEs-pytorch
## Requirements
Python>= 3.5
Pytorch >=1.3
Pytorch lightning >= 0.6.0

## Usage

### Train VAEs
python train.py --config configs/<config-file-name.yaml>

### Project EEg data to representative features
python project.py --config configs/<config-file-name.yaml>

### Evaluate Patient profile/ cpc outcome
python evaluate.py
