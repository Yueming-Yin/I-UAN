# Improved Universal Adaptation Networks (I-UAN)
Code release for Unveiling Class-Labeling Structure for Universal Domain Adaptation (arxiv: 2010.04873)
## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- download datasets

- write your config file

- `python main.py --config train-config.yaml`

- train (configurations in `officehome-train-config.yaml` are only for officehome dataset):

  `python main.py --config train-config.yaml`

- test

  `python main.py --config test-config.yaml`
  
- monitor (tensorboard required)

  `tensorboard --logdir .`
