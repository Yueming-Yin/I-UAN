# Improved Universal Adaptation Network (I-UAN)
Code release for **[Pseudo-Margin-Based Universal Domain Adaptation](https://www.sciencedirect.com/science/article/abs/pii/S0950705121005773)**
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

## Ciatations
# In Latex
```
@article{yin2021pseudo,
  title={Pseudo-margin-based universal domain adaptation},
  author={Yin, Yueming and Yang, Zhen and Wu, Xiaofu and Hu, Haifeng},
  journal={Knowledge-Based Systems},
  volume={229},
  pages={107315},
  year={2021},
  publisher={Elsevier}
}
```
# In Word
```
Yin, Yueming, Zhen Yang, Xiaofu Wu, and Haifeng Hu. "Pseudo-margin-based universal domain adaptation." Knowledge-Based Systems 229 (2021): 107315.
```
