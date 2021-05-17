# Solving the Optional stopping problem of Bermudan options with CNN

## Abstract:

To be created

## Requirements

- Python 3.6
- [gym](https://github.com/openai/gym)
- [Keras 2.4.3](https://pypi.org/project/Keras/)
- [TensorFlow 1.12.0](https://pypi.org/project/tensorflow/)
- [ta 0.5.25](https://pypi.org/project/ta/)
- [Empyrical 0.5.5](https://pypi.org/project/empyrical/)
- [Scikit-learn 0.20.0](https://pypi.org/project/scikit-learn/)
- [Pytorch 1.7.0](https://pytorch.org/)

## Usage

First, install prerequisites

```
$ pip install gym
$ pip install keras==2.4.3
$ pip install ta
$ pip install empyrical
$ pip install -U scikit-learn
```

Check pytorch address for the compatible version

To train and test the model use [main.py](/main.py) calling the libraries that train and test the specific models

To use real data, add the file to the Dataset file and replace the file variable in main:

```python
file = r'\SP500- daily - 30Y_train.csv'
```

## Plotting Results

There is a file used for [plotting](/plotting_results.py)

<p align="center">
    <img src="https://raw.githubusercontent.com/leokan92/Contextual-bandit-Resnet-trading/main/images/test_btc.png?token=AINPHV254E7JCKAETMAPYVK72FHK6" width="640"\>
</p>


## References



If you re-use this work, please cite:

```
@article{Felizardo2021,
  Title                    = {Solving the Optional stopping problem of Bermudan options with CNN},
  Author                   = {Felizardo, Leonardo},
  journal                  = {},
  Year                     = {2020},
  volume                   = {},
  number                   = {},
  pages                    = {},
}
```







