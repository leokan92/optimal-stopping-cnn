# Solving the Optional stopping problem of Bermudan options with CNN

## Abstract:

To be created

## Requirements

- Python 3.8
- [fbm 0.3.0](https://pypi.org/project/fbm/)
- [timesynth 0.2.4](https://github.com/TimeSynth/TimeSynth)
- [Pytorch 1.8.1](https://pytorch.org/)

## Usage

First, install prerequisites

```
$ pip install fbm
$ pip install timesynth
$ pip3 install torch==1.8.1
```

Check pytorch address for the compatible version

To train and test the model use [main.py](/main.py) calling the libraries that train and test the specific models

To use real data, add the file to the Dataset file and replace the file variable in main:

```python
file = r'\SP500- daily - 30Y_train.csv'
```

## Plotting Results

There is a file used for [plotting](/plotting_results.py)


## References

If you re-use this work, please cite:

```
@article{Felizardo2021,
  Title                    = {Solving the Optional stopping problem of Bermudan options with CNN},
  Author                   = {Felizardo, Leonardo},
  journal                  = {},
  Year                     = {2021},
  volume                   = {},
  number                   = {},
  pages                    = {},
}
```







