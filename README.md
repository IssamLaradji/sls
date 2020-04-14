## Sls - Stochastic Line Search (NeurIPS2019) [[paper]](https://arxiv.org/abs/1905.09997)[[video]](https://www.youtube.com/watch?v=3Jx0tuZ1ERs)

Train faster and better with the SLS optimizer. The following 3 steps are there for getting started.

### 1. Install requirements

`pip install -r requirements.txt`

### 2. Mnist experiment

`python trainval.py -e mnist -sb ../results -d ../data -r 1`

where `-e` is the experiment group, `-sb` is the result directory, and `-d` is the dataset directory.

### 3. Cifar100 experiment

`python trainval.py -e cifar100 -sb ../results -d ../data -r 1`


#### Visualize

To view the results, create the jupyter file as follows and run its cells,
```
python create_jupyter.py
```

![alt text](neurips2019/cifar100.jpg)


#### Citation

```
@inproceedings{vaswani2019painless,
  title={Painless stochastic gradient: Interpolation, line-search, and convergence rates},
  author={Vaswani, Sharan and Mishkin, Aaron and Laradji, Issam and Schmidt, Mark and Gidel, Gauthier and Lacoste-Julien, Simon},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3727--3740},
  year={2019}
}
```
