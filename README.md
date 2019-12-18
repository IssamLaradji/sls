<<<<<<< HEAD
## SgdArmijo - Stochastic Line Search [[paper]](https://arxiv.org/abs/1905.09997)

We propose the optimizer SgdArmijo, a stochastic line-search method 
that achieves superior generalization score and convergence speed. 
The script below is how it can be used in a training loop.

### 1. Quickstart
Run the following command for illustrating the SgdArmijo optimizer,
```
python trainval.py -d mnist -m mlp
```

It will train MLP with SgdArmijo on MNIST for 5 epochs.

Run `python trainval.py -d cifar10 -m resnet34`

It will train ResNet34 with SgdArmijo on CIFAR10 for 5 epochs.

### 2. Using it in your code

```
opt = sls.SgdArmijo(model.parameters(),
                    n_batches_in_epoch=len(train_loader))
=======
## Sls - Stochastic Line Search (NeurIPS2019) [[paper]](https://arxiv.org/abs/1905.09997)[[video]](https://www.youtube.com/watch?v=3Jx0tuZ1ERs)

We propose the optimizer Sls, a stochastic line-search method 
that achieves superior generalization score and convergence speed. 
The script below is how it can be used in a training loop.

### 0. Highlights

SLS trains ResNet-34 faster and better than Adam on CIFAR100,

![alt text](Slides/sls.jpeg)

### 1. Quickstart
Run the following command for illustrating the Sls optimizer,
```
python trainval.py -e mnist
```


### 2. Using it in your code

```
opt = sls.Sls(model.parameters())
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67
                       
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    opt.zero_grad()
    
    def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
          
            return loss
            
    opt.step(closure)
```

<<<<<<< HEAD
### 3. Things to note
- The code does not yet work with `dropout`.
- Make sure the network is overparametrized for the problem.


### 4. Using SgdArmijo
=======

### 3. Using Sls
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67
  1. `pip install --upgrade git+https://github.com/IssamLaradji/sls.git`
  2. define your optimizer as something like,
  ```
  import sls
<<<<<<< HEAD
  opt = sls.SgdArmijo(model.parameters(),
                       n_batches_in_epoch=len(train_loader))
  ```

### 5. How is it different from other torch optimizers?

1) SgdArmijo needs the number of batches in an epoch. It can be obtained from
`train_loader` like this,
    ```
    sls.SgdArmijo(model.parameters(), n_batches_in_epoch=len(train_loader))
    ```
2) SgdArmijo needs a closure when it makes a step like `opt.step(closure)`. The closure should only compute
=======
  opt = sls.Sls(model.parameters())
  ```

### 4. How is it different from other torch optimizers?

1) Sls needs the number of batches in an epoch. It can be obtained from
`train_loader` like this,
    ```
    sls.Sls(model.parameters(), n_batches_per_epoch=len(train_loader))
    ```
2) Sls needs a closure when it makes a step like `opt.step(closure)`. The closure should only compute
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67
and return the loss without calling `loss.backward()`. Here is an example.

    - For Adam it is, 
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
            loss.backward()
            return loss
        ```
        
<<<<<<< HEAD
    - For SgdArmijo it is, 
=======
    - For Sls it is, 
>>>>>>> 88b9e8dd238f7e125e67430c5f9e96533a878d67
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
          
            return loss          
        ```
