## SgdArmijo - Stochastic Line Search [[paper]](https://arxiv.org/abs/1905.09997)

We propose the optimizer SgdArmijo, a stochastic line-search method 
that achieves superior generalization score and convergence speed.

### 0. Concerns
- The code does not yet work with `dropout`.

### 1. A minimal example
Run the following command for illustrating the SgdArmijo optimizer,
```
python example.py
```

It will train MLP with SgdArmijo on MNIST for 5 epochs.

### 2. Using SgdArmijo
  1. `pip install --upgrade git+https://github.com/IssamLaradji/sls.git`
  2. define your optimizer as something like,
  ```
  import sls
  opt = sls.SgdArmijo(model.parameters(),
                       n_batches_in_epoch=len(train_loader))
  ```

### 3. How is it different from other torch optimizers?

1) SgdArmijo needs the number of batches in an epoch. It can be obtained from
`train_loader` like this,
    ```
    sls.SgdArmijo(model.parameters(), n_batches_in_epoch=len(train_loader))
    ```
2) SgdArmijo needs a closure when it makes a step like `opt.step(closure)`. The closure should only compute
and return the loss without calling `loss.backward()`. Here is an example.

    - For Adam it is, 
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
            loss.backward()
            return loss
        ```
        
    - For SgdArmijo it is, 
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
          
            return loss          
        ```


