## SGD_Armijo - Stochastic Line Search [[paper]](https://arxiv.org/abs/1905.09997)

We propose the optimizer SGD_Armijo, a stochastic line-search method 
that achieves superior generalization score and convergence speed.

### 0. Concerns
- The code does not yet work with `dropout`.

### 1. A minimal example
Run the following command for illustrating the SGD_Armijo optimizer,
```
python example.py
```

It will train MLP with SGD_Armijo on MNIST for 5 epochs.

### 2. Using SGD_Armijo
  1. `pip install --upgrade git+https://github.com/IssamLaradji/sls.git`
  2. define your optimizer as something like,
  ```
  import sls
  opt = sls.SGD_Armijo(model.parameters(),
                       n_batches_in_epoch=len(train_loader))
  ```

### 3. How is it different from other torch optimizers?

1) SGD_Armijo needs the number of batches in an epoch. It can be obtained from
`train_loader` like this,
    ```
    sls.SGD_Armijo(model.parameters(), n_batches_in_epoch=len(train_loader))
    ```
2) SGD_Armijo needs a closure when it makes a step like `opt.step(closure)`. The closure should only compute
and return the loss without calling `loss.backward()`. Here is an example.

    - For Adam it is, 
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
            loss.backward()
            return loss
        ```
        
    - For SGD_Armijo it is, 
        ```
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
          
            return loss          
        ```


