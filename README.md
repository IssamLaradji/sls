## Painless Stochastic Gradient [[paper]](https://arxiv.org/abs/1905.09997)

We propose the optimizer SGD_Armijo, a stochastic line-search method 
that achieves superior generalization score and convergence speed.

### 1. Running a minimal example with SGD_Armijo optimizer
Run the following command,
```
python example.py
```

It will train MLP with SGD_Armijo on MNIST for 5 epochs.

### 2. Using the SGD_Armijo optimizer

  1. copy  the folder`sls` into your project; then
  2. define your optimizer something like,
  ```
  from sls import optim
  opt = optim.sgd_armijo.SGD_Armijo(model,
                                    n_batches_in_epoch=len(train_loader))
  ```

### 3. How is it different from other torch optimizers?

1) SGD_Armijo needs `model` in its constructor instead of `model.parameters()`. Here is an example.
    - For Adam it is, `torch.optim.Adam(model.parameters(), ...)`;
    - For SGD_Armijo it is, `optim.sgd_armijo.SGD_Armijo(model, ...)`.

2) SGD_Armijo needs the number of batches in an epoch as,
    ```
    optim.sgd_armijo.SGD_Armijo(model, n_batches_in_epoch=len(train_loader))
    ```
3) SGD_Armijo needs a closure when it makes a step like `opt.step(closure)`. The closure should only compute
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
