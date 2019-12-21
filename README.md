## Sls - Stochastic Line Search (NeurIPS2019) [[paper]](https://arxiv.org/abs/1905.09997)[[video]](https://www.youtube.com/watch?v=3Jx0tuZ1ERs)

Train faster and better with the SLS optimizer. The following 3 steps are there for getting started.

1. `pip install --upgrade git+https://github.com/IssamLaradji/sls.git`
2. Define your optimizer as something like,
  ```
  import sls
  opt = sls.Sls(model.parameters())
  # or as follows for better line search
  # opt = sls.Sls(model.parameters(), n_batches_per_epoch=len(train_loader))
  ```

3. Update your model as follows,

   ```
   for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        opt.zero_grad()
        
        # this closure is necessary for performing line search
        def closure():
            probs = F.log_softmax(model(images), dim=1)
            loss = F.nll_loss(probs, labels, reduction="sum")
          
            return loss
            
    opt.step(closure)
    ```
  
  
### Experiments

## CIFAR100

- Run `python trainval.py -e cifar100 -sb <SAVEDIR_BASE>`
  - `<SAVEDIR_BASE>` is the directory where the experiments are saved.
- View the results by running `python view_plots.py -e cifar100 -sb <SAVEDIR_BASE>`. This saves the following plot,


![alt text](results/cifar100.jpeg)

## MNIST

- Run `python trainval.py -e mnist -sb <SAVEDIR_BASE>`
  - `<SAVEDIR_BASE>` is the directory where the experiments are saved.
- View the results by running `python view_plots.py -e mnist -sb <SAVEDIR_BASE>`.


<!-- ![alt text](results/mnist.jpeg) -->

