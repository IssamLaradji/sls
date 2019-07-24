import sls
import torchvision
import tqdm
import pandas as pd
import models

from torch.nn import functional as F
from torch.utils.data import DataLoader



def main():
    print("Dataset: MNIST - Model: MLP - Optimizer: SGD_Armijo")

    # Load MNIST
    train_set = torchvision.datasets.MNIST("data", train=True,
                                     download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.5,), (0.5,))
                                     ]))
    train_loader = DataLoader(train_set, drop_last=True, shuffle=True, batch_size=128)

    # Create model
    model = models.MLP(n_classes=10, dropout=True).cuda()

    # Run Optimizer
    opt = sls.SGD_Armijo(model.parameters(),
                         n_batches_in_epoch=len(train_loader))

    result_dict = []
    for epoch in range(5):
        # 1. Compute loss over train loader
        model.eval()
        print("Evaluating Epoch %d" % epoch)

        loss_sum = 0.
        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            loss_sum += compute_loss(model, images, labels)

        loss_avg = float(loss_sum / len(train_set))
        result_dict += [{"loss_avg":loss_avg, "epoch":epoch}]

        # 2. Train over train loader
        model.train()
        print("Training Epoch %d" % epoch)

        for images,labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            opt.zero_grad()
            closure = lambda : compute_loss(model, images, labels)
            opt.step(closure)

        print(pd.DataFrame(result_dict))


def compute_loss(model, images, labels):
    probs = F.log_softmax(model(images), dim=1)
    loss = F.nll_loss(probs, labels, reduction="sum")

    return loss


if __name__ == "__main__":
    main()
