import torchvision
import tqdm
import pandas as pd

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sls import optim


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
    model = MLP(n_classes=10).cuda()

    # Run Optimizer
    opt = optim.sgd_armijo.SGD_Armijo(model,
                                      n_batches_in_epoch=len(train_loader))

    result_dict = []
    for epoch in range(5):
        # 1. Compute loss over train loader
        print("Evaluating Epoch %d" % epoch)
        loss_sum = 0.
        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            loss_sum += compute_loss(model, images, labels)

        loss_avg = float(loss_sum / len(train_set))
        result_dict += [{"loss_avg":loss_avg, "epoch":epoch}]

        # 2. Train over train loader
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

class MLP(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[1000],
                 n_classes=10, bias=True):
        super().__init__()

        self.input_size = input_size
        self.output_size = n_classes
        self.squeeze_output = False
        self.act = F.relu

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = self.act(Z)
        logits = self.output_layer(out)

        return logits

if __name__ == "__main__":
    main()
