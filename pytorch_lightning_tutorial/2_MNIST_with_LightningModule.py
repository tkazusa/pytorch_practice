import torch
import torchvision.transforms as T
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl

from net import Net
from pytorch_lightning import LightningModule
from torchvision.datasets import MNIST

class ImageClassifier(LightningModule):
    def __init__(self, model=None, lr=1.0, gamma=0.7, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model or Net()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]
    
    
if __name__ == "__main__":
    print('here')
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size)
    net = Net()
    trainer = pl.Trainer()
    trainer.fit(net, train_loader, test_loader)