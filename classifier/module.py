import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

# from cifar10_models.densenet import densenet121, densenet161, densenet169
# from cifar10_models.googlenet import googlenet
# from cifar10_models.inception import inception_v3
# from cifar10_models.mobilenetv2 import mobilenet_v2
# from cifar10_models.resnet import resnet18, resnet34, resnet50
from vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from scheduler import WarmupCosineLR
from pytorch_metric_learning import losses, miners, regularizers

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    # "resnet18": resnet18(),
    # "resnet34": resnet34(),
    # "resnet50": resnet50(),
    # "densenet121": densenet121(),
    # "densenet161": densenet161(),
    # "densenet169": densenet169(),
    # "mobilenet_v2": mobilenet_v2(),
    # "googlenet": googlenet(),
    # "inception_v3": inception_v3(),
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        #self.criterion = torch.nn.CrossEntropyLoss()
        #self.miner = miners.MultiSimilarityMiner()
        self.criterion = losses.TripletMarginLoss(smooth_loss=True)
        #self.criterion = losses.CentroidTripletLoss()

        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]

    # def accuracy(self):
    #     #put the kmeans alg here
    #     return 0

    # def forward(self, batch):
    #     images, labels = batch
    #     # predictions = self.model(images)
    #     # loss = self.criterion(predictions, labels)
    #     # accuracy = self.accuracy(predictions, labels)
    #     embeddings = self.model(images)
    #     # hard_pairs = self.miner(embeddings, labels)
    #     loss = self.criterion(embeddings, labels)
    #     accuracy = self.accuracy(embeddings, labels)
    #     return loss, accuracy * 100

    def forward(self, batch):
        images, labels = batch
        x = self.model.features(images)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        embeddings = self.model.classifier(x)
        predictions = self.model.classifier2(embeddings)
        loss = self.criterion(embeddings, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]