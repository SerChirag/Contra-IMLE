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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

        self.miner = miners.MultiSimilarityMiner()
        self.tripletlossmetric = losses.TripletMarginLoss(margin=1, smooth_loss=True)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]
        self.tsne = TSNE(random_state=0)

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

    def execute_model(self, batch):
        images, labels = batch
        x = self.model.features(images)
        x = self.model.avgpool(x)
        embeddings = x.view(x.size(0), -1)
        x = self.model.classifier(embeddings)
        predictions = self.model.classifier2(x)
        return embeddings, predictions, labels

    def display_results(self, embeddings, labels, batch_nb):
        if batch_nb == 99:
            train_tsne_embeds = self.tsne.fit_transform(embeddings.cpu().detach().numpy())
            print(train_tsne_embeds)
            plt.scatter(train_tsne_embeds[:, 0], train_tsne_embeds[:, 1], c=labels.cpu().numpy())
            plt.show()

    def forward(self, batch):
        embeddings, predictions, labels = self.execute_model(batch)
        loss1, loss2, accuracy = self.calculate_loss_acc(embeddings, predictions, labels)
        return loss1, loss2, accuracy

    def calculate_loss_acc(self, embeddings, predictions, labels):
        accuracy = self.accuracy(predictions, labels)
        hard_pairs = self.miner(embeddings, labels)
        triplet_loss = self.tripletlossmetric(embeddings, labels, hard_pairs)
        classifier_loss = self.criterion(predictions, labels)
        return triplet_loss, classifier_loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        embeddings, predictions, labels = self.execute_model(batch)
        triplet_loss, cls_loss, accuracy = self.calculate_loss_acc(embeddings, predictions, labels)
        #self.display_results(embeddings, labels, batch_nb)

        self.log("loss/train_triplet", triplet_loss)
        self.log("loss/train_cls", cls_loss)
        self.log("loss/train", triplet_loss + cls_loss)
        self.log("acc/train", accuracy)
        return triplet_loss + cls_loss

    def validation_step(self, batch, batch_nb):
        triplet_loss, cls_loss, accuracy = self.forward(batch)
        self.log("loss/val_triplet", triplet_loss)
        self.log("loss/val_cls", cls_loss)
        self.log("loss/val", triplet_loss + cls_loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        triplet_loss, cls_loss, accuracy = self.forward(batch)
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
