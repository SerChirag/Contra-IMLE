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
import torch.nn as nn
import torch

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
        # self.allEmbeddings = torch.zeros(0, 512).cuda()
        # self.allLabels = torch.zeros(0).cuda()

        self.model = all_classifiers[self.hparams.classifier]
        self.tsne = TSNE(random_state=0)
        self.tripletlossconstant = 1
        # self.tripletlossconstant = self.hparams.triplet_constant
        #
        # self.tripletlossmetric = nn.TripletMarginLoss(margin=self.hparams.triplet_margin, reduction='mean')

    def calculate_centroids(self, embeddings, labels):
        positives = torch.zeros(embeddings.size())  # should be Batch x Embedding
        negatives = torch.zeros(embeddings.size())
        for i in range(embeddings.size(dim=0)):
            curLabel = labels[i]
            posLabels = labels == curLabel
            posLabels = posLabels.nonzero()
            negLabels = labels != curLabel
            negLabels = negLabels.nonzero()
            positiveExamples = embeddings[posLabels]
            # Negative samples should be changed, instead create one for each class, but this is much easier
            negativeExamples = embeddings[negLabels]
            mean_pos = torch.mean(positiveExamples, dim=0)
            mean_neg = torch.mean(negativeExamples, dim=0)
            positives[i] = mean_pos
            negatives[i] = mean_neg
        return positives.cuda(), negatives.cuda()

    def calculate_loss_acc2(self, embeddings, predictions, labels):
        accuracy = self.accuracy(predictions, labels)
        positive, negative = self.calculate_centroids(embeddings, labels)
        triplet_loss = self.tripletlossmetric(embeddings, positive, negative)
        classifier_loss = self.criterion(predictions, labels)
        return self.tripletlossconstant * triplet_loss, classifier_loss, accuracy * 100

    def execute_model(self, batch):
        images, labels = batch
        x = self.model.features(images)
        x = self.model.avgpool(x)
        embeddings = x.view(x.size(0), -1)
        x = self.model.classifier(embeddings)
        predictions = self.model.classifier2(x)
        return embeddings, predictions, labels

    def display_results(self, embeddings, labels):
        train_tsne_embeds = self.tsne.fit_transform(embeddings.cpu().detach().numpy())
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
        return self.tripletlossconstant * triplet_loss, classifier_loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        embeddings, predictions, labels = self.execute_model(batch)
        triplet_loss, cls_loss, accuracy = self.calculate_loss_acc(embeddings, predictions, labels)

        # self.allEmbeddings = torch.cat((self.allEmbeddings, embeddings), 0)
        # self.allLabels = torch.cat((self.allLabels, labels), 0)

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
        # if batch_nb == 4:
        #     self.separateByClass()

    def separateByClass(self):
        allClassEmbeddings = torch.zeros(10, 5000, self.allEmbeddings.size(1))
        for i in range(10):
            ids = self.allLabels == i
            ids = ids.nonzero()
            curClassEmbeddings = self.allEmbeddings[ids].squeeze()
            allClassEmbeddings[i] = curClassEmbeddings
        import pickle
        my_file = open("feature_vectors_mtl.pkl", "wb")
        pickle.dump(allClassEmbeddings, my_file)
        my_file.close()
        exit(0)

    def test_step(self, batch, batch_nb):
        embeddings, predictions, labels = self.execute_model(batch)
        triplet_loss, cls_loss, accuracy = self.calculate_loss_acc(embeddings, predictions, labels)
        #self.display_results(embeddings, labels)
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
