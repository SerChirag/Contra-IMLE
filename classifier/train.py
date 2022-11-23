import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module


def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        model_name = "combinedmodel_" + str(args.triplet_margin) + "margin_ctl_lr" + str(args.learning_rate)

        if args.logger == "wandb":
            logger = WandbLogger(name=model_name, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        checkpoint = ModelCheckpoint(filename=model_name + '{epoch}', monitor="acc/val", mode="max", save_last=False)

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            gpus=-1,
            deterministic=True,
            weights_summary=None,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            checkpoint_callback=checkpoint,
            precision=args.precision,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        if bool(args.pretrained_cp):
            # Set desired pt path here
            # state_dict = os.path.join(
            #   "cifar10", "fxv8rjdl", "epoch=4-step=974.ckpt"
            # )
            checkpoint_path = args.pretrained_data
            model.load_from_checkpoint(checkpoint_path)
            #model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            print("Testing the model ", args.classifier)
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pretrained_cp", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pretrained_data", type=str, default=os.path.join("cifar10_models",
                                                                            "state_dicts",
                                                                            "resnet18.pt"))
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--triplet_margin", type=float, default=0.05)
    parser.add_argument("--triplet_constant", type=float, default=1)

    args = parser.parse_args()
    main(args)
