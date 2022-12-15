
VGG13 BN Feature Vector Weights: https://1sfu-my.sharepoint.com/:u:/g/personal/cva19_sfu_ca/EUqqwCBqdxFHia2Xm1UXLugBE9ABm0roXE4ToGHFaDS3kw?e=BDehHH

VVG13 BN Triplet Loss Model Feature Vector Weights: https://1sfu-my.sharepoint.com/:u:/g/personal/las18_sfu_ca/EaDvLQax57FJjUwMU4l7muQBHOT1yr9eDcXRZQQRnyz5Dw?e=XPiDke

MixMatch Feature Vector Weights: https://drive.google.com/file/d/1iXZriuRufqHB8j2ZiRSMrd_DA69P-wI1/view?usp=share_link
IMLE using mixmatch feature vector checkpoint at 500 epoch: https://drive.google.com/file/d/1_Jn_pCOKiOrapp-3ajUGpWRhaS1npurW/view?usp=sharing

**Triplet Loss Classifier Model**
- Remember to login to Wandb, it will also work though without a login, but you miss out on the graphs

#Download pretrained weights
python train.py --download_weights 1 

#Test pretrained models
python train.py --test_phase 1 --pretrained_cp 1 --pretrained_data /path/to/data --classifier vgg13_bn

#Train model
python train.py --classifier vgg13_bn
#To Save the feature weights
Uncomment the lines 44, 45, 109, 110, 124, 125 and run
python train.py --classifier vgg13_bn --pretrained_cp 1 --pretrained_data /path/to/data

**IMLE Implementation**
imle_deep.py is the training file to train IMLE on feature vectors. Checkpoints can be added as flags to resume training.
Feature vector files must be downloaded first and replaced in the 'load_features' function accordingly. Current code loads autoencoder feature vectors.
Note: Mixmatch feature vectors are 128 dims in size so zdim must be changed to (128,1,1) instead of (256,1,1) if using mixmatch feature vectors

**Mixmatch Implementation**
README to train mixmatch from scratch can be found under the mixmatch folder. The code is currently set up to train on the CIFAR-10 dataset. To extract features from the trained model requires uncommenting and commenting of specific code in the following files: model/wideresnet.py, train.py, dataset/cifar10.py. Instructions for lines to comment/uncomment are provided in each of these files

**REFERENCES**
Triplet loss model build off of classifiers from: https://github.com/huyvnphan/PyTorch_CIFAR10
Mixmatch implementation is borrowed from: https://github.com/YU1ut/MixMatch-pytorch
