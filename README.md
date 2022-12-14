
VGG13 BN Feature Vector Weights: https://1sfu-my.sharepoint.com/:u:/g/personal/cva19_sfu_ca/EUqqwCBqdxFHia2Xm1UXLugBE9ABm0roXE4ToGHFaDS3kw?e=BDehHH

VVG13 BN Triplet Loss Model Feature Vector Weights: https://1sfu-my.sharepoint.com/:u:/g/personal/las18_sfu_ca/EaDvLQax57FJjUwMU4l7muQBHOT1yr9eDcXRZQQRnyz5Dw?e=XPiDke

For Classifier
- Remember to login to Wandb, it will also work though without a login, but you miss out on the graphs

#Download pretrained weights
python train.py --download_weights 1 

#Test pretrained models
python train.py --test_phase 1 --pretrained 1 --classifier vgg13_bn

#Train model
python train.py --classifier vgg13_bn 

