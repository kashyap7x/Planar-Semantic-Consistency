# Planar-Semantic-Consistency

PyTorch implementation of self-supervised training technique using novel view synthesis as an auxiliary training signal for semantic segmentation networks.

* `lib` contains an implementation of synchronized batch norm for PyTorch
* `monodepth` contains TensorFlow code for genrating approximate dense disparity maps used during training
* `pretrained` contains ResNet-18 models trained on ImageNet and GTA5
* `train.py` is the script to be executed 
