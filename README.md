# inpainting_cGAN
This repository is a Tensorflow implementation of "Conditional Attribute를 통한 Inpainting GAN 모델의 성능 개선"

## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- scipy 0.19.0
- opencv 3.2.0
- pyamg 3.3.2
- opencv 4.1.0

## Usage
Directory Hierarchy
```text
inpainting_cGAN
├── Data/
│   ├── celebA/
│   ├── SVHN/
│   └── VUB/
└── src/
    ├── dataset.py
    ├── dcgan.py
    ├── download.py
    ├── image_edit.py
    ├── inpaint_main.py
    ├── inpaint_model.py
    ├── inpaint_solver.py
    ├── main.py
    ├── mask_generator.py
    ├── poissonblending.py
    ├── solver.py
    ├── tensorflow_utils.py
    └── utils.py
```
### Download Dataset
You can use `download.py` to download datasets such as celebA and MNIST. You ***must*** put your dataset files under
 `Data/` or you can manually set the directory in the `dataset.py` file.

### Train GAN or cGAN
To train the model implemented in the `dcgan.py` file, run the next code.
```shell script
> python main.py --is_train=true --dataset=celebA 
```

### Train inpainting model

### Attributions
- This project borrowed some code from 
[semantic-image-inpainting](https://github.com/ChengBinJin/semantic-image-inpainting) and 
[DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow).
- Special thanks to Sungkyunkwan University.
