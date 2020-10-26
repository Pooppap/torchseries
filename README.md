# torchseries

torchseries is a repertoire consist of augmentation techniques commonly used with time-series data, (especially IMU data), implemented primarily in PyTorch environment.
Most augmentations implemented in this package took inspiration from T. T. Um et al., [“Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,”](https://arxiv.org/abs/1706.00527). 
The interp1d function is a stripped-down modification from this [interp1d repo](https://github.com/aliutkus/torchinterp1d.git)

## Requirements

|Packages |Version   |
|---------|----------|
|`PyTorch`|`>=1.6.0` |
|`Numpy`  |`>=1.18.0`|
|`Scipy`  |`>=1.4.0` |
