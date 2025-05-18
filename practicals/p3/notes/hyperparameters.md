# Baseline Model Hyperparameter Research

## Standard Baseline Model configuration

```
model = models.unet_2d(input_size=(480, 640, 3), filter_num=[64, 128, 256, 512, 1024],
                      n_labels=1, stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Sigmoid',
                      batch_norm=True, pool='max', unpool=False, name='unet')
```

## Hyperparameters
- `filter_num`: Number of elements in the list determines the number of levels of downsampling (and upsampling will be the same), and its values are the number of filters (channels) in each convolutional layer inside the corresponding level.
- `stack_num_down`: Number of convolutional layers per downsampling level.
- `stack_num_up`: Number of convolutional layers per upsampling level.
- `activation`: Activation function after each convolutional layer (both up and down).
- `output_activation`: Activation function after the final convolutional layer.
- `batch_norm`: Whether to apply batch normalization after each convolutional layer (before activation) or not.
- `pool`: Type of pooling used in each downsampling step.
- `unpool`: Type of unpooling used in each upsambling step. If `False`, `Conv2DTranspose + batch_norm + activation` is performed.


## Additional Notes
- *Input Size* is determined by the data. We could test different resolutions, but it is not the main focus of the project (and probably not worth it).
- `num_labels` is the number of output channels. For our case, it will be 1; but if we wanted to predict, for instance, RGB, it would be 3.
- *MaxPooling* used for each step of the downsampling is always 2x2 (halves both width and height of images). This is standard and not presented as a parameter of the model definition.

## Conclusions
- It is probably interesting to study the effect of varying `filter_num`. Maybe not its values, but definitely its length.
- For instance, we could test a reduced version with `[64, 256, 1024]` &rarr; I think this will have approximately the same number of parameters, but we should check.
- I do not think it is worth it to test `stack_num_down`, `stack_num_up`, `activation`, nor `output_activation`.
- It could also be insightful to test the influence of `batch_norm`.
- Maybe `pool` and/or `unpool` could be tested, depending on resource constraints. TBD when we have an estimate for training times.
- We have to decide if we prefer to perform grid search on little amount of hyperparameters or sequential search on a bigger amount of them.
  - *P.S.* Data Augmentation will also be tested, which should be considered when deciding the amount of experiments we want to do.
