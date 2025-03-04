# Task-specific parameters

img_size = 224
num_classes = 20
voc_classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
test_imagenet = True
root = './VOCdevkit/VOC2012/'
per_sample_normalization = True
data_augmentation = True

# Experiment-specific parameters were moved to experiment_config.py

# batch_size = 32
# n_epochs = 1
# net_name = [['resnet50','ResNet50'], ['inception_v3','InceptionV3'], ['mobilenet_v2','MobileNetV2']][0]
# train_from_scratch = True
# last_layer_activation = ['softmax', 'sigmoid', None][1]
# loss = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'][1]

