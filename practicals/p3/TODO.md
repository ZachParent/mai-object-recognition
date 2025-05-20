## TODO
- [x] dataset @pedro
- [x] preprocessing + image generation @pedro
- [x] runner @zach
  - [x] get base U-Net2D replica in torch with standard hyperparameters
- [x] baseline model @bruno
  - [ ] Hyperparameter tuning
  - [ ] Data Augmentation
- [x] perceptual loss @bruno
  - [x] Normal Maps
  - [x] L1 or L2 discrepancy error
- [x] BSC running compatibility @zach
- [x] visualizations @zach
- [ ] include smpl in the input @zach
- [ ] vision transformers x 3 @ sheena
- [ ] analyze results @all
- [ ] report / presentation @all

## notes 2025-05-13
- subject might go out of the scene, ignore these frames
  - count number of pixels in the mask and thresholding => if the # pixels with a person is < 10% of the image, ignore the frame
  - this is the simple way
  - the purpose of the task is to play with smpl
    - we use pose information for one task
    - we can use it also to check if the person is out of the scene

- try no more than 3 options for the vision transformer
  - original unet + 3 more options

## Assignment info
### Deliverable 3 - Body and cloth depth estimation
- [ ] Given an image of a person, cropped in the preprocessing (red rectangle), the goal is to estimate its corresponding depth image.

### Dataset
- [x] A subset of the CLOTH3D++ dataset (https://chalearnlap.cvc.uab.cat/dataset/38/description/), could be downloaded from [cloth3d++_subset.zip](https://cvcuab-my.sharepoint.com/:u:/g/personal/mmadadi_cvc_uab_cat/EaJUHQv5N2dEjvA51WbGLdIB5aVjZfQraF0Fa0tprVMBYA?e=rJv9sZ).
    - The first 128 folders (00001 to 00152) must be used for training.
    - The following 16 folders (00153 to 00168) for validation.
    - The rest for test.
- [x] You can play with the data, see data structure, visualize the 3D, render depth images and extract frames from the RGB videos using the given starter kit.
- [x] If you use Colab, the content of the starter kit must be uploaded to your Google Drive under "cloth3d" folder.

### Preprocessing
- [x] Crop and save the images such that:
    - 1 The center of the subject and cropping to be the same.
    - 2 Leave 10px margin between cropping and subject boundaries.
    - 3 Apply square cropping.
    - NOTE: in some frames the subject may go out of the scene. You can ignore these frames, for instance by counting the number of pixels in the mask and thresholding.
- [ ] The frames must be saved as `.jpg` under "image" folder with naming protocol "<folder name where the video is located>_<frame number>".
- [ ] Rendered depth must be cropped similar to RGB frame and saved as `.npy` (using `numpy.save`) under "depth" folder with the same naming protocol as RGB frames.
- [ ] The cropped images can be resized to 256x256 before saving them.
    - Note that resizing the depth images could be tricky especially for the boundaries. You can use nearest neighbor values to avoid the effect of the zero background.

### Tasks
- A simple baseline code is given for depth estimation using UNET architecture.
- Perform the following tasks:
    - [x] Preprocess the data.
    - [x] Tune some of the hyperparameters of your choice and apply data augmentation relevant to the problem.
    - [ ] Study the vision transformer architectures by replacing the UNET backbone. Try no more than 3 options.
    - [ ] Develop a perceptual loss, along with the main loss, by computing the normal map from the estimated and ground truth depth images and minimize their discrepancy error by L1, L2 or any relevant loss. Apply the loss on your best performing network so far and compare the results.
    - [ ] By the help of the starter kit, create a color-coded SMPL pose image, similar to the next image, concatenate it with the RGB image and use it to train/test your best network. Can this complementary information help the network to improve the results? Note that we will use ground truth pose for test images as well just to study the impact of this extra information.
- [ ] Thoroughly discuss the results both quantitatively and qualitatively.

### Tips and modifications required on the code
- For a faster I/O operation, you may:
    - 1 Train with multiple workers, e.g. `model.fit(......., workers=4)`.
    - 2 (Optional) Save the whole data in a `tfrecord` file and iterate over it, an example here: https://keras.io/examples/keras_recipes/tfrecord/
