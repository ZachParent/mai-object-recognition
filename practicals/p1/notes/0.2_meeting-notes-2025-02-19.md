# Meeting Project 1: CNN February 19, 2025

- Nested directories → Bring data into better structure
    - Load as pandas dataframe and save as csv → Rows images, cols 0/1 has airplane or not
- Analyze data
    - Count objects
    - Count images
    - Mean / variance of the area ratio of objects per label → Use bounding box to compute the area
        - Annotations (XML files) convert this into a better structure
        - Confusing that there are less tags than images
        - Image, class, left, right, top bottom; instead of dividing by image divide by bounding boxes (no naming convention needed, just index)

### Tasks

- Preprocessing → **Pedro**
- Analyze the dataset → **Zach**
- The runner/ test framework (networks 1, 2 and 3) → **Sheena**
    - Warm-up

— Decide on best performing network and continue with this **(Sheena)**

- Data augmentation algorithms (2) → **Mattin**
- Data balancing algorithms (2) → **Zach**
- Redesign the classifier head → **Bruno**
- Data visualization → **Zach**
- (Report)
- Write email → Zach (Today)

Be done with coding part **02.03**

- Figuring out what the hell we are doing
- Main structure
- Experimentation
- Report

### RQs

- Training
    - Pretrained weights with warmup
    - Pretrained weights without warmup
    - Pretrained from scratch
- Hyperparams
    - Batch size [16, 32, 64, (128)]
    - Learning rate
    - Optimizer
    - …
- Which performance measure?
    - We are looking for whatever the fuck multilabel

### Questions

- Are we missing the train val
- Are we allowed to use Keras implementations
- Can we start before the corresponding Q&A sessions (do we have all information/data)