import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_segmentation(image, mask, id_to_name, show=True):
    """Utility function to visualize segmentation masks"""

    # Create a colormap for visualization
    cmap = plt.cm.get_cmap("tab20", len(id_to_name))

    plt.figure(figsize=(16, 8))

    # Display original image
    plt.subplot(1, 2, 1)
    if isinstance(image, torch.Tensor):
        # Denormalize and convert to numpy for visualization
        img = image.permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
    else:
        img = image
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # Display segmentation mask
    plt.subplot(1, 2, 2)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3))
    for class_id in np.unique(mask):
        if class_id == 0:  # Background
            continue

        # Create color mask
        class_mask = mask == class_id
        color = cmap(class_id)[:3]  # RGB
        for i in range(3):
            colored_mask[class_mask, i] = color[i]

    plt.imshow(colored_mask)

    # Create a legend
    handles = []
    for class_id in sorted(np.unique(mask)):
        if class_id in id_to_name:
            color = cmap(class_id)[:3]
            patch = plt.Rectangle((0, 0), 1, 1, fc=color)
            handles.append((patch, id_to_name[class_id]))

    # Only show legend if there are classes to display
    if handles:
        patches, labels = zip(*handles)
        plt.legend(patches, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.tight_layout()
    if show:
        plt.show()

    return plt.gcf()  # Return the figure for saving if needed
