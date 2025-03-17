from config import RAW_DATA_DIR
from data_load_test import setup_fashionpedia, load_fashionpedia_categories, create_data_loaders, visualize_segmentation
from pathlib import Path


def main():
    data_dir = Path(RAW_DATA_DIR)

    # Setup paths
    data_paths = setup_fashionpedia(data_dir)

    # Load category mappings
    category_mappings = load_fashionpedia_categories(data_paths['train_ann_file'])
    print(f"Total number of classes: {category_mappings['num_classes']}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_paths, category_mappings, batch_size=4)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Visualize a sample
    for images, masks in train_loader:
        # Display the first image and mask in batch
        visualize_segmentation(images[0], masks[0], category_mappings['id_to_name'])
        break

if __name__ == "__main__":
    main()
