import os
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image


class ObjectLabel:
    def __init__(self, name, bounding_box):
        self.name = name
        self.bounding_box = bounding_box
        self.area = self.calculate_area()

    def calculate_area(self):
        xmin, ymin, xmax, ymax = self.bounding_box
        return (xmax - xmin) * (ymax - ymin)


def load_names_list(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def read_content(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    object_labels = []
    for obj in root.iter("object"):
        class_name = obj.find("name").text
        ymin = int(round(float(obj.find("bndbox/ymin").text)))
        xmin = int(round(float(obj.find("bndbox/xmin").text)))
        ymax = int(round(float(obj.find("bndbox/ymax").text)))
        xmax = int(round(float(obj.find("bndbox/xmax").text)))

        bounding_box = [xmin, ymin, xmax, ymax]
        object_labels.append(ObjectLabel(class_name, bounding_box))

    return object_labels


def process_all_data(data_list, root):
    records = []

    for file_name in data_list:
        img_path = os.path.join(root, "JPEGImages", f"{file_name}.jpg")
        xml_path = os.path.join(root, "Annotations", f"{file_name}.xml")

        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            continue

        object_labels = read_content(xml_path)
        img = Image.open(img_path)
        img_width, img_height = img.size

        records.append(
            {
                "file_path": file_name,
                "object_labels": [
                    {
                        "name": obj.name,
                        "bounding_box": obj.bounding_box,
                        "area": obj.area,
                    }
                    for obj in object_labels
                ],
                "image_area": img_width * img_height,
            }
        )

    return pd.DataFrame(records)


data_directory_path = "mai-object-recognition/practicals/p1/data"
voc_root = os.path.join(data_directory_path, "VOC2012")
train_list = load_names_list(os.path.join(data_directory_path, "train.txt"))
test_list = load_names_list(os.path.join(data_directory_path, "test.txt"))

test_df = process_all_data(test_list, voc_root)
train_df = process_all_data(train_list, voc_root)

test_df.to_json(
    os.path.join(data_directory_path, "preprocessed_jsons/test.json"),
    orient="records",
    indent=4,
)
train_df.to_json(
    os.path.join(data_directory_path, "preprocessed_jsons/train.json"),
    orient="records",
    indent=4,
)
