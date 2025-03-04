
import cv2
import xml.etree.ElementTree as ET
from config import voc_classes, num_classes
import numpy as np

def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_objects = []
    for boxes in root.iter('object'):

        classname = boxes.find("name").text
        list_with_all_objects.append(voc_classes[classname])

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_objects, list_with_all_boxes

def load_batch(data_list, step, batch_size, raw_data_dir, img_size):
  X, Y = [], []
  for f in data_list[step*batch_size : (step+1)*batch_size]:
    img = cv2.imread(raw_data_dir / 'JPEGImages' / (f + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, (img_size, img_size))
    X.append(img)

    classes = np.zeros(num_classes)
    try:
      cnames, _ = read_content(raw_data_dir / 'Annotations' / (f + '.xml'))
    except:
       print(f)
    for c in cnames:
        classes[c] = 1.0
    Y.append(classes)

  return (np.array(X), np.array(Y))