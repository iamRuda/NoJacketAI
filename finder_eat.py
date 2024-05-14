from imageai.Detection import ObjectDetection
import os

from sympy import true
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

execution_path = os.getcwd()


# Инициализация модели
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("ai_model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

# Загрузка изображения
path_input_images = "input/"
list_images = [f for f in os.listdir(path_input_images) if os.path.isfile(os.path.join(path_input_images, f))]
output_folder_path = "output/list"

custom_objects = detector.CustomObjects(wine_glass=true, bowl=true, cup=true, bottle=true, banana=true, apple=true, sandwich=true,
                                        orange=true, broccoli=true, carrot=true, pizza=true, cake=true, donut=true,
                                        )

for i in tqdm(range(len(list_images))):
    path_input_image = "input/" + list_images[i]
    path_output_image = "output/" + list_images[i]
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, path_input_image),
                                                 output_image_path=os.path.join(execution_path, path_output_image),
                                                 minimum_percentage_probability=30,
                                                 display_object_name=False,
                                                 display_percentage_probability=False,
                                                 )

    # Создание папки для сохранения вырезанных областей
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Обработка и сохранение вырезанных областей
    for j, eachObject in enumerate(detections):
        print(eachObject)
        box_points = eachObject["box_points"]
        region = (box_points[0], box_points[1], box_points[2], box_points[3])
        image = Image.open(path_input_image)
        image_array = np.array(image)
        plt.imshow(image_array)
        x1, y1, x2, y2 = region
        plt.plot([x1, x2], [y1, y1], color='red', linewidth=2)
        plt.plot([x1, x2], [y2, y2], color='red', linewidth=2)
        plt.plot([x1, x1], [y1, y2], color='red', linewidth=2)
        plt.plot([x2, x2], [y1, y2], color='red', linewidth=2)
        plt.axis('off')
        plt.show()
