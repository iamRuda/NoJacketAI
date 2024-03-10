from imageai.Detection import ObjectDetection
import os
from tqdm.auto import tqdm
from PIL import Image

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

custom_objects = detector.CustomObjects(person=True)

for i in tqdm(range(len(list_images))):
    path_input_image = "input/" + list_images[i]
    path_output_image = "output/" + list_images[i]
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, path_input_image),
                                                 output_image_path=os.path.join(execution_path, path_output_image),
                                                 minimum_percentage_probability=30,
                                                 display_object_name=False,
                                                 display_percentage_probability=False,
                                                 custom_objects=custom_objects)

    # Создание папки для сохранения вырезанных областей
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Обработка и сохранение вырезанных областей
    for j, eachObject in enumerate(detections):
        box_points = eachObject["box_points"]
        region = (box_points[0], box_points[1], box_points[2], box_points[3])
        image = Image.open(path_input_image)
        cropped_image = image.crop(region)
        cropped_image.save(os.path.join(output_folder_path, f"{list_images[i][:-4]}_person_{j + 1}.jpg"))
    """
    for eachObject in detections:
        print(list_images[i], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
    """
