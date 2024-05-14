from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "ai_model/yolov3.pt"))
detector.loadModel()

# Загрузка изображения
path_input_images = "input/"
list_images = [f for f in os.listdir(path_input_images) if os.path.isfile(os.path.join(path_input_images, f))]
output_folder_path = "output/list"

print(os.path.join(execution_path, path_input_images, list_images[0]))
path_input_image = "input/" + list_images[3]
detections = detector.detectObjectsFromImage(input_image=path_input_image, output_image_path=os.path.join(execution_path, "imagenew.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")