import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from tensorflow import keras
import os
import time

# Инициализация модели для детекции объектов
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("ai_model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

# Инициализация модели классификации
model = keras.models.load_model('NoJacket.model')
CATEGORIES = ['jacket', 'no_jacket']

# Путь к директории с изображениями
image_dir = 'test'
detected_dir = 'detected'
temp_dir = 'temp'

# Создание директорий для сохранения результатов и временных файлов
os.makedirs(detected_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_gray, (256, 256))
    reshaped_img = resized_img.reshape(-1, 256, 256, 1)
    return reshaped_img

# Получение списка файлов в директории
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Обработка всех изображений в директории
if not image_files:
    print(f"Нет изображений в директории: {image_dir}")
else:
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        # Детекция объектов на изображении
        frame_image_path = os.path.join(temp_dir, "temp_frame.jpg")
        cv2.imwrite(frame_image_path, frame)
        detections = detector.detectObjectsFromImage(input_image=frame_image_path,
                                                     output_image_path=os.path.join(temp_dir, "temp_output.jpg"),
                                                     minimum_percentage_probability=30,
                                                     display_object_name=False,
                                                     display_percentage_probability=False,
                                                     custom_objects=detector.CustomObjects(person=True))
        if detections:
            for i, eachObject in enumerate(detections):
                box_points = eachObject["box_points"]
                cropped_image = frame[box_points[1]:box_points[3], box_points[0]:box_points[2]]

                # Предсказание на вырезанном изображении
                processed_img = process_image(cropped_image)
                prediction = model.predict(processed_img)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CATEGORIES[predicted_class_index]

                if predicted_class == 'jacket':
                    # Создание копии исходного изображения для каждого найденного "jacket"
                    temp_frame = frame.copy()

                    # Добавление метки и рамки только для "jacket"
                    cv2.putText(temp_frame, predicted_class, (box_points[0], box_points[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(temp_frame, (box_points[0], box_points[1]), (box_points[2], box_points[3]),
                                  (0, 255, 0), 2)

                    # Сохранение изображения с выделением
                    output_path = os.path.join(detected_dir,
                                               f"detected_person_with_jacket_{i}_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                    cv2.imwrite(output_path, temp_frame)
                    print(f"Изображение сохранено: {output_path}")

        else:
            print("Объекты не найдены на изображении:", image_path)
