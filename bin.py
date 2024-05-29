import cv2
import time
import numpy as np
from imageai.Detection import ObjectDetection
from tensorflow import keras

# Инициализация модели для детекции объектов
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("ai_model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

# Инициализация модели классификации
model = keras.models.load_model('NoJacket.model')
CATEGORIES = ['jacket', 'no_jacket']

# Настройка видеопотока с камеры
cap = cv2.VideoCapture(0)
custom_objects = detector.CustomObjects(person=True)

def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_gray, (256, 256))
    reshaped_img = resized_img.reshape(-1, 256, 256, 1)
    return reshaped_img

# Переменная для отслеживания времени последнего обработанного кадра
last_processed_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры.")
        break

    # Проверка, прошло ли уже 5 секунд с момента обработки последнего кадра
    current_time = time.time()
    if current_time - last_processed_time >= 5:
        # Сохранение текущего кадра в виде изображения
        frame_image_path = "temp_frame.jpg"
        cv2.imwrite(frame_image_path, frame)

        # Детекция объектов на изображении
        detections = detector.detectObjectsFromImage(input_image=frame_image_path,
                                                     output_image_path="temp_output.jpg",
                                                     minimum_percentage_probability=30,
                                                     display_object_name=False,
                                                     display_percentage_probability=False,
                                                     custom_objects=custom_objects)
        if detections:
            for eachObject in detections:
                box_points = eachObject["box_points"]
                region = (box_points[0], box_points[1], box_points[2], box_points[3])
                cropped_image = frame[box_points[1]:box_points[3], box_points[0]:box_points[2]]

                # Предсказание на вырезанном изображении
                processed_img = process_image(cropped_image)
                prediction = model.predict(processed_img)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CATEGORIES[predicted_class_index]

                # Вывод результата
                print(f"Prediction: {predicted_class}")
                cv2.putText(frame, predicted_class, (box_points[0], box_points[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (box_points[0], box_points[1]), (box_points[2], box_points[3]), (0, 255, 0), 2)

                # Если обнаружена верхняя одежда, сохраняем изображение целиком
                if predicted_class == 'jacket':
                    cv2.imwrite(f"detected_person_with_jacket_{time.strftime('%Y%m%d%H%M%S')}.jpg", cropped_image)

        # Обновление времени последнего обработанного кадра
        last_processed_time = current_time

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Прерывание цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
