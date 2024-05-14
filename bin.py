import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# Загрузка модели и категорий
model = keras.models.load_model('NoJacket.model')
CATEGORIES = ['jacket', 'no_jacket']  # Предположим, что у вас всего два класса


# Функция для обработки изображения
def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_gray, (256, 256))
    reshaped_img = resized_img.reshape(-1, 256, 256, 1)
    return reshaped_img


# Открытие видеопотока с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры.")
        break

    # Обработка изображения и получение предсказания
    processed_img = process_image(frame)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CATEGORIES[predicted_class_index]

    # Вывод результата на кадр
    cv2.putText(frame, predicted_class, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Прерывание цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
