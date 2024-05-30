import os
import cv2
import time
import numpy as np
from flask import Response, Flask, render_template
from multiprocessing import Process, Manager
from imageai.Detection import ObjectDetection
from tensorflow import keras

# Инициализация Flask приложения
app = Flask(__name__)
source = "rtsp://admin:admin@192.168.0.99:554/0"

# Инициализация модели для детекции объектов
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("ai_model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

# Инициализация модели классификации
model = keras.models.load_model('NoJacket.model')
CATEGORIES = ['jacket', 'no_jacket']

detected_dir = 'detected'
detected_dir2 = 'check'
temp_dir = 'temp'

def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_gray, (256, 256))
    reshaped_img = resized_img.reshape(-1, 256, 256, 1)
    return reshaped_img


def cache_frames(source: str, last_frame: list, running) -> None:
    """ Кэширование кадров """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0  # Счетчик кадров
    temp_image_path = "temp_frame.jpg"

    while running.value:
        ret, frame = cap.read()  # Чтение кадра
        if ret:  # Если кадр считан
            frame_count += 1
            if frame_count % 15 == 0:  # Обрабатывать каждый 10-й кадр
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_colored = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(temp_image_path, frame_colored)

                # Детекция объектов на изображении
                detections = detector.detectObjectsFromImage(input_image=temp_image_path,
                                                             output_image_path=temp_image_path,
                                                             minimum_percentage_probability=30,
                                                             display_object_name=False,
                                                             display_percentage_probability=False,
                                                             custom_objects=detector.CustomObjects(person=True))

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
                        cv2.rectangle(frame, (box_points[0], box_points[1]), (box_points[2], box_points[3]),
                                      (0, 255, 0), 2)

                        # Если обнаружена верхняя одежда, сохраняем изображение целиком
                        if predicted_class == 'jacket':
                            output_path = os.path.join(detected_dir,
                                                       f"detected_person_with_jacket_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                            cv2.imwrite(output_path, frame)
                            print(f"Изображение сохранено: {output_path}")
                        elif predicted_class == 'no_jacket':
                            output_path = os.path.join(detected_dir2,
                                                       f"detected_person_with_nojacket_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                            cv2.imwrite(output_path, frame)
                            print(f"Изображение сохранено: {output_path}")

                else:
                    pass

            frame = cv2.resize(frame, (1280, 900))  # Изменение размера кадра
            _, buffer = cv2.imencode('.png', frame)  # Кодирование кадра в PNG
            last_frame[0] = buffer.tobytes()  # Кэширование кадра
        else:
            break
        time.sleep(1 / (fps + 1))  # Интервал между кадрами
    cap.release()


def generate(shared_last_frame: list):
    """ Генератор кадров """
    frame_data = None
    while True:
        if frame_data != shared_last_frame[0]:  # Если кадр изменился
            frame_data = shared_last_frame[0]
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame_data + b'\r\n')  # HTTP ответ для потоковой передачи
        time.sleep(1 / 15)  # Задержка


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(generate(last_frame),
                    mimetype="multipart/x-mixed-replace; boundary=frame")  # Запуск генератора


if __name__ == '__main__':
    with Manager() as manager:
        last_frame = manager.list([None])  # Кэш последнего кадра
        running = manager.Value('i', 1)  # Управляемый флаг для контроля выполнения процесса

        # Создаём процесс для кэширования кадров
        p = Process(target=cache_frames, args=(source, last_frame, running))
        p.start()

        # Запуск Flask-приложения в блоке try/except
        try:
            app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            p.join()  # Ожидаем завершения процесса
        finally:
            running.value = 0  # Устанавливаем флаг в 0, сигнализируя процессу о необходимости завершения

        p.terminate()  # Принудительно завершаем процесс, если он все еще выполняется
        p.join()  # Убедимся, что процесс
