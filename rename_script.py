import os

folder_path = "input/"
count = 0
# Перебираем все файлы в папке
for filename in os.listdir(folder_path):
    count += 1
    old_file_path = os.path.join(folder_path, filename)

    # Проверяем, является ли путь файлом
    if os.path.isfile(old_file_path):
        # Генерируем новое имя файла (можно использовать любую логику)
        new_filename = str(count)
        filename[:4]
        new_file_path = os.path.join(folder_path, new_filename + ".jpg")

        # Переименовываем файл
        os.rename(old_file_path, new_file_path)
        print(f"Файл {filename} переименован в {new_filename}")
