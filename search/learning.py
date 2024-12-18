from ultralytics import YOLO

# Указываем путь к конфигурационному файлу данных
DATA_CONFIG = r'C:\Users\vanoh\OneDrive\Desktop\123\data.yaml'

# Создаём модель (можно использовать предобученную модель)
model = YOLO('yolov8s.pt')  # Выбираем версию модели: n (nano), s (small), m (medium), l (large)

# Обучаем модель
model.train(
    data=DATA_CONFIG,       # Путь к файлу data.yaml
    epochs=15,              # Количество эпох обучения
   
    batch=8,                # Размер батча
    name='yolo8_drone'      # Название эксперимента
)
