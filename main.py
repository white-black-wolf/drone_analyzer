from ultralytics import YOLO

import math
# Загрузка обученной модели
model = YOLO("best.pt")

# Запуск предсказания на изображении
results = model.predict(
    source=r"drone.webp", 
    save=True, 
    show=True, 
    conf=0.15,  
    iou=0.5     
)

# Показать результаты
for result in results:
    result.show()

    boxes = results[0].boxes.xyxy  # Координаты bounding box в формате [x1, y1, x2, y2]
    confidence = results[0].boxes.conf  # Уверенность предсказания
    classes = results[0].boxes.cls  # Классы объектов

    # Фильтрация только объектов класса "drone" (класс для дронов, например, 0)
    for i in range(len(classes)):
        if classes[i] == 0:  # Индекс для класса "drone"
            x1, y1, x2, y2 = boxes[i]
            
            print(f"Confidence: {confidence[i]:.2f}")


