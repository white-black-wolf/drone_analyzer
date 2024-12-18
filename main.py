from ultralytics import YOLO

import math
# Загрузка обученной модели
model = YOLO("C:/Users/vanoh/OneDrive/Desktop/123/runs/detect/yolo8_drone2/weights/best.pt")

# Запуск предсказания на изображении
results = model.predict(
    source=r"C:\Users\vanoh\OneDrive\Desktop\123\drone5.jpeg", 
    save=True, 
    show=True, 
    conf=0.1,  
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



        focal_length = 1000  # фокусное расстояние в пикселях (примерное значение, можно измерить или вычислить)
        real_drone_width = 250  # реальная ширина дрона в миллиметрах (например, для DJI Phantom 4 - 250 мм)

        def calculate_distance(focal_length, real_width, image_width):
           
            distance = (focal_length * real_width) / image_width  # Расстояние в миллиметрах
            distance_meters = distance / 1000  # Конвертируем в метры
            return distance_meters
        object_width_on_image = x2 - x1
        distance_to_drone = calculate_distance(focal_length, real_drone_width, object_width_on_image)
        print(f"Расстояние до дрона: {distance_to_drone:.2f} метров")