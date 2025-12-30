from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Incarcam modelul
    model = YOLO('yolov8n.pt')

    # 2. Pornim antrenarea
    results = model.train(
        data='datasets/pothole_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        batch=16,
        name='yolo_potholes',
        workers=0  # <--- MODIFICAREA MAGICĂ (Rezolva eroarea pe Windows)
    )

    print("Antrenare completa!")