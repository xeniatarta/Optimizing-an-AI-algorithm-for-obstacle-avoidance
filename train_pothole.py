from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Incarcam un model de baza (pre-antrenat, dar "gol" de cunostinte specifice)
    model = YOLO('yolov8n.pt')

    # 2. Pornim antrenarea
    # data: calea catre fisierul data.yaml din folderul descarcat
    # epochs: de cate ori sa treaca prin toate pozele (50 e un start bun)
    # imgsz: marimea imaginii (640 e standard)
    # device: 0 inseamna placa ta video NVIDIA
    results = model.train(
        data='datasets/pothole_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        batch=16,
        name='yolo_potholes',
        workers = 0
    )

    print("Antrenare completa! Modelul este salvat in runs/detect/yolo_potholes/weights/best.pt")