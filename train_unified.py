from ultralytics import YOLO

if __name__ == '__main__':
    # Incarcam modelul de baza
    model = YOLO('yolov8n.pt')

    # Antrenam pe datasetul COMBINAT
    results = model.train(
        data='datasets/combined_dataset/data.yaml', 
        epochs=50,       # 50-100 e bine
        imgsz=640,       # Antrenam la 640 pentru calitate
        batch=16,
        device=0,
        name='yolo_unified', # Numele noului model
        half=True        # Economiseste memorie
    )

    print("✅ Model UNIFICAT salvat in runs/detect/yolo_unified/weights/best.pt")