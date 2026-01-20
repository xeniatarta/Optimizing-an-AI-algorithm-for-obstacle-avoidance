from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Incarcam modelul TAU antrenat anterior (best.pt)
    # ATENTIE: Verifica daca calea este corecta. 
    # Daca vrei sa folosesti modelul de la gropi ca baza, lasa calea asta.
    # Daca ai mutat best.pt, pune noua cale aici.
    path_to_best_model = 'runs/detect/yolo_potholes3/weights/best.pt'
    
    print(f"Incarc din: {path_to_best_model}")
    model = YOLO(path_to_best_model)

    # 2. Pornim antrenarea pe noul dataset (ANIMALS)
    # YOLO va detecta automat ca numarul de clase e diferit (Gropi vs Animale)
    # si va reseta ultimul strat (Head-ul) pastrand cunostintele de baza.
    results = model.train(
        data='datasets/animals_dataset/data.yaml',  
        epochs=50,       # Poti creste la 100 daca ai timp
        imgsz=640,
        device=0,        
        batch=16,
        name='yolo_animals',  # Nume nou pentru a nu suprascrie folderul potholes
        workers=2        # Pe Windows uneori 0 sau 2 e mai stabil decat default
    )

    print("Antrenare completa! Noul model este salvat in runs/detect/yolo_animals/weights/best.pt")