import os
import shutil
import yaml

# --- CONFIGURARE ---
# Lista dataseturilor si noul ID pe care il vor avea
# Format: ("cale_folder_vechi", ID_NOU, "nume_clasa")
DATASETS = [
    ("datasets/pothole_dataset", 0, "pothole"),
    ("datasets/obstacles1_dataset", 1, "obstacle"),
    ("datasets/animals_dataset", 2, "animal"),
    # Daca vrei si masini, trebuie sa ai un dataset cu masini aici
    # ("datasets/cars_dataset", 3, "car") 
]

OUTPUT_DIR = "datasets/combined_dataset"

def merge_data():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Cream structura noua
    for split in ['train', 'valid']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    print(f">>> Incep combinarea in '{OUTPUT_DIR}'...")

    for dataset_path, new_class_id, class_name in DATASETS:
        print(f" Procesez: {class_name} (ID Nou: {new_class_id})...")
        
        for split in ['train', 'valid']:
            src_img_dir = os.path.join(dataset_path, split, "images")
            src_lbl_dir = os.path.join(dataset_path, split, "labels")
            
            if not os.path.exists(src_img_dir): continue

            # Copiem si modificam
            files = os.listdir(src_img_dir)
            for f in files:
                if not (f.endswith('.jpg') or f.endswith('.png')): continue
                
                # 1. Copiem Imaginea
                shutil.copy(os.path.join(src_img_dir, f), 
                            os.path.join(OUTPUT_DIR, split, "images", f"{class_name}_{f}"))
                
                # 2. Modificam si Copiem Label-ul
                label_name = f.rsplit('.', 1)[0] + ".txt"
                src_label = os.path.join(src_lbl_dir, label_name)
                dst_label = os.path.join(OUTPUT_DIR, split, "labels", f"{class_name}_{label_name}")
                
                if os.path.exists(src_label):
                    with open(src_label, 'r') as file:
                        lines = file.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        # Inlocuim ID-ul vechi (primul numar) cu cel NOU
                        # Atentie: Presupunem ca datasetul vechi are o singura clasa (0)
                        parts[0] = str(new_class_id) 
                        new_lines.append(" ".join(parts) + "\n")
                    
                    with open(dst_label, 'w') as file:
                        file.writelines(new_lines)

    # Generam data.yaml
    yaml_content = f"""
path: ../{OUTPUT_DIR}
train: train/images
val: valid/images

nc: {len(DATASETS)}
names: {[d[2] for d in DATASETS]}
    """
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print("✅ Gata! Dataset combinat creat.")

if __name__ == "__main__":
    merge_data()