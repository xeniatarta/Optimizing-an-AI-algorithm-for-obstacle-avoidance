import os
import yaml

# 1. Calculam calea absoluta catre folderul dataset-ului
current_dir = os.getcwd()
dataset_dir = os.path.join(current_dir, "datasets", "combined_dataset")
yaml_file = os.path.join(dataset_dir, "data.yaml")

# Verificam daca exista
if not os.path.exists(yaml_file):
    print(f"❌ Eroare: Nu gasesc fisierul {yaml_file}")
    exit()

print(f"🔧 Repar fisierul: {yaml_file}")
print(f"📂 Noua cale absoluta: {dataset_dir}")

# 2. Citim si modificam
with open(yaml_file, 'r') as f:
    content = yaml.safe_load(f)

# Punem calea absoluta (fixeaza problema cu C:\Windows\System32)
content['path'] = dataset_dir 
# Ne asiguram ca train/val sunt relative la 'path'
content['train'] = "train/images"
content['val'] = "valid/images"

# 3. Salvam inapoi
with open(yaml_file, 'w') as f:
    yaml.dump(content, f, default_flow_style=False)

print("✅ Gata! Calea a fost corectata. Acum poti rula train_unified.py")