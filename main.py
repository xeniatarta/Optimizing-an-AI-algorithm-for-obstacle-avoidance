import torch
from torch.utils.cpp_extension import load
from ultralytics import YOLO
import cv2
import glob
import os
import time
import numpy as np

# --- CONFIGURARE ---
INPUT_FOLDER = "images_in"
OUTPUT_FOLDER = "images_out"
# Asigura-te ca aceasta cale e corecta (fara extensie daca vrei auto-detectie)
POTHOLE_MODEL_PATH = "runs/detect/yolo_potholes3/weights/best" 

VISUALIZE = True
SAVE_OUTPUT = True

# Controler PD
KP = 1.0
KD = 0.6
last_raw_steering = 0.0

# 1. Verificam structura de foldere
if not os.path.exists(INPUT_FOLDER):
    print(f" EROARE: Folderul '{INPUT_FOLDER}' nu exista! Creeaza-l langa main.py.")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"CUDA Disponibil: {torch.cuda.is_available()}")

# --- 2. COMPILARE CUDA (Cu Timer) ---
print(">>> Compilez algoritmul CUDA...")
t_compile_start = time.time()
try:
    avoidance_cuda = load(
        name="avoidance_cuda",
        sources=["csrc/avoidance.cpp", "csrc/avoidance_kernel.cu"],
        extra_cflags=['/std:c++17', '/O2'],
        extra_cuda_cflags=['-allow-unsupported-compiler', '-O3', '--use_fast_math'],
        verbose=False
    )
    t_compile_end = time.time()
    print(f"✅ Compilare reusita! (Timp: {t_compile_end - t_compile_start:.2f} secunde)")
except Exception as e:
    print(f"❌ EROARE COMPILARE: {e}")
    exit()

# --- 3. DESENARE CURBA ---
def draw_curved_path(img, obstacle_box, direction_sign):
    h, w = img.shape[:2]
    start_pt = np.array([w // 2, h - 20])
    end_pt = np.array([w // 2, h // 3])

    ox1, oy1, ox2, oy2 = obstacle_box
    obs_center_y = int((oy1 + oy2) / 2)
    safety_margin = 120

    if direction_sign > 0:
        apex_x = min(w - 30, int(ox2 + safety_margin))
    else:
        apex_x = max(30, int(ox1 - safety_margin))

    apex_pt = np.array([apex_x, obs_center_y])
    p1 = start_pt + np.array([int(direction_sign * 80), -80])
    p2 = apex_pt
    p3 = np.array([w // 2, obs_center_y - 120])
    p4 = end_pt

    path_points = []
    for t in np.linspace(0, 1, 40):
        q0 = (1 - t) * start_pt + t * p1
        q1 = (1 - t) * p1 + t * p2
        q2 = (1 - t) * p2 + t * p3
        q3 = (1 - t) * p3 + t * p4
        r0 = (1 - t) * q0 + t * q1
        r1 = (1 - t) * q1 + t * q2
        r2 = (1 - t) * q2 + t * q3
        point = (1 - t) * r0 + t * r1
        path_points.append(point.astype(np.int32))

    cv2.polylines(img, [np.array(path_points)], False, (0, 255, 255), 4, cv2.LINE_AA)
    if len(path_points) > 5:
        cv2.arrowedLine(img, tuple(path_points[-5]), tuple(path_points[-1]), (0, 255, 255), 4, tipLength=0.5)

# --- 4. MODELE (Cu Prioritate Engine) ---
def load_optimized_model(path_prefix, task='detect'):
    if os.path.exists(path_prefix + ".engine"):
        print(f"🚀 Incarc Engine (Viteza Maxima): {path_prefix}.engine")
        return YOLO(path_prefix + ".engine", task=task)
    elif os.path.exists(path_prefix + ".onnx"):
        print(f"🚄 Incarc ONNX: {path_prefix}.onnx")
        return YOLO(path_prefix + ".onnx", task=task)
    elif os.path.exists(path_prefix + ".pt"):
        print(f"⚠️ Incarc PT: {path_prefix}.pt")
        return YOLO(path_prefix + ".pt")
    return None

print("\n>>> Incarc Modelele AI...")
t_models_start = time.time()
model_cars = load_optimized_model("yolov8n") # Fara extensie, cauta singur
if model_cars is None: model_cars = YOLO("yolov8n.pt")
model_potholes = load_optimized_model(POTHOLE_MODEL_PATH)
t_models_end = time.time()
print(f"Modele incarcate in {t_models_end - t_models_start:.2f} secunde.")

# --- 5. PROCESARE ---
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

print(f"\n>>> Am gasit {len(image_files)} imagini.")

if len(image_files) == 0:
    print("⚠️ ATENTIE: Nu am gasit imagini! Verifica folderul images_in.")
    exit()

for i, img_path in enumerate(image_files):
    t_start_frame = time.time()

    # 1. LOAD IMAGE
    t_load_start = time.time()
    frame = cv2.imread(img_path)
    t_load_end = time.time()

    if frame is None: continue
    h, w, _ = frame.shape

    # 2. INFERENTA (ZERO-COPY: Datele raman pe GPU!)
    t_infer_start = time.time()
    
    gpu_boxes_list = [] # Lista pentru tensorii CUDA

    # Vehicule
    res_cars = model_cars(frame, verbose=False, classes=[2, 5, 7])
    if res_cars[0].boxes.shape[0] > 0:
        # Păstrăm pe GPU (.xyxy), NU facem .cpu() aici!
        gpu_boxes_list.append(res_cars[0].boxes.xyxy)

    # Gropi
    if model_potholes:
        res_holes = model_potholes(frame, verbose=False, conf=0.25)
        if res_holes[0].boxes.shape[0] > 0:
             # Păstrăm pe GPU
            gpu_boxes_list.append(res_holes[0].boxes.xyxy)
    
    t_infer_end = time.time()

    # 3. LOGICA (Concatenare + Kernel pe GPU)
    t_logic_start = time.time()

    # Concatenare instanta pe VRAM
    if len(gpu_boxes_list) > 0:
        final_boxes_gpu = torch.cat(gpu_boxes_list, dim=0)
    else:
        final_boxes_gpu = torch.empty((0, 4), device='cuda')

    # Trimitem direct tensorul GPU la kernelul C++
    raw_steering = 0.0
    if final_boxes_gpu.shape[0] > 0:
        # Asigurare ca e pe cuda (desi ar trebui sa fie deja)
        if final_boxes_gpu.device.type != 'cuda': 
            final_boxes_gpu = final_boxes_gpu.cuda()
            
        raw_steering = avoidance_cuda.compute_steering(final_boxes_gpu, float(w))

    # Controler PD
    p_term = raw_steering * KP
    d_term = (raw_steering - last_raw_steering) * KD
    final_steering = p_term + d_term
    last_raw_steering = raw_steering

    t_logic_end = time.time()

    # 4. VIZUALIZARE & SALVARE (Doar aici mutam pe CPU - Partea Lenta)
    t_vis_start = time.time()

    if VISUALIZE and SAVE_OUTPUT:
        closest_obstacle_box = None
        max_y2 = -1
        
        if final_boxes_gpu.shape[0] > 0:
            # ACUM transferam pe CPU pentru desenare
            cpu_boxes = final_boxes_gpu.cpu().numpy()
            
            # Recalculam logica de "cel mai apropiat" pentru desenare
            # (Nu am facut-o in inferenta ca sa nu incetinim AI-ul)
            for box in cpu_boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Cautam obstacolul critic pentru colorare
                if y2 > max_y2:
                    max_y2 = y2
                    closest_obstacle_box = box

            # Desenam toate cutiile
            for box in cpu_boxes:
                x1, y1, x2, y2 = map(int, box)
                is_closest = (closest_obstacle_box is not None and np.array_equal(box, closest_obstacle_box))
                color = (0, 0, 255) if is_closest else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Desenam sageata / curba
        if closest_obstacle_box is not None and abs(final_steering) > 0.3:
            direction = np.sign(final_steering)
            draw_curved_path(frame, closest_obstacle_box, direction)
            cv2.putText(frame, "OCOLIRE", (w // 2 - 50, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            center_x = w // 2
            cv2.arrowedLine(frame, (center_x, h - 50), (center_x, h - 200), (0, 255, 0), 4, tipLength=0.3)

        save_path = os.path.join(OUTPUT_FOLDER, "result_" + os.path.basename(img_path))
        cv2.imwrite(save_path, frame)

    t_vis_end = time.time()
    t_end_frame = time.time()

    # Statistici
    time_load = (t_load_end - t_load_start) * 1000
    time_infer = (t_infer_end - t_infer_start) * 1000
    time_logic = (t_logic_end - t_logic_start) * 1000
    time_vis = (t_vis_end - t_vis_start) * 1000
    fps = 1.0 / (t_end_frame - t_start_frame)

    print(f"Img {i} | Steer: {final_steering:.2f} | FPS: {fps:.1f} || "
          f"Load: {time_load:.1f}ms | Infer: {time_infer:.1f}ms | "
          f"Logic: {time_logic:.1f}ms | Vis/Save: {time_vis:.1f}ms")

print("\n✅ Procesare completa.")