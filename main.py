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
# Calea de baza (fara extensie) - scriptul cauta singur .engine, .onnx, .pt
POTHOLE_MODEL_PATH = "runs/detect/yolo_potholes3/weights/best"

VISUALIZE = True
SAVE_OUTPUT = True

# Controler PD (Pentru stabilitatea directiei calculate)
KP = 1.0
KD = 0.6
last_raw_steering = 0.0

# 1. Verificam structura de foldere
if not os.path.exists(INPUT_FOLDER):
    print(f" EROARE: Folderul '{INPUT_FOLDER}' nu exista! Creeaza-l langa main.py.")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"CUDA Disponibil: {torch.cuda.is_available()}")

# --- 2. COMPILARE CUDA (JIT) ---
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
    print("SFAT: Verifica daca ai instalat Visual Studio si CUDA Toolkit corect.")
    exit()


# --- 3. FUNCTII LOGICA & VIZUALIZARE ---

def check_center_path(boxes, img_w, img_h):
    """
    Verifica daca tunelul central din fata masinii este liber.
    Returneaza: (is_clear, lane_left_x, lane_right_x)
    """
    # Presupunem ca masina ocupa 35% din latimea imaginii la baza
    car_width = img_w * 0.35 
    lane_left = int((img_w - car_width) // 2)
    lane_right = int((img_w + car_width) // 2)
    
    # Ne uitam pana la 50% din inaltime (nu ne intereseaza obstacolele de la orizont indepartat)
    look_ahead_y = img_h * 0.5 

    is_clear = True
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Ignoram obstacolele care sunt prea sus (prea departe)
        if y2 < look_ahead_y:
            continue

        # Verificam intersectia pe orizontala
        # Daca dreptunghiul obstacolului se suprapune cu banda noastra
        if (x2 > lane_left) and (x1 < lane_right):
            is_clear = False
            break
            
    return is_clear, lane_left, lane_right

def draw_car_tracks(img, steering, obstacle_box=None, is_straight=True):
    h, w = img.shape[:2]
    
    # --- CONFIGURARE PERSPECTIVA (Sincronizat cu CUDA) ---
    # Aceste valori trebuie sa reflecte ce am pus in avoidance_kernel.cu
    # CAR_TOTAL_WIDTH era 0.45 (45%)
    # TIRE_WIDTH era 0.12 (12%)
    
    car_total_w_bottom = int(w * 0.45)   
    tire_w_bottom = int(w * 0.12)
    
    # La orizont, totul e mai mic (efect 3D)
    car_total_w_horizon = int(w * 0.10)
    tire_w_horizon = int(w * 0.03)
    
    start_pt = np.array([w // 2, h])
    
    # Generare puncte centrale (Coloana vertebrala)
    center_points = []
    num_points = 40
    
    if is_straight:
        end_x = w // 2 + int(steering * 150) # Sensibilitate vizuala
        end_y = int(h * 0.4)
        
        for t in np.linspace(0, 1, num_points):
            x = int((1 - t) * start_pt[0] + t * end_x)
            y = int((1 - t) * start_pt[1] + t * end_y)
            center_points.append([x, y])
        color_tracks = (0, 255, 0) # Verde
    else:
        # Logica curba (Bezier) ramane similara, dar mai fina
        if obstacle_box is not None:
            ox1, oy1, ox2, oy2 = map(int, obstacle_box)
            obs_center_y = int((oy1 + oy2) / 2)
            direction = np.sign(steering) if abs(steering) > 0.01 else 1
            
            # Apex calculation
            dist_factor = (h - obs_center_y) / h
            current_car_w = car_total_w_horizon + (car_total_w_bottom - car_total_w_horizon) * dist_factor
            
            # Ocolim la limita (jumatate de masina + marja mica)
            safety_offset = int(current_car_w / 2) + 20 
            
            if direction > 0: apex_x = min(w - 20, ox2 + safety_offset)
            else: apex_x = max(20, ox1 - safety_offset)
                
            p0 = start_pt
            p1 = start_pt + np.array([int(direction * 100), -int(h * 0.2)])
            p2 = np.array([apex_x, obs_center_y])
            p3 = np.array([w // 2, int(h * 0.35)])

            for t in np.linspace(0, 1, num_points):
                if t < 0.5:
                    tt = t * 2
                    q0 = (1-tt)*p0 + tt*p1
                    q1 = (1-tt)*p1 + tt*p2
                    pt = (1-tt)*q0 + tt*q1
                else:
                    tt = (t - 0.5) * 2
                    q0 = (1-tt)*p2 + tt*p3
                    q1 = (1-tt)*p3 + tt*p3
                    pt = (1-tt)*q0 + tt*q1
                center_points.append(pt.astype(int))
        color_tracks = (0, 255, 255) # Galben

    # --- DESENARE SINE ROTI (Linii Fine) ---
    left_track_inner = []
    left_track_outer = []
    right_track_inner = []
    right_track_outer = []
    
    for i, pt in enumerate(center_points):
        t = i / (num_points - 1)
        
        # Interpolare latimi
        current_total_w = car_total_w_bottom * (1-t) + car_total_w_horizon * t
        current_tire_w = tire_w_bottom * (1-t) + tire_w_horizon * t
        
        half_w = current_total_w / 2
        
        # Coordonate X pentru roti
        lx_center = pt[0] - half_w + (current_tire_w / 2)
        rx_center = pt[0] + half_w - (current_tire_w / 2)
        
        # Putem desena conturul rotii sau doar o linie subtire pe mijlocul ei
        # Aici desenam o linie care reprezinta centrul anvelopei
        left_track_inner.append([int(lx_center), pt[1]])
        right_track_inner.append([int(rx_center), pt[1]])

    # Desenam cu thickness=2 (subtire)
    pts_l = np.array(left_track_inner, np.int32).reshape((-1, 1, 2))
    pts_r = np.array(right_track_inner, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(img, [pts_l], False, color_tracks, 2, cv2.LINE_AA)
    cv2.polylines(img, [pts_r], False, color_tracks, 2, cv2.LINE_AA)
    
    # Traverse (Traversele sunt utile pentru perceptia adancimii)
    for i in range(0, num_points, 5):
        cv2.line(img, tuple(left_track_inner[i]), tuple(right_track_inner[i]), color_tracks, 1, cv2.LINE_AA)

    status_text = "STRADDLE (CENTRU)" if (is_straight and obstacle_box is not None) else ("LIBER" if is_straight else "OCOLIRE")
    cv2.putText(img, status_text, (w//2 - 80, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tracks, 2)


# --- 4. SELECTOR MODELE (Engine > ONNX > PT) ---
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

# --- MAIN: INCARCARE SI PROCESARE ---

print("\n>>> Incarc Modelele AI...")
# Modelul standard (masini)
model_cars = load_optimized_model("runs/detect/yolo_animals/weights/best") 
if model_cars is None: model_cars = YOLO("yolov8n.pt")

# Modelul gropi
model_potholes = load_optimized_model(POTHOLE_MODEL_PATH)
print("Modele incarcate.")

# Cautam imagini
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

print(f"\n>>> Am gasit {len(image_files)} imagini.")

if len(image_files) == 0:
    print("⚠️ Nu am gasit imagini! Pune poze in folderul 'images_in'.")
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
    
    gpu_boxes_list = [] # Aici adunam rezultatele de la ambele modele

    # Vehicule
    res_cars = model_cars(frame, verbose=False, classes=[2, 5, 7])
    if res_cars[0].boxes.shape[0] > 0:
        gpu_boxes_list.append(res_cars[0].boxes.xyxy) # Pastram pe GPU

    # Gropi
    if model_potholes:
        res_holes = model_potholes(frame, verbose=False, conf=0.25)
        if res_holes[0].boxes.shape[0] > 0:
            gpu_boxes_list.append(res_holes[0].boxes.xyxy) # Pastram pe GPU
    
    t_infer_end = time.time()

    # 3. LOGICA (Concatenare + Kernel pe GPU)
    t_logic_start = time.time()

    # Concatenare instanta pe VRAM
    if len(gpu_boxes_list) > 0:
        final_boxes_gpu = torch.cat(gpu_boxes_list, dim=0)
    else:
        final_boxes_gpu = torch.empty((0, 4), device='cuda')

    # Trimitem direct tensorul GPU la kernelul C++ pentru calculul fortei
    raw_steering = 0.0
    if final_boxes_gpu.shape[0] > 0:
        if final_boxes_gpu.device.type != 'cuda': 
            final_boxes_gpu = final_boxes_gpu.cuda()
            
        raw_steering = avoidance_cuda.compute_steering(final_boxes_gpu, float(w))

    # Controler PD pentru netezire
    p_term = raw_steering * KP
    d_term = (raw_steering - last_raw_steering) * KD
    final_steering = p_term + d_term
    last_raw_steering = raw_steering

    t_logic_end = time.time()

    # 4. VIZUALIZARE & SALVARE (Aici mutam pe CPU - Partea Lenta, doar pt oameni)
    t_vis_start = time.time()

    if VISUALIZE and SAVE_OUTPUT:
        cpu_boxes = []
        if final_boxes_gpu.shape[0] > 0:
            cpu_boxes = final_boxes_gpu.cpu().numpy()
            
            # A. VERIFICARE CULOAR (GAP DETECTION)
            is_path_clear, l_left, l_right = check_center_path(cpu_boxes, w, h)

            # B. IDENTIFICARE OBSTACOL CRITIC SI DESENARE PATRATE
            closest_obstacle_box = None
            max_y2 = -1
            
            for box in cpu_boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Definim zona masinii
                car_center_x = w // 2
                lane_half_w = (w * 0.35) // 2
                
                # Verificam daca obstacolul blocheaza fizic masina
                is_blocking = (x2 > car_center_x - lane_half_w) and \
                              (x1 < car_center_x + lane_half_w) and \
                              (y2 > h * 0.4)
                
                if not is_path_clear and is_blocking:
                    color = (0, 0, 255) # ROSU = Pericol Iminent
                    # Acesta este obstacolul pe care trebuie sa-l ocolim
                    if y2 > max_y2:
                        max_y2 = y2
                        closest_obstacle_box = box
                else:
                    color = (0, 255, 0) # VERDE = Obstacol, dar nu blocheaza

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # C. DESENARE SINE ROTI
            if is_path_clear:
                # Cazul: Gropi stanga/dreapta (Poza 2), dar tunel liber -> Mergem DREPT
                draw_car_tracks(frame, 0.0, is_straight=True)
            else:
                # Cazul: Obstacol pe mijloc (Poza 1) -> OCOLIM
                # Daca nu am gasit un obstacol specific, folosim doar steering-ul general
                draw_car_tracks(frame, final_steering, obstacle_box=closest_obstacle_box, is_straight=False)

        else:
            # Niciun obstacol detectat -> Mergem DREPT
            draw_car_tracks(frame, 0.0, is_straight=True)

        # Salvare
        save_path = os.path.join(OUTPUT_FOLDER, "result_" + os.path.basename(img_path))
        cv2.imwrite(save_path, frame)

    t_vis_end = time.time()
    t_end_frame = time.time()

    # CALCUL TIMPI
    time_load = (t_load_end - t_load_start) * 1000
    time_infer = (t_infer_end - t_infer_start) * 1000
    time_logic = (t_logic_end - t_logic_start) * 1000
    time_vis = (t_vis_end - t_vis_start) * 1000
    fps = 1.0 / (t_end_frame - t_start_frame)

    print(f"Img {i} | Steer: {final_steering:.2f} | FPS: {fps:.1f} || "
          f"Load: {time_load:.1f}ms | Infer: {time_infer:.1f}ms | "
          f"Logic: {time_logic:.1f}ms | Vis/Save: {time_vis:.1f}ms")

print("\n✅ Procesare completa.")