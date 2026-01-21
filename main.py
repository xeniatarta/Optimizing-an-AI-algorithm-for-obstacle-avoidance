import torch
from torch.utils.cpp_extension import load
from ultralytics import YOLO
import cv2
import glob
import os
import time
import numpy as np
from threading import Thread
import queue

# --- CONFIGURARE ---
INPUT_FOLDER = "images_in"
OUTPUT_FOLDER = "images_out"

# MODEL UNIFICAT
MODEL_PATH = "runs/detect/yolo_unified2/weights/best" 

VISUALIZE = True
SAVE_OUTPUT = False 

# Controler PD
KP = 1.0
KD = 0.6
last_raw_steering = 0.0

# --- CLASA 1: CITIRE RAPIDA ---
class ThreadedImageLoader:
    def __init__(self, file_list, queue_size=4):
        self.file_list = file_list
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        for img_path in self.file_list:
            if self.stopped: break
            frame = cv2.imread(img_path)
            if frame is not None:
                self.q.put((img_path, frame))
        self.stopped = True
        self.q.put(None)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True

# --- CLASA 2: SCRIERE RAPIDA ---
class ThreadedImageWriter:
    def __init__(self, queue_size=10):
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def save(self, frame, path):
        if self.stopped: return
        try:
            self.q.put_nowait((frame.copy(), path))
        except queue.Full:
            pass 

    def update(self):
        while not self.stopped:
            try:
                data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame, path = data
            cv2.imwrite(path, frame)
            self.q.task_done()

    def stop(self):
        self.stopped = True
        self.thread.join()

# Verificari foldere
if not os.path.exists(INPUT_FOLDER):
    print(f" EROARE: Folderul '{INPUT_FOLDER}' nu exista!")
    exit()
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"CUDA Disponibil: {torch.cuda.is_available()}")

# --- 2. COMPILARE CUDA ---
print(">>> Compilez algoritmul CUDA...")
try:
    avoidance_cuda = load(
        name="avoidance_cuda",
        sources=["csrc/avoidance.cpp", "csrc/avoidance_kernel.cu"],
        extra_cflags=['/std:c++17', '/O2'],
        extra_cuda_cflags=['-allow-unsupported-compiler', '-O3', '--use_fast_math'],
        verbose=False
    )
    print("✅ Compilare reusita!")
except Exception as e:
    print(f"❌ EROARE COMPILARE: {e}")
    exit()

# --- 3. LOGICA ---
def check_center_path(boxes, img_w, img_h):
    car_width = img_w * 0.35 
    lane_left = int((img_w - car_width) // 2)
    lane_right = int((img_w + car_width) // 2)
    look_ahead_y = img_h * 0.5 
    is_clear = True
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        if y2 < look_ahead_y: continue
        if (x2 > lane_left) and (x1 < lane_right):
            is_clear = False
            break
    return is_clear, lane_left, lane_right

def draw_car_tracks(img, steering, obstacle_box=None, is_straight=True):
    """ Deseneaza sinele rotilor cu perspectiva 3D (Subtire si Precis). """
    h, w = img.shape[:2]
    
    # Sincronizare cu CUDA (procente din latime)
    car_total_w_bottom = int(w * 0.45)   
    tire_w_bottom = int(w * 0.12)
    car_total_w_horizon = int(w * 0.10)
    tire_w_horizon = int(w * 0.03)
    
    start_pt = np.array([w // 2, h])
    center_points = []
    num_points = 40
    
    # --- 1. GENERARE TRAIECTORIE CENTRALA ---
    if is_straight:
        # Linie dreapta simpla
        end_x = w // 2 + int(steering * 150)
        end_y = int(h * 0.4)
        for t in np.linspace(0, 1, num_points):
            x = int((1 - t) * start_pt[0] + t * end_x)
            y = int((1 - t) * start_pt[1] + t * end_y)
            center_points.append([x, y])
        color_tracks = (0, 255, 0) # Verde
    else:
        # Logica curba (Bezier) pentru ocolire lina
        if obstacle_box is not None:
            ox1, oy1, ox2, oy2 = map(int, obstacle_box)
            obs_center_y = int((oy1 + oy2) / 2)
            direction = np.sign(steering) if abs(steering) > 0.01 else 1
            
            # Calculam punctul de ocolire (Apex)
            dist_factor = (h - obs_center_y) / h
            current_car_w = car_total_w_horizon + (car_total_w_bottom - car_total_w_horizon) * dist_factor
            safety_offset = int(current_car_w / 2) + 20 
            
            if direction > 0: apex_x = min(w - 20, ox2 + safety_offset)
            else: apex_x = max(20, ox1 - safety_offset)
                
            p0 = start_pt
            p1 = start_pt + np.array([int(direction * 100), -int(h * 0.2)])
            p2 = np.array([apex_x, obs_center_y])
            p3 = np.array([w // 2, int(h * 0.35)])

            # Interpolare Bezier
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
        else:
            # Fallback (Curba lina fara obstacol specific) - REPARATIA E AICI
            end_x = w // 2 + int(steering * 200)
            end_y = int(h * 0.4)
            # Folosim o curba patratica simpla
            ctrl_x = w // 2 + int(steering * 100)
            ctrl_y = int(h * 0.7)
            
            p0 = start_pt
            p1 = np.array([ctrl_x, ctrl_y])
            p2 = np.array([end_x, end_y])
            
            for t in np.linspace(0, 1, num_points):
                pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                center_points.append(pt.astype(int))

        color_tracks = (0, 255, 255) # Galben

    # --- 2. DESENARE SINE ROTI (Linii Fine) ---
    left_track = []
    right_track = []
    
    if len(center_points) > 0:
        for i, pt in enumerate(center_points):
            t = i / (num_points - 1)
            
            # Interpolare perspectiva pentru latimea masinii
            current_total_w = car_total_w_bottom * (1-t) + car_total_w_horizon * t
            current_tire_w = tire_w_bottom * (1-t) + tire_w_horizon * t
            
            half_w = current_total_w / 2
            
            # Coordonate X pentru roti
            lx = pt[0] - half_w + (current_tire_w / 2)
            rx = pt[0] + half_w - (current_tire_w / 2)
            
            left_track.append([int(lx), pt[1]])
            right_track.append([int(rx), pt[1]])

        pts_l = np.array(left_track, np.int32).reshape((-1, 1, 2))
        pts_r = np.array(right_track, np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts_l], False, color_tracks, 2, cv2.LINE_AA)
        cv2.polylines(img, [pts_r], False, color_tracks, 2, cv2.LINE_AA)
        
        # Traverse
        for i in range(0, num_points, 5):
             if i < len(left_track) and i < len(right_track):
                cv2.line(img, tuple(left_track[i]), tuple(right_track[i]), color_tracks, 1, cv2.LINE_AA)

    status_text = "STRADDLE" if (is_straight and obstacle_box is not None) else ("LIBER" if is_straight else "OCOLIRE")
    cv2.putText(img, status_text, (w//2 - 60, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tracks, 2)

# --- 4. INCARCARE MODEL ---
def load_optimized_model(path_prefix, task='detect'):
    if os.path.exists(path_prefix + ".engine"):
        print(f"🚀 Engine: {path_prefix}.engine")
        return YOLO(path_prefix + ".engine", task=task)
    elif os.path.exists(path_prefix + ".pt"):
        print(f"⚠️ PT: {path_prefix}.pt")
        return YOLO(path_prefix + ".pt")
    return None

print("\n>>> Incarc Modelul Unificat...")
model = load_optimized_model(MODEL_PATH)
if model is None: model = YOLO("yolov8n.pt")
print("✅ Model incarcat.")

# --- 5. PROCESARE ---
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
print(f"\n>>> Procesez {len(image_files)} imagini.")

loader = ThreadedImageLoader(image_files).start()
writer = ThreadedImageWriter().start() 

i = 0
last_vis_time = 0
VIS_INTERVAL = 0.033 

while True:
    data = loader.read()
    if data is None: break
    img_path, frame = data
    if frame is None: continue
    
    t_start = time.time()
    h, w = frame.shape[:2]

    # 1. INFERENTA
    gpu_boxes_list = [] 
    if model:
        # half=True e OK aici
        res = model(frame, verbose=False, conf=0.40, imgsz=640, half=True)
        if res[0].boxes.shape[0] > 0:
            gpu_boxes_list.append(res[0].boxes.xyxy)

    # 2. CUDA
    if len(gpu_boxes_list) > 0:
        final_boxes_gpu = torch.cat(gpu_boxes_list, dim=0)
        # REPARATIE CRITICA: Convertim la FLOAT32 pentru C++
        final_boxes_gpu = final_boxes_gpu.float() 
    else:
        final_boxes_gpu = torch.empty((0, 4), device='cuda')

    raw_steering = 0.0
    if final_boxes_gpu.shape[0] > 0:
        if final_boxes_gpu.device.type != 'cuda': final_boxes_gpu = final_boxes_gpu.cuda()
        raw_steering = avoidance_cuda.compute_steering(final_boxes_gpu, float(w))

    p_term = raw_steering * KP
    d_term = (raw_steering - last_raw_steering) * KD
    final_steering = p_term + d_term
    last_raw_steering = raw_steering
    
    # 3. VIZUALIZARE
    should_show = VISUALIZE and (time.time() - last_vis_time > VIS_INTERVAL)
    
    if SAVE_OUTPUT or should_show:
        cpu_boxes = []
        if final_boxes_gpu.shape[0] > 0:
             cpu_boxes = final_boxes_gpu.cpu().numpy()
             is_clear, _, _ = check_center_path(cpu_boxes, w, h)
             
             closest_obstacle_box = None
             max_y2 = -1
             for box in cpu_boxes:
                 x1, y1, x2, y2 = map(int, box)
                 lane_center = w // 2
                 lane_width_px = w * 0.35
                 is_blocking = (x2 > lane_center - lane_width_px/2) and \
                               (x1 < lane_center + lane_width_px/2) and \
                               (y2 > h * 0.4)
                 
                 color = (0, 0, 255) if (not is_clear and is_blocking) else (0, 255, 0)
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                 
                 if not is_clear and is_blocking and y2 > max_y2:
                    max_y2 = y2
                    closest_obstacle_box = box
             
             draw_car_tracks(frame, final_steering, obstacle_box=closest_obstacle_box, is_straight=is_clear)
        else:
             draw_car_tracks(frame, 0.0, is_straight=True)

        if SAVE_OUTPUT:
            save_path = os.path.join(OUTPUT_FOLDER, "res_" + os.path.basename(img_path))
            writer.save(frame, save_path) 

        if should_show:
            fps_disp = 1.0 / (time.time() - t_start)
            cv2.putText(frame, f"FPS: {fps_disp:.0f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Preview", frame)
            cv2.waitKey(1)
            last_vis_time = time.time()

    fps = 1.0 / (time.time() - t_start)
    print(f"Img {i} | FPS: {fps:.1f} | Steer: {final_steering:.2f}")
    i += 1

loader.stop()
writer.stop() 
cv2.destroyAllWindows()
print("\n✅ Gata.")