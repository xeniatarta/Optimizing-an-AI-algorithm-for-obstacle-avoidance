#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h> 

// --- CONFIGURARE FIZICA ---
#define REPULSION_STRENGTH 80.0f 
#define SAFETY_MARGIN 0.01f 

// --- CONFIGURARE "BUFFER" (Marja de siguranta) ---
// 0.80 inseamna ca folosim 80% din capacitatea optima a unui bloc.
// Lasam 20% "liber" pentru a reduce densitatea termica.
#define OCCUPANCY_FACTOR 0.80f 

__global__ void potential_field_kernel(
    const float* boxes, 
    int num_boxes, 
    float image_width, 
    float* total_force
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // --- CITIRE OPTIMIZATA ---
    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float x2 = boxes[idx * 4 + 2];
    float y2 = boxes[idx * 4 + 3];

    float box_center_x = (x1 + x2) * 0.5f; 
    float area_norm = ((x2 - x1) * (y2 - y1)) / (image_width * image_width);

    if (area_norm < SAFETY_MARGIN) return; 

    // --- LOGICA GAUSSIANA ---
    float half_width = image_width * 0.5f;
    float relative_pos = (box_center_x - half_width) / half_width;
    float closeness = y2 / image_width; 
    
    float direction = (relative_pos < 0.0f) ? 1.0f : -1.0f;
    float center_weight = expf(-5.0f * relative_pos * relative_pos);

    float force = direction * area_norm * (closeness * closeness) * center_weight * REPULSION_STRENGTH;
    
    atomicAdd(total_force, force);
}

// --- LANSARE INTELIGENTA CU LIMITARE ---
void launch_avoidance_kernel(
    const float* boxes, 
    int num_boxes, 
    float image_width, 
    float* total_force, 
    cudaStream_t stream
) {
    int minGridSize;
    int bestBlockSize;
    
    // 1. Intrebam driverul: Care este MAXIMUL posibil pe acest hardware?
    // Pe Jetson Orin Nano s-ar putea sa zica 512 sau 1024.
    // Pe RTX 3050 s-ar putea sa zica 1024.
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, potential_field_kernel, 0, num_boxes);

    // 2. Aplicam "Frana" voluntara (ex: -20%)
    int safeBlockSize = (int)(bestBlockSize * OCCUPANCY_FACTOR);

    // 3. REGULA DE AUR CUDA: Trebuie sa fie multiplu de 32 (Warp Size)
    // Daca safeBlockSize e 413, il rotunjim in jos la 384 (cel mai apropiat multiplu de 32)
    // Altfel, GPU-ul se blocheaza sau merge prost.
    if (safeBlockSize < 32) safeBlockSize = 32;
    safeBlockSize = (safeBlockSize / 32) * 32;

    // 4. Calculam numarul de blocuri necesar cu noua dimensiune "relaxata"
    int gridSize = (num_boxes + safeBlockSize - 1) / safeBlockSize;

    // Lansam kernelul
    potential_field_kernel<<<gridSize, safeBlockSize, 0, stream>>>(
        boxes, num_boxes, image_width, total_force
    );
    
    // Debug (Optional - va aparea in terminalul unde rulezi python)
    // printf("Max Block: %d | Used Block (80%%): %d | Grid: %d\n", bestBlockSize, safeBlockSize, gridSize);
}