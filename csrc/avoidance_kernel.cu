#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// AM STERS: #define THREADS_PER_BLOCK 256  <-- Nu mai avem nevoie de asta

// --- CALIBRARE DIMENSIUNI MASINA (Procente din Latimea Imaginii) ---
#define CAR_TOTAL_WIDTH 0.45f  
#define TIRE_WIDTH 0.12f       

#define REPULSION_STRENGTH 180.0f 
#define MIN_AREA_THRESHOLD 0.002f 

// Functie ajutatoare pentru MAX atomic
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void potential_field_kernel(
    const float* boxes, 
    int num_boxes, 
    float image_width, 
    float* forces 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // Citire coordonate
    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float x2 = boxes[idx * 4 + 2];
    float y2 = boxes[idx * 4 + 3];

    // Normalizare
    float nx1 = x1 / image_width;
    float nx2 = x2 / image_width;
    float ny2 = y2 / image_width; 
    
    float box_w = nx2 - nx1;
    float box_h = (y2 - y1) / image_width;
    if (box_w * box_h < MIN_AREA_THRESHOLD) return;

    // --- GEOMETRIE ---
    float center = 0.5f;
    float half_total = CAR_TOTAL_WIDTH / 2.0f;
    
    float L_outer = center - half_total;
    float L_inner = center - half_total + TIRE_WIDTH;

    float R_inner = center + half_total - TIRE_WIDTH;
    float R_outer = center + half_total;

    float proximity = ny2 * ny2 * ny2; 
    
    // 1. Roata STANGA
    float overlap_L = fmaxf(0.0f, fminf(nx2, L_inner) - fmaxf(nx1, L_outer));
    if (overlap_L > 0.0f) {
        float impact = (overlap_L / TIRE_WIDTH) * REPULSION_STRENGTH * proximity;
        atomicMaxFloat(&forces[0], impact);
    }

    // 2. Roata DREAPTA
    float overlap_R = fmaxf(0.0f, fminf(nx2, R_outer) - fmaxf(nx1, R_inner));
    if (overlap_R > 0.0f) {
        float impact = (overlap_R / TIRE_WIDTH) * REPULSION_STRENGTH * proximity;
        atomicMaxFloat(&forces[1], impact);
    }
}

// --- FUNCTIA DE LANSARE OPTIMIZATA DINAMIC ---
void launch_avoidance_kernel(const float* boxes, int num_boxes, float image_width, float* forces, cudaStream_t stream) {
    
    int minGridSize;
    int blockSize;

    // Aici intrebam placa video: "Pentru functia 'potential_field_kernel', 
    // care este cel mai bun numar de thread-uri per block ca sa tinem GPU-ul ocupat 100%?"
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, potential_field_kernel, 0, 0);

    // Calculam numarul de blocuri pe baza blockSize-ului optim returnat de GPU
    int numBlocks = (num_boxes + blockSize - 1) / blockSize;

    // Lansam kernelul
    potential_field_kernel<<<numBlocks, blockSize, 0, stream>>>(boxes, num_boxes, image_width, forces);
}