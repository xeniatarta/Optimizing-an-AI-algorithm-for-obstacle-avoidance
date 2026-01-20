#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// --- CALIBRARE DIMENSIUNI MASINA (Procente din Latimea Imaginii) ---
// Latimea TOTALA a masinii (din exteriorul rotii stangi pana in exteriorul rotii drepte)
#define CAR_TOTAL_WIDTH 0.45f  // 45% din latimea imaginii

// Latimea UNEI ANVELOPE
#define TIRE_WIDTH 0.12f       // 12% din latimea imaginii

// Forta de respingere
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
    float* forces // [0] = Stanga, [1] = Dreapta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // Citire coordonate
    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float x2 = boxes[idx * 4 + 2];
    float y2 = boxes[idx * 4 + 3];

    // Normalizare (0.0 - 1.0)
    float nx1 = x1 / image_width;
    float nx2 = x2 / image_width;
    float ny2 = y2 / image_width; // Proximitate
    
    // Calculam aria pentru filtrare zgomot
    float box_w = nx2 - nx1;
    float box_h = (y2 - y1) / image_width;
    if (box_w * box_h < MIN_AREA_THRESHOLD) return;

    // --- DEFINIRE GEOMETRIE ROTI ---
    float center = 0.5f;
    float half_total = CAR_TOTAL_WIDTH / 2.0f;
    
    // Zona Roata STANGA: [outer_edge_L, inner_edge_L]
    float L_outer = center - half_total;
    float L_inner = center - half_total + TIRE_WIDTH;

    // Zona Roata DREAPTA: [inner_edge_R, outer_edge_R]
    float R_inner = center + half_total - TIRE_WIDTH;
    float R_outer = center + half_total;

    // Factor proximitate (cubica pentru reactie lina dar ferma aproape)
    float proximity = ny2 * ny2 * ny2; 
    
    // --- VERIFICARE COLIZIUNE CU ROTILE ---

    // 1. Verificam coliziunea cu ROATA STANGA
    // Intersectia dintre [nx1, nx2] si [L_outer, L_inner]
    float overlap_L = fmaxf(0.0f, fminf(nx2, L_inner) - fmaxf(nx1, L_outer));
    
    if (overlap_L > 0.0f) {
        // Lovim roata stanga -> Impingem spre DREAPTA
        // Forta proportionala cu cat de mult acoperim roata
        float impact = (overlap_L / TIRE_WIDTH) * REPULSION_STRENGTH * proximity;
        atomicMaxFloat(&forces[0], impact);
    }

    // 2. Verificam coliziunea cu ROATA DREAPTA
    // Intersectia dintre [nx1, nx2] si [R_inner, R_outer]
    float overlap_R = fmaxf(0.0f, fminf(nx2, R_outer) - fmaxf(nx1, R_inner));

    if (overlap_R > 0.0f) {
        // Lovim roata dreapta -> Impingem spre STANGA
        float impact = (overlap_R / TIRE_WIDTH) * REPULSION_STRENGTH * proximity;
        atomicMaxFloat(&forces[1], impact);
    }

    // NOTA: Daca obstacolul este strict intre L_inner si R_inner (in gaura dintre roti),
    // overlap_L si overlap_R vor fi 0.0f, deci forta va fi ZERO. 
    // Masina va merge inainte ("Straddle").
}

void launch_avoidance_kernel(const float* boxes, int num_boxes, float image_width, float* forces, cudaStream_t stream) {
    int blocks = (num_boxes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    potential_field_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(boxes, num_boxes, image_width, forces);
}