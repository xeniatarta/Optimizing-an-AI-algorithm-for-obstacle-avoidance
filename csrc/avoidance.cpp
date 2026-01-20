#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Modificat: primim un array de forte (size 2), nu un singur float
void launch_avoidance_kernel(const float* boxes, int num_boxes, float image_width, float* forces, cudaStream_t stream);

float compute_steering(torch::Tensor boxes, float image_width) {
    TORCH_CHECK(boxes.device().is_cuda(), "Boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.is_contiguous(), "Boxes must be contiguous");
    
    int num_boxes = boxes.size(0);
    if (num_boxes == 0) return 0.0f;

    // ALOCAM 2 FLOATURI: [0] = Presiune din Stanga, [1] = Presiune din Dreapta
    auto forces = torch::zeros({2}, boxes.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_avoidance_kernel(boxes.data_ptr<float>(), num_boxes, image_width, forces.data_ptr<float>(), stream);
    
    // Luam rezultatele pe CPU
    auto forces_cpu = forces.cpu();
    float push_right = forces_cpu[0].item<float>(); // Obstacolele din stanga ne imping dreapta
    float push_left = forces_cpu[1].item<float>();  // Obstacolele din dreapta ne imping stanga
    
    // REZULTATUL FINAL: Lupta intre cele doua presiuni
    // Daca groapa dreapta e mare (push_left mare) si cea stanga e mica (push_right mic),
    // ne vom muta stanga DOAR pana cand intram in raza gropii mici, moment in care push_right creste si ne echilibreaza.
    return push_right - push_left;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_steering", &compute_steering, "Compute steering force (CUDA)");
}