#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>           // <--- ASTA LIPSEA
#include <c10/cuda/CUDAStream.h>    // <--- SI ASTA

void launch_avoidance_kernel(const float* boxes, int num_boxes, float image_width, float* total_force, cudaStream_t stream);

float compute_steering(torch::Tensor boxes, float image_width) {
    TORCH_CHECK(boxes.device().is_cuda(), "Boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.is_contiguous(), "Boxes must be contiguous");
    int num_boxes = boxes.size(0);
    if (num_boxes == 0) return 0.0f;
    auto total_force = torch::zeros({1}, boxes.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // <--- ACUM VA MERGE
    launch_avoidance_kernel(boxes.data_ptr<float>(), num_boxes, image_width, total_force.data_ptr<float>(), stream);
    return total_force.item<float>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_steering", &compute_steering, "Compute steering force (CUDA)");
}