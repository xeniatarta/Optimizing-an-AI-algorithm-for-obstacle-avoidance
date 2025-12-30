from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Numele extensiei pe care o vom importa in Python
ext_modules = [
    CUDAExtension(
        name='avoidance_cuda',
        sources=[
            'csrc/avoidance.cpp',       # Fisierul C++ (Podul)
            'csrc/avoidance_kernel.cu', # Fisierul CUDA (Matematica)
        ],
    )
]

setup(
    name='avoidance_cuda',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)