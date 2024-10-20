from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_args = [
    "-O3",
    "-use_fast_math",
    "-prec-div=false",
    "-prec-sqrt=false",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # "-gencode=arch=compute_86,code=sum_86" #3090: compute_86/sum_86
    # "-gencode=arch=compute_89,code=sum_89",#4090: compute_89/sum_89
]

setup(
    name='gemlite',
    version="0.3.0",
    url="https://github.com/mobiusml/gemlite/",
    author="Dr. Hicham Badri",
    author_email="hicham@mobiuslabs.com",
    license="Apache 2",
    ext_modules=[
        CUDAExtension('gemlite_lib', [
            "gemlite/cuda_kernels/gemlite_lib.cpp",
            "gemlite/cuda_kernels/gemv_A16fWnO16f_int32packing.cu",
            "gemlite/cuda_kernels/gemv_A8iWnO32i_int32packing.cu",
            "gemlite/cuda_kernels/helper.cu"
            ],
        extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": extra_args,
            }
        ),
    ],
    packages=find_packages(include=["gemlite", "gemlite.*"]),
    package_data={
        "gemlite": ["gemlite/*.py"],
    },
    include_package_data=True,
    cmdclass={'build_ext': BuildExtension},
    install_requires=["numpy", "ninja", "triton>=3.0.0"], #3.0.0+dedb7bdf33
)

# python3 setup.py install
