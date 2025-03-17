from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

setup(
    name='gemlite',
    version="0.4.3",
    url="https://github.com/mobiusml/gemlite/",
    author="Dr. Hicham Badri",
    author_email="hicham@mobiuslabs.com",
    license="Apache 2",
    packages=find_packages(include=["gemlite", "gemlite.*"]),
    package_data={
        "gemlite": ["gemlite/*.py", "configs/*.json"],
    },
    include_package_data=True,
    cmdclass={'build_ext': BuildExtension},
    install_requires=["numpy", "triton>=3.1.0"],
)

# python3 setup.py install
