# encoding: utf-8
# Legacy script, if you want to build without poetry etc..., just bare metal
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class GetPybindInclude:
    def __str__(self) -> str:
        import pybind11

        return pybind11.get_include()


if sys.platform == "win32":
    platform_compile_args = ["/O2", "/std:c++17", "/GL"]
    platform_linker_args = ["/LTCG"]
else:
    platform_compile_args = ["-O3", "-std=c++17", "-flto"]
    platform_linker_args = ["-flto"]

ext_modules = [
    Extension(
        "mc_dagprop._core",
        sources=["mc_dagprop/_core.cpp"],
        include_dirs=[GetPybindInclude()],
        language="c++",
        extra_compile_args=platform_compile_args,
        extra_link_args=platform_linker_args,
    )
]

setup(
    name="mc_dagprop",
    version="0.0.1",
    author="Florian Flükiger",
    description="Fast, Simple, Monte Carlo DAG propagation simulator with user-defined delay distributions.",
    packages=find_packages(include=["mc_dagprop*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    package_data={"mc_dagprop": ["py.typed", "*.pyi"]},
    include_package_data=True,
)
