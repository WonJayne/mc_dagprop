# encoding: utf-8
# Legacy script, if you want to build without poetry etc..., just bare metal
import sys
import tomllib
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class GetPybindInclude:
    def __str__(self) -> str:
        import pybind11

        return pybind11.get_include()


if sys.platform == "win32":
    platform_compile_args = [
        "/O2",  # optimize for speed
        "/Ot",  # favor speed over size
        "/Ob2",  # inline any suitable functions
        "/Oi",  # generate intrinsic functions for memcpy etc.
        "/Oy",  # omit frame pointers
        "/fp:fast",  # fast (non-strict) floating-point
        "/Gy",  # enable function-level linking
        "/GL",  # whole-program optimization
        "/std:c++20",
    ]
    platform_linker_args = ["/LTCG", "/INCREMENTAL:NO"]
else:
    platform_compile_args = ["-O3", "-std=c++20", "-flto"]
    platform_linker_args = ["-flto"]

# Read version from pyproject.toml if available so that the legacy
# setuptools build produces the same package version as the Poetry
# build defined in ``pyproject.toml``.
pyproject_path = Path(__file__).with_name("pyproject.toml")
if pyproject_path.exists():
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    version = pyproject.get("project", {}).get("version", "0.6.0")
else:
    version = "0.6.0"

ext_modules = [
    Extension(
        "mc_dagprop.monte_carlo._core",
        sources=["mc_dagprop/monte_carlo/_core.cpp"],
        include_dirs=[GetPybindInclude(), "mc_dagprop/monte_carlo"],
        language="c++",
        extra_compile_args=platform_compile_args,
        extra_link_args=platform_linker_args,
    )
]

setup(
    name="mc_dagprop",
    version=version,
    author="Florian Flükiger",
    description="Fast, Simple, Monte Carlo DAG propagation simulator with user-defined delay distributions.",
    packages=find_packages(include=["mc_dagprop*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    package_data={"mc_dagprop": ["py.typed"], "mc_dagprop.monte_carlo": ["*.pyi"]},
    include_package_data=True,
)
