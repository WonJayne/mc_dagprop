# Legacy script, if you want to build without poetry etc..., just bare metal
import os
import runpy
import sys
import tomllib
from pathlib import Path
from typing import Final, Protocol, cast

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class BuildFlagResolver(Protocol):
    def __call__(self, platform: str, *, instrumented: bool, use_lto: bool) -> tuple[list[str], list[str]]: ...


def load_build_flag_resolver() -> BuildFlagResolver:
    """Load package build flags without relying on the backend's import path."""
    script_path = Path(__file__).resolve().parent / "scripts" / "build_flags.py"
    script_namespace = runpy.run_path(str(script_path))
    try:
        resolver = script_namespace["resolve_platform_build_flags"]
    except KeyError as exc:
        raise ImportError(f"build flag resolver is missing from {script_path}") from exc
    if not callable(resolver):
        raise TypeError(f"build flag resolver in {script_path} is not callable")
    return cast(BuildFlagResolver, resolver)


class GetPybindInclude:
    def __str__(self) -> str:
        import pybind11  # noqa: PLC0415

        return pybind11.get_include()


INSTRUMENTED: Final[bool] = os.getenv("MC_DAGPROP_INSTRUMENTED", "0") == "1"
USE_LTO: Final[bool] = not INSTRUMENTED and os.getenv("MC_DAGPROP_ENABLE_LTO", "1") == "1"
resolve_platform_build_flags = load_build_flag_resolver()
platform_compile_args, platform_linker_args = resolve_platform_build_flags(
    sys.platform, instrumented=INSTRUMENTED, use_lto=USE_LTO
)

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
        sources=["src/mc_dagprop/monte_carlo/_core.cpp"],
        include_dirs=[GetPybindInclude(), "src/mc_dagprop/monte_carlo"],
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
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    package_data={"mc_dagprop": ["py.typed"], "mc_dagprop.monte_carlo": ["*.pyi"]},
    include_package_data=True,
)
