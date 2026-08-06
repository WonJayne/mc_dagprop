"""Resolve extension build flags for release and instrumented builds."""

from __future__ import annotations


def resolve_platform_build_flags(platform: str, *, instrumented: bool, use_lto: bool) -> tuple[list[str], list[str]]:
    """Return package-owned compile and linker flags for one target platform."""
    if platform == "win32":
        compile_args = ["/std:c++20"]
        linker_args = ["/INCREMENTAL:NO"]
        if not instrumented:
            compile_args[:0] = [
                "/O2",  # optimize for speed
                "/Ot",  # favor speed over size
                "/Ob2",  # inline any suitable functions
                "/Oi",  # generate intrinsic functions for memcpy etc.
                "/Oy",  # omit frame pointers
                "/fp:precise",  # preserve IEEE-sensitive numerical checks
                "/Gy",  # enable function-level linking
            ]
            if use_lto:
                compile_args.append("/GL")
                linker_args.append("/LTCG")
        return compile_args, linker_args

    compile_args = ["-std=c++20"]
    linker_args: list[str] = []
    if not instrumented:
        compile_args.insert(0, "-O3")
        if use_lto:
            compile_args.append("-flto")
            linker_args.append("-flto")

    if platform.startswith("linux"):
        compile_args.append("-fvisibility=hidden")

    return compile_args, linker_args
