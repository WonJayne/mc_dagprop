from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from demo.inspection import retrieve_absolute_and_relative_delays
from scripts.build_flags import resolve_platform_build_flags
from scripts.check_release_tag import read_project_version, require_matching_tag
from scripts.run_readme_examples import extract_python_blocks

from mc_dagprop import Activity, DagContext, Event, EventTimestamp, GenericDelayGenerator, MonteCarloPropagator


def test_configured_release_tag_matches_project_version() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    version = read_project_version(repository_root / "pyproject.toml")
    require_matching_tag(f"v{version}", version)


def test_mismatched_release_tag_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match package version"):
        require_matching_tag("v1.0.0", "1.0.0rc1")


def test_setup_loads_build_flags_without_checkout_on_import_path(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-I", str(repository_root / "setup.py"), "--name"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "mc_dagprop"


def test_publish_workflow_requires_quality_and_sanitizer_gates() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    publish_workflow = (repository_root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    assert "uses: ./.github/workflows/quality.yml" in publish_workflow
    assert "uses: ./.github/workflows/sanitizers.yml" in publish_workflow
    assert "needs: [build_wheels, test_sdist, quality_gate, sanitizer_gate]" in publish_workflow


def test_cpp_coverage_build_creates_installed_metadata() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    quality_workflow = (repository_root / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    assert "python setup.py egg_info" in quality_workflow
    assert "python setup.py build_ext --inplace --force" in quality_workflow


def test_quality_builds_native_module_before_unqualified_type_check() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    quality_workflow = (repository_root / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    native_build = quality_workflow.index("poetry run python setup.py build_ext --inplace --force")
    openbus_checks = quality_workflow.index("run: ./scripts/check.sh")
    assert native_build < openbus_checks


def test_source_tree_type_build_dependencies_are_in_the_locked_dev_environment() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    development_dependencies = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]
    assert development_dependencies["setuptools"] == ">=77,<83"
    assert development_dependencies["pybind11"] == ">=2.13"


def test_contributor_entrypoints_use_only_the_openbus_quality_toolchain() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    contributor_entrypoints = (
        "README.md",
        "devtools.cmd",
        "pipeline.sh",
        "pipeline.bat",
        "scripts/check.sh",
        "scripts/check.bat",
        "scripts/format.sh",
        "scripts/format.bat",
    )
    forbidden_tools = ("mypy", "pylint", "isort", "radon")

    for relative_path in contributor_entrypoints:
        contents = (repository_root / relative_path).read_text(encoding="utf-8").lower()
        assert all(tool not in contents for tool in forbidden_tools), relative_path


@pytest.mark.parametrize("workflow_name", ["quality.yml", "sanitizers.yml"])
def test_release_gate_workflow_is_reusable(workflow_name: str) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    workflow = (repository_root / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")

    assert "  workflow_call:" in workflow


def test_address_sanitizer_preloads_the_cpp_exception_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    sanitizer_workflow = (repository_root / ".github" / "workflows" / "sanitizers.yml").read_text(encoding="utf-8")

    asan_position = sanitizer_workflow.index("libasan.so")
    cpp_runtime_position = sanitizer_workflow.index("libstdc++.so.6")
    assert asan_position < cpp_runtime_position


def test_readme_contains_executable_python_examples() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    blocks = extract_python_blocks((repository_root / "README.md").read_text(encoding="utf-8"))
    assert len(blocks) >= 3


def test_pypi_metadata_uses_canonical_project_links() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
        project_urls = tomllib.load(pyproject_file)["project"]["urls"]
    readme = (repository_root / "README.md").read_text(encoding="utf-8")

    assert project_urls == {
        "Repository": "https://github.com/WonJayne/mc_dagprop",
        "Documentation": "https://github.com/WonJayne/mc_dagprop/blob/main/docs/semantics.md",
        "Issues": "https://github.com/WonJayne/mc_dagprop/issues",
    }
    for relative_target in (
        "](THIRD_PARTY_NOTICES.md)",
        "](docs/semantics.md)",
        "](docs/openbus_integration.md)",
        "](RELEASE_NOTES.md)",
    ):
        assert relative_target not in readme


@pytest.mark.parametrize(
    ("platform", "optimization_flag", "compile_lto_flag", "link_lto_flag"),
    [("linux", "-O3", "-flto", "-flto"), ("darwin", "-O3", "-flto", "-flto"), ("win32", "/O2", "/GL", "/LTCG")],
)
def test_release_extension_build_keeps_optimization_and_lto(
    platform: str, optimization_flag: str, compile_lto_flag: str, link_lto_flag: str
) -> None:
    compile_args, linker_args = resolve_platform_build_flags(platform, instrumented=False, use_lto=True)

    assert optimization_flag in compile_args
    assert compile_lto_flag in compile_args
    assert link_lto_flag in linker_args


def test_windows_release_build_does_not_enable_fast_math() -> None:
    compile_args, _ = resolve_platform_build_flags("win32", instrumented=False, use_lto=True)

    assert "/fp:precise" in compile_args
    assert "/fp:fast" not in compile_args


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_instrumented_extension_build_does_not_override_instrumentation_flags(platform: str) -> None:
    compile_args, linker_args = resolve_platform_build_flags(platform, instrumented=True, use_lto=True)

    package_optimization_flags = {"-O3", "/O2", "/Ot", "/Ob2", "/Oi", "/Oy", "/fp:fast", "/Gy"}
    package_lto_flags = {"-flto", "/GL", "/LTCG"}
    assert package_optimization_flags.isdisjoint(compile_args)
    assert package_lto_flags.isdisjoint([*compile_args, *linker_args])


def test_inspection_demo_helper_runs_on_a_public_simulation_result() -> None:
    events = (Event("source", EventTimestamp(0.0, 10.0, 0.0)), Event("target", EventTimestamp(0.0, 10.0, 0.0)))
    context = DagContext(events, {(0, 1): Activity(0, 2.0, 1)}, ((1, ((0, 0),)),))
    result = MonteCarloPropagator(context, GenericDelayGenerator()).run(0)

    absolute, relative = retrieve_absolute_and_relative_delays(context, result)

    assert absolute == {1: [0.0]}
    assert relative == {1: [0.0]}
