"""Tests for the command-line contract: exit codes, reports, backend guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagiana.backends import CPU, GPU, GpuProbe, resolve_backend
from kagiana.cli import EXIT_INPUT_ERROR, EXIT_OK, build_parser, main


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    captured = capsys.readouterr().out
    for command in ("pharm2d", "pharm3d", "backends", "validate"):
        assert command in captured


def test_the_documented_3d_defaults_are_in_force() -> None:
    args = build_parser().parse_args(
        ["pharm3d", "--library", "l.csv", "--pharmacophore", "p.csv",
         "--fdef", "f.fdef", "--out", "o.csv"]
    )
    assert args.conformers == 10
    assert args.max_poses == 20
    assert args.limit_upper == 1.5
    assert args.limit_lower == 0.5
    assert args.timeout == 10.0
    assert args.chunksize == 1_000_000
    assert args.large is False
    assert args.backend == "auto"


@pytest.mark.parametrize(
    ("command", "complete"),
    [
        (
            "pharm2d",
            ["--library", "l.csv", "--query", "q.csv", "--fdef", "f.fdef",
             "--out", "o.csv", "--params", "p.json"],
        ),
        (
            "pharm3d",
            ["--library", "l.csv", "--pharmacophore", "p.csv", "--fdef", "f.fdef",
             "--out", "o.csv"],
        ),
    ],
)
def test_every_path_must_be_given(command: str, complete: list[str]) -> None:
    """No input or output path may fall back to a built-in location."""
    parser = build_parser()
    parser.parse_args([command, *complete])

    # Dropping any one of the required options must be rejected.
    for position in range(0, len(complete), 2):
        without = complete[:position] + complete[position + 2 :]
        with pytest.raises(SystemExit):
            parser.parse_args([command, *without])


def test_backends_reports_this_machine(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backends", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["cpu"]["available"] is True
    assert report["cpu"]["default_jobs"] >= 1
    assert "rdkit_version" in report
    assert "available" in report["gpu"]


def test_validate_passes_on_good_inputs(
    capsys: pytest.CaptureFixture[str],
    library: Path,
    fdef: Path,
    pharmacophore: Path,
    params: Path,
    query: Path,
) -> None:
    code = main(
        [
            "validate",
            "--library", str(library),
            "--fdef", str(fdef),
            "--pharmacophore", str(pharmacophore),
            "--params", str(params),
            "--query", str(query),
            "--json",
        ]
    )
    assert code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_reports_a_feature_type_the_fdef_does_not_define(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, library: Path, fdef: Path
) -> None:
    bogus = tmp_path / "p.csv"
    bogus.write_text(
        "feature_type,X,Y,Z\nDonor,0,0,0\nUnicorn,3,0,0\n", encoding="utf-8"
    )
    code = main(
        ["validate", "--library", str(library), "--fdef", str(fdef),
         "--pharmacophore", str(bogus), "--json"]
    )
    assert code == EXIT_INPUT_ERROR
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any("Unicorn" in error for error in report["errors"])


def test_a_missing_file_exits_with_the_input_error_code(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    code = main(
        [
            "pharm3d",
            "--library", str(tmp_path / "absent.csv"),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(tmp_path / "out.csv"),
            "--quiet",
        ]
    )
    assert code == EXIT_INPUT_ERROR


def test_a_nonsense_parameter_exits_with_the_input_error_code(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    code = main(
        [
            "pharm3d",
            "--library", str(library),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(tmp_path / "out.csv"),
            "--conformers", "0",
            "--quiet",
        ]
    )
    assert code == EXIT_INPUT_ERROR


def test_requesting_the_gpu_without_one_fails_rather_than_pretending(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """There is no CPU fallback inside the GPU extension, so --backend gpu
    must fail loudly instead of quietly screening on the CPU."""
    from kagiana.backends import probe_gpu

    if probe_gpu().available:
        pytest.skip("this machine has a usable GPU")

    code = main(
        [
            "pharm3d",
            "--library", str(library),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(tmp_path / "out.csv"),
            "--backend", "gpu",
            "--quiet",
        ]
    )
    assert code == EXIT_INPUT_ERROR
    assert not (tmp_path / "out.csv").exists()


def test_auto_falls_back_to_the_cpu_when_no_gpu_is_usable() -> None:
    backend, _probe, notes = resolve_backend("auto")
    from kagiana.backends import probe_gpu

    if probe_gpu().available:
        assert backend == GPU
    else:
        assert backend == CPU
        assert any("falling back" in note for note in notes)


def test_asking_for_the_cpu_skips_the_gpu_probe_entirely() -> None:
    backend, probe, notes = resolve_backend("cpu")
    assert backend == CPU
    assert notes == []
    assert probe.available is False


def test_the_probe_serialises_for_machine_consumers() -> None:
    payload = GpuProbe(reasons=["no device"]).to_dict()
    assert payload["available"] is False
    assert payload["reasons"] == ["no device"]
    assert payload["device_count"] == 0
