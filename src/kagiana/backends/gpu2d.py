"""2D screening with the similarity matrix computed on the GPU.

Only part of this work can move to the GPU.  Building a 2D pharmacophore
fingerprint is RDKit graph work on the CPU, so it stays on the worker
processes; what the GPU takes over is the all-against-one bit-vector
comparison.  That means the speed-up here is bounded by fingerprint
generation, and matters most for large libraries.

The intersection count the three scores need is obtained either from
nvMolKit's Tanimoto kernel or, when the packed width is one it will not take,
from a popcount on the GPU.  Both are exact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..pool import KillableWorkerPool
from ..progress import note, progress_bar
from ..query2d import Params2D, build_sig_factory, fingerprint, reference_fingerprint
from ..schema import Status

NAN = float("nan")


def init_worker(
    fdef_path: str, params: Params2D, reference_smiles: str
) -> Callable[[str], tuple[tuple[int, ...], int] | None]:
    """Worker task returning the on-bit indices of a molecule's fingerprint."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    sig_factory = build_sig_factory(fdef_path, params)

    def task(smiles: str) -> tuple[tuple[int, ...], int] | None:
        query_fp = fingerprint(smiles, sig_factory)
        if query_fp is None:
            return None
        return tuple(query_fp.GetOnBits()), query_fp.GetNumBits()

    return task


def pack_uint8(on_bits: list[tuple[int, ...]], n_bits: int) -> np.ndarray:
    """Pack on-bit index lists into a ``(n, ceil(n_bits / 8))`` uint8 matrix."""
    n_bytes = (n_bits + 7) // 8
    packed = np.zeros((len(on_bits), n_bytes), dtype=np.uint8)
    for row, bits in enumerate(on_bits):
        for bit in bits:
            packed[row, bit >> 3] |= np.uint8(1 << (bit & 7))
    return packed


def _intersections_with_nvmolkit(
    reference: np.ndarray, library: np.ndarray, reference_count: int, counts: np.ndarray
) -> np.ndarray | None:
    """Intersection counts from nvMolKit's Tanimoto kernel, or ``None``.

    The kernel reports Tanimoto similarity ``t``; with the two on-bit counts
    known, the intersection follows exactly from ``i = t (a + b) / (1 + t)``.
    Returns ``None`` when the kernel will not accept this packed width, so the
    caller can fall back.
    """
    try:
        import torch
        from nvmolkit.similarity import crossTanimotoSimilarity
    except Exception:  # noqa: BLE001
        return None

    # nvMolKit works on 32-bit words.
    if reference.shape[1] % 4 != 0:
        return None
    try:
        words_ref = reference.view(np.uint32)
        words_lib = library.view(np.uint32)
        similarity = crossTanimotoSimilarity(
            torch.from_numpy(np.ascontiguousarray(words_ref)).cuda(),
            torch.from_numpy(np.ascontiguousarray(words_lib)).cuda(),
        )
        torch.cuda.synchronize()
        tanimoto = similarity.numpy().reshape(-1)
    except Exception:  # noqa: BLE001
        return None

    totals = counts.astype(np.float64) + float(reference_count)
    intersections = np.where(
        tanimoto > 0, tanimoto * totals / (1.0 + tanimoto), 0.0
    )
    return np.rint(intersections).astype(np.int64)


def intersections_numpy(reference: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Intersection counts on the CPU, used as the reference implementation."""
    common = np.bitwise_and(library, reference)
    return np.unpackbits(common, axis=1).sum(axis=1).astype(np.int64)


#: Rows per popcount slice, so that device memory stays bounded no matter how
#: many molecules a chunk holds.
_POPCOUNT_SLICE = 65_536


def _intersections_with_popcount(
    reference: np.ndarray, library: np.ndarray
) -> np.ndarray:
    """Intersection counts from a byte-wise popcount on the GPU."""
    import torch

    device = torch.device("cuda")
    lookup = torch.tensor(
        [bin(value).count("1") for value in range(256)],
        dtype=torch.int16,
        device=device,
    )
    reference_t = torch.from_numpy(reference).to(device)

    results: list[np.ndarray] = []
    for start in range(0, len(library), _POPCOUNT_SLICE):
        block = torch.from_numpy(library[start : start + _POPCOUNT_SLICE]).to(device)
        common = torch.bitwise_and(block, reference_t)
        counts = lookup[common.long()].sum(dim=1, dtype=torch.int64)
        results.append(counts.cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(results) if results else np.zeros(0, dtype=np.int64)


def score_chunk(
    smiles: list[str],
    fdef_path: str,
    params: Params2D,
    reference_smiles: str,
    jobs: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    """Fingerprint a chunk on the CPU, then score it against the GPU."""
    sig_factory = build_sig_factory(fdef_path, params)
    reference_fp = reference_fingerprint(reference_smiles, sig_factory)
    reference_count = reference_fp.GetNumOnBits()
    n_bits = reference_fp.GetNumBits()

    rows: list[dict[str, Any]] = [{} for _ in smiles]
    on_bits: list[tuple[int, ...]] = []
    scored_positions: list[int] = []

    with progress_bar(len(smiles), "pharm2d fingerprints", quiet) as bar, KillableWorkerPool(
        n_workers=min(jobs, max(1, len(smiles))),
        initializer=init_worker,
        initargs=(fdef_path, params, reference_smiles),
        timeout=None,
    ) as pool:
        for result in pool.map_ordered(smiles):
            position = result.index
            if result.status != Status.OK or result.value is None:
                status = (
                    Status.PARSE_ERROR
                    if result.status == Status.OK
                    else result.status
                )
                detail = (
                    "RDKit could not parse the SMILES string"
                    if result.status == Status.OK
                    else result.detail
                )
                rows[position] = {
                    "tanimoto": NAN,
                    "dice": NAN,
                    "simpson": NAN,
                    "status": status,
                    "elapsed_s": result.elapsed_s,
                    "_detail": detail,
                }
            else:
                bits, _width = result.value
                on_bits.append(bits)
                scored_positions.append(position)
                rows[position] = {"elapsed_s": result.elapsed_s}
            bar.update(1)

    if not scored_positions:
        return rows

    reference_packed = pack_uint8([tuple(reference_fp.GetOnBits())], n_bits)
    library_packed = pack_uint8(on_bits, n_bits)
    counts = np.array([len(bits) for bits in on_bits], dtype=np.int64)

    intersections = _intersections_with_nvmolkit(
        reference_packed, library_packed, reference_count, counts
    )
    if intersections is None:
        note(
            "gpu 2D: using a GPU popcount for the similarity matrix "
            "(this fingerprint width is not accepted by the nvMolKit kernel)",
            quiet,
        )
        intersections = _intersections_with_popcount(reference_packed, library_packed)

    for offset, position in enumerate(scored_positions):
        a = reference_count
        b = int(counts[offset])
        i = int(intersections[offset])
        union = a + b - i
        total = a + b
        smaller = min(a, b)
        rows[position].update(
            {
                "tanimoto": i / union if union else 0.0,
                "dice": 2 * i / total if total else 0.0,
                "simpson": i / smaller if smaller else 0.0,
                "status": Status.OK,
                "_detail": "",
            }
        )
    return rows


def expected_scores_match_cpu(a: int, b: int, i: int) -> tuple[float, float, float]:
    """The same score formulas as the CPU backend, exposed for tests."""
    union = a + b - i
    total = a + b
    smaller = min(a, b)
    return (
        i / union if union else 0.0,
        2 * i / total if total else 0.0,
        i / smaller if smaller else 0.0,
    )
