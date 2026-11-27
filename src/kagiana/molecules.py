"""Turning a library cell into an RDKit molecule.

A library is someone else's data, so a cell can hold a valid SMILES string, a
typo, or nothing at all.  Each case has to be reported rather than guessed at,
and a blank cell is the trap: RDKit accepts the empty string and hands back a
molecule with no atoms, which would otherwise be scored as a perfectly valid
molecule that happens to carry no pharmacophore features.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem


def mol_from_smiles(smiles: Any) -> Any | None:
    """Parse a SMILES string, or return ``None`` if it describes no molecule."""
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol
