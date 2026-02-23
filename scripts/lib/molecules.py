"""
Shared molecular manipulation functions for cyclic peptide MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdCoordGen, rdDepictor

def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES string to RDKit molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def is_valid_molecule(mol: Chem.Mol) -> bool:
    """Check if molecule is valid."""
    if mol is None:
        return False
    try:
        # Basic validity checks
        return mol.GetNumAtoms() > 0
    except Exception:
        return False

def is_cyclic_peptide(mol: Chem.Mol) -> bool:
    """Check if molecule is a cyclic peptide."""
    if not is_valid_molecule(mol):
        return False

    try:
        ring_info = mol.GetRingInfo()
        return ring_info.NumRings() > 0
    except Exception:
        return False

def generate_2d_coords(mol: Chem.Mol, use_coord_gen: bool = True) -> Chem.Mol:
    """Generate 2D coordinates for a molecule."""
    if not is_valid_molecule(mol):
        return mol

    try:
        if use_coord_gen:
            rdCoordGen.AddCoords(mol)
        else:
            rdDepictor.Compute2DCoords(mol)
        return mol
    except Exception:
        return mol

def generate_3d_conformer(mol: Chem.Mol, num_conformers: int = 1) -> Chem.Mol:
    """Generate 3D conformer(s) for a molecule."""
    if not is_valid_molecule(mol):
        return mol

    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        return mol
    except Exception:
        return mol

def calculate_molecular_properties(mol: Chem.Mol) -> dict:
    """Calculate basic molecular properties."""
    if not is_valid_molecule(mol):
        return {}

    try:
        from rdkit.Chem import Descriptors

        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
            'ring_count': Descriptors.RingCount(mol),
            'hbd_count': Descriptors.NumHDonors(mol),
            'hba_count': Descriptors.NumHAcceptors(mol)
        }
        return properties
    except Exception:
        return {}

def save_molecule(mol: Chem.Mol, file_path: Union[str, Path], format: str = "sdf") -> bool:
    """Save molecule to file in specified format."""
    if not is_valid_molecule(mol):
        return False

    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "pdb":
            Chem.MolToPDBFile(mol, str(file_path))
        elif format.lower() == "sdf":
            writer = Chem.SDWriter(str(file_path))
            writer.write(mol)
            writer.close()
        elif format.lower() == "smi":
            with open(file_path, 'w') as f:
                f.write(Chem.MolToSmiles(mol))
        else:
            return False

        return True
    except Exception:
        return False

def load_molecules_from_file(file_path: Union[str, Path]) -> List[Chem.Mol]:
    """Load molecules from file (SDF, SMILES, etc.)."""
    molecules = []
    file_path = Path(file_path)

    if not file_path.exists():
        return molecules

    try:
        if file_path.suffix.lower() == '.sdf':
            supplier = Chem.SDMolSupplier(str(file_path))
            molecules = [mol for mol in supplier if mol is not None]
        elif file_path.suffix.lower() in ['.smi', '.smiles']:
            with open(file_path, 'r') as f:
                for line in f:
                    smiles = line.strip()
                    if smiles:
                        mol = parse_smiles(smiles)
                        if mol is not None:
                            molecules.append(mol)

        return molecules
    except Exception:
        return molecules