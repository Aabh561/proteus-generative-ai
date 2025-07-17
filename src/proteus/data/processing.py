from Bio.PDB import PDBParser
import numpy as np
import warnings
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Union

warnings.filterwarnings("ignore", message="Found disconnected chain(s)")

def parse_protein_structure(pdb_id: str, pdb_file: Path) -> Union[List[dict], None]:
    parser = PDBParser()
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

    atom_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    for atom in residue:
                        atom_record = {
                            "pdb_id": pdb_id,
                            "chain_id": chain.id,
                            "residue_name": residue.get_resname(),
                            "residue_seq_id": residue.get_id()[1],
                            "atom_name": atom.get_name(),
                            "x_coord": atom.get_coord()[0],
                            "y_coord": atom.get_coord()[1],
                            "z_coord": atom.get_coord()[2],
                            "bfactor": atom.get_bfactor()
                        }
                        atom_data.append(atom_record)
    return atom_data

def process_pdb_directory(pdb_dir: Path, output_file: Path):
    all_atom_data = []
    pdb_files = list(pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files to process.")
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_id = pdb_file.stem
        parsed_data = parse_protein_structure(pdb_id, pdb_file)
        if parsed_data:
            all_atom_data.extend(parsed_data)
    if not all_atom_data:
        print("No data was processed. Exiting.")
        return
    df = pd.DataFrame(all_atom_data)
    try:
        df.to_parquet(output_file, index=False)
        print(f"\n✅ Success: Processed {len(df)} atoms from {len(pdb_files)} files.")
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to Parquet file: {e}")