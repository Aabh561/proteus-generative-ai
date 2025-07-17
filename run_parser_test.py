import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.proteus.data.processing import process_pdb_directory

def main():
    print("--- Running Proteus Data Processing Pipeline ---")
    
    data_dir = project_root / "data"
    pdb_dir = data_dir / "pdb"
    processed_dir = data_dir / "processed"
    
    processed_dir.mkdir(exist_ok=True)
    
    output_file = processed_dir / "protein_structures.parquet"
    
    process_pdb_directory(pdb_dir, output_file)

if __name__ == "__main__":
    main()