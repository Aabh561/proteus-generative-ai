#!/bin/bash

PDB_IDS=(
  "1L2Y" "2N7B" "1A00" "1B0B" "1C0A" "1D0A" "1E0A" "1F0A" "1G0A" "1H0A"
)
OUTPUT_DIR="data/pdb/"
mkdir -p $OUTPUT_DIR
echo "Downloading PDB files to $OUTPUT_DIR..."
for id in "${PDB_IDS[@]}"
do
  URL="https://files.rcsb.org/download/${id}.pdb"
  echo "Fetching ${id}..."
  curl -s -o "${OUTPUT_DIR}${id}.pdb" "$URL"
done
echo "✅ Download complete."