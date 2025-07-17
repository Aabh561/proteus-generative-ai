import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from models.diffusion import ProteusTransformer

# --- 1. Configuration ---
DATA_PATH = Path("../../data/processed/protein_structures.parquet")
MODEL_SAVE_PATH = Path("../../results/trained_models/proteus_transformer_v1.pth")
# This is the corrected line:
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- 2. Data Preparation ---
print("Loading and preparing data...")
df = pd.read_parquet(DATA_PATH)
backbone_df = df[df['atom_name'] == 'CA'].copy()

SEQUENCE_LENGTH = 128
sequences = backbone_df.groupby('pdb_id')['location_token'].apply(list).tolist()

X, y = [], []
for seq in sequences:
    if len(seq) > SEQUENCE_LENGTH:
        for i in range(len(seq) - SEQUENCE_LENGTH):
            X.append(seq[i:i + SEQUENCE_LENGTH])
            y.append(seq[i + 1:i + SEQUENCE_LENGTH + 1])

X = np.array(X)
y = np.array(y)

# --- 3. PyTorch DataLoader ---
BATCH_SIZE = 32
X_tensor = torch.from_numpy(X).long()
y_tensor = torch.from_numpy(y).long()
train_data = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. Model & Training Setup ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEAD = 8
D_HID = 512
N_LAYERS = 4
DROPOUT = 0.1
model = ProteusTransformer(VOCAB_SIZE, D_MODEL, N_HEAD, D_HID, N_LAYERS, DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10

# --- 5. Training Loop ---
print("--- Starting Model Training ---")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

# --- 6. Save the Final Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")