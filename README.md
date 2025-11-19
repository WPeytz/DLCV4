# Project 4.1 - Object Proposals for Pothole Detection

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `opencv-contrib-python` is required for Selective Search.

### 2. Sync Data from HPC

The dataset is located at `/dtu/datasets1/02516/potholes/` on DTU's HPC.

#### Option A: Work directly on HPC
SSH into the HPC and run the notebook there:
```bash
ssh <username>@login.hpc.dtu.dk
# Navigate to your project and run Jupyter
```

#### Option B: Sync data locally
```bash
# Create local data directory
mkdir -p data/potholes

# Sync from HPC (replace <username> with your DTU username)
rsync -avz --progress <username>@transfer.gbar.dtu.dk:/dtu/datasets1/02516/potholes/ ./data/potholes/

# Then update DATA_DIR in main.ipynb to "./data/potholes"
```

#### Option C: Use sshfs to mount HPC storage
```bash
# Install sshfs (macOS)
brew install macfuse sshfs

# Mount HPC storage
mkdir -p ~/hpc_mount
sshfs <username>@transfer.gbar.dtu.dk:/dtu/datasets1 ~/hpc_mount

# Use path: ~/hpc_mount/02516/potholes
```

## Project Structure

```
DLCV4/
├── data_loader.py      # Dataset class, XML parsing
├── visualization.py    # Plotting utilities
├── proposals.py        # Selective Search extraction
├── evaluation.py       # IoU, recall, label assignment
├── main.ipynb          # Main notebook
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Usage

1. Open `main.ipynb` in Jupyter
2. Update `DATA_DIR` to point to your data location
3. Run all cells to complete the 4 tasks

## Tasks Overview

1. **Data Visualization**: Load and visualize images with GT boxes
2. **Extract Proposals**: Run Selective Search on all images
3. **Evaluate Proposals**: Compute recall vs. number of proposals
4. **Label Assignment**: Assign positive/negative labels based on IoU

## Output Files

- `proposals_train.npy`: Extracted proposals for all training images
- `training_data.npy`: Proposals with labels for detector training
