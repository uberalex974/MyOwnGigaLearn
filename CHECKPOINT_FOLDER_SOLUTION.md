# Checkpoint Folder Solution

## Problem
Checkpoints are currently saved relative to the executable location (e.g., `out/build/x64-relwithdebinfo/checkpoints/`), which gets wiped during rebuilds.

## Solution
Modify the checkpoint folder configuration in your training files:

### For Training (ExampleMain.cpp)
Add this line before creating the learner:
```cpp
// Save checkpoints to project root (preserved across rebuilds)
cfg.checkpointFolder = "C:/Giga/GigaLearnCPP/checkpoints";
```

### For Deployment (ExampleMainOptimized.cpp)  
Add this line before creating the learner:
```cpp
// Save checkpoints to project root (preserved across rebuilds)
cfg.checkpointFolder = "C:/Giga/GigaLearnCPP/checkpoints_deploy";
```

## Location
- **Training Checkpoints**: `C:\Giga\GigaLearnCPP\checkpoints\`
- **Deployment Checkpoints**: `C:\Giga\GigaLearnCPP\checkpoints_deploy\`

## Benefits
- ✅ Preserved across rebuilds
- ✅ Separate folders for training vs deployment
- ✅ Easy to locate and backup models
- ✅ No interference between training and deployment runs

## Note
The checkpoints will be organized in timestep-numbered subfolders:
```
C:/Giga/GigaLearnCPP/checkpoints/
├── 10000/
├── 20000/
├── 30000/
└── ...
```
