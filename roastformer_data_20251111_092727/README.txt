# RoastFormer Data Package

This package contains everything needed to train RoastFormer on Google Colab.

## Contents

### Preprocessed Data
- train_profiles.json - Training roast profiles
- val_profiles.json - Validation roast profiles
- train_metadata.csv - Training metadata
- val_metadata.csv - Validation metadata
- dataset_stats.json - Dataset statistics

### Source Code
- src/dataset/preprocessed_data_loader.py - Data loading
- src/model/transformer_adapter.py - Transformer model
- train_transformer.py - Training script

## How to Use

1. Upload this zip file to Google Colab
2. Extract: `!unzip roastformer_data_*.zip`
3. Open RoastFormer_Colab_Training.ipynb
4. Run all cells

## Dataset Info

See dataset_stats.json for:
- Number of profiles
- Train/val split
- Feature dimensions
- Unique values per feature

Generated: 2025-11-11 09:27:27

