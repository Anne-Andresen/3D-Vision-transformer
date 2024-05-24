# 3D-Vision-Transformer

## Self Configuring and Adapting Vision Transformer for Segmentation of 3D Images

This repository contains the implementation of a self-configuring and adapting vision transformer designed for the segmentation of 3D images.

## Getting Started

### Prerequisites

- Anaconda (for managing the environment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Anne-Andresen/3D-Vision-transformer.git
   cd Former
   ```
2. Create and activate the conda environment:
   ``` bash
   conda env create -f environment.yml
   source activate Former
   ```
3. Create and activate the conda environment:

   ``` bash
   pip install -e .
   ```
### Training

To train the model, follow these steps:

1. Convert the dataset:
   ```
   nnFormer_convert_decathlon_task -i ../DATASET/Former/Former_data/Task01_OAR

   ```
2. Plan and preprocess the dataset:
   ``` bash
   nnFormer_plan_and_preprocess -t 1

   ```
3. Train the model:
   ``` bash
   nnFormer_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]

   ```
- Example for DATASET_NAME_OR_ID:
   ``` bash
   nnFormer_train -t 1
   ```
- Example for FOLD values: [0, 1, 2, 3, 4]

 ### Contact

For any questions or support, you can reach me via email at:

aha.andresen@gmail.com
anan@clin.au.dk
### TODO
Update README


Note: This README file is a work in progress and will be updated as the project evolves.
  
