# 3D-Vision-transformer
Self configuring and adapting vision transformer for segmentation of 3d images


```
git clone https://github.com/Anne-Andresen/3D-Vision-transformer.git
cd Former
conda env create -f environment.yml
source activate Former
pip install -e .

To train:

nnFormer_convert_decathlon_task -i ../DATASET/Former/Former_data/Task01_OAR

nnFormer_plan_and_preprocess -t 1

nnFormer_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]

e.g. DATASET_NAME_OR_ID: -t 1

e.g. [0, 1, 2, 3, 4]


```
You can reach me here or on email: aha.andresen@gmail.com or anan@clin.au.dk

TODO:
Update read me
