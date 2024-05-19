
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "Former"
default_plans_identifier = "FormerPlansv2.1"
default_data_identifier = 'FormerData_plans_v2.1'
default_trainer = "FormerTrainerV2"
default_cascade_trainer = "FormerTrainerV2CascadeFullRes"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

#base = os.environ['nnFormer_raw_data_base'] if "nnFormer_raw_data_base" in os.environ.keys() else None
#preprocessing_output_dir = os.environ['nnFormer_preprocessed'] if "nnFormer_preprocessed" in os.environ.keys() else None
#network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None
base = '/data/'
preprocessing_output_dir = '/reprocessed/'
network_training_output_dir_base = 'trained_models/'
if base is not None:
  raw_data = join(base, "data")
    cropped_data = join(base, "cropped_data")
    maybe_mkdir_p(raw_data)
    maybe_mkdir_p(cropped_data)
else:
    print("")
    nnFormer_cropped_data = nnFormer_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("set up paths")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined "
          "inference. This is not intended behavior)
    network_training_output_dir = None
