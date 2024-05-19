

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_new")
    output_file = join(network_training_output_dir, "summary_structseg_5folds.csv")

    folds = (0, 1, 2, 3, 4)
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "nnFormerPlans"

    overwrite_plans = {
        'nnFormerTrainerV2_2': ["nnFormerPlans", "nnFormerPlans_customClip"], # r
        'nnFormerTrainerV2_2_noMirror': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_lessMomentum_noMirror': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_2_structSeg_noMirror': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_2_structSeg': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_lessMomentum_noMirror_structSeg': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_FabiansResUNet_structSet_NoMirror_leakyDecoder': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_FabiansResUNet_structSet_NoMirror': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r
        'nnFormerTrainerV2_FabiansResUNet_structSet': ["nnFormerPlans", "nnFormerPlans_customClip"],  # r

    }

    trainers = ['nnFormerTrainer'] + [
        'nnFormerTrainerV2_2',
        'nnFormerTrainerV2_lessMomentum_noMirror',
        'nnFormerTrainerV2_2_noMirror',
        'nnFormerTrainerV2_2_structSeg_noMirror',
        'nnFormerTrainerV2_2_structSeg',
        'nnFormerTrainerV2_lessMomentum_noMirror_structSeg',
        'nnFormerTrainerV2_FabiansResUNet_structSet_NoMirror_leakyDecoder',
        'nnFormerTrainerV2_FabiansResUNet_structSet_NoMirror',
        'nnFormerTrainerV2_FabiansResUNet_structSet',
    ]

    datasets = \
        {"Task049_StructSeg2019_Task1_HaN_OAR": ("3d_fullres",  "3d_lowres", "2d"),
        "Task050_StructSeg2019_Task2_Naso_GTV": ("3d_fullres", "3d_lowres", "2d"),
        "Task051_StructSeg2019_Task3_Thoracic_OAR": ("3d_fullres", "3d_lowres", "2d"),
        "Task052_StructSeg2019_Task4_Lung_GTV": ("3d_fullres", "3d_lowres", "2d"),
}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                if len(c) > 3:
                    n = c[3]
                else:
                    n = "2"
                s1 = s + "_" + n
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                f.write("\n")

                if all_present:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
