#!/usr/bin/python3
# -*- coding: utf-8 -*-


import datetime
import os
import random

import criteria
import dataset
import decision_tree

from sklearn.model_selection import StratifiedKFold#, KFold
import numpy as np


NUM_PROCS = 3
RANDOM_SEEDS = [65537, 986112772, 580170418, 897083807, 1286664107, 899169460, 1728505703,
                423232363, 1576030107, 1102706565, 756372267, 1041481669, 500571641, 1196189230,
                49471178, 827006267, 1581871235, 1249719834, 1281093615, 603059048, 1217122226,
                1784259850, 1636348000, 169498007, 1644610044, 1001000160, 884435702, 759171700,
                1729486164, 735355316, 590274769, 1685315218, 1811339189, 1436392076, 966320783,
                332035403, 1477247432, 1277551165, 395864655, 1334785552, 1863196977, 420054870,
                691025606, 1670255402, 535409696, 1556940403, 1036018082, 1120042937, 2128605734,
                1359372989, 335126928, 2109877295, 2070420066, 276135419, 1966874990, 1599483352,
                509177296, 8165980, 95787270, 343114090, 1938652731, 487748814, 1904611130,
                828040489, 620410008, 1013438160, 1422307526, 140262428, 1885459289, 116052729,
                1232834979, 708634310, 1761972120, 1247444947, 1585555404, 1859131028, 455754736,
                286190308, 1082412114, 2050198702, 998783919, 1496754253, 1371389911, 1314822048,
                1157568092, 332882253, 1647703149, 2011051574, 1222161240, 1358795771, 927702031,
                760815609, 504204359, 1424661575, 1228406087, 1971630940, 1758874112, 1403628276,
                643422904, 1196432617]
NUM_RANDOM_SEEDS_TO_USE = 20


def rank_attributes(dataset_name, train_dataset, criterion, output_file_descriptor, seed_num,
                    output_split_char, num_folds=10, use_stop_conditions=False):

    sample_indices_and_classes = list(enumerate(train_dataset.sample_class))
    seed = RANDOM_SEEDS[seed_num - 1]

    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(sample_indices_and_classes)
    shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)

    for (fold_number,
         (training_randomized_indices,
          validation_randomized_indices)) in enumerate(
              StratifiedKFold(n_splits=num_folds).split(shuffled_sample_indices,
                                                        shuffled_sample_classes)):

        training_samples_indices = [shuffled_sample_indices[index]
                                    for index in training_randomized_indices]
        validation_sample_indices = [shuffled_sample_indices[index]
                                     for index in validation_randomized_indices]

        print('Fold #{}'.format(fold_number + 1))
        node = decision_tree.TreeNode(train_dataset,
                                      training_samples_indices,
                                      train_dataset.valid_nominal_attribute[:],
                                      max_depth_remaining=1,
                                      min_samples_per_node=1,
                                      use_stop_conditions=use_stop_conditions)
        if criterion.name == 'Max Cut':
            (ranked_attributes_p_val,
             ranked_attributes_gain) = criterion.evaluate_all_attributes(node)
            save_fold_info(dataset_name, train_dataset, node, fold_number, criterion.name,
                           ranked_attributes_p_val, training_samples_indices,
                           validation_sample_indices, output_split_char,
                           output_file_descriptor, seed_num)
            save_fold_info(dataset_name, train_dataset, node, fold_number, 'Max Cut Naive',
                           ranked_attributes_gain, training_samples_indices,
                           validation_sample_indices, output_split_char,
                           output_file_descriptor, seed_num)
        else:
            ranked_attributes = criterion.evaluate_all_attributes(node)
            save_fold_info(dataset_name, train_dataset, node, fold_number, criterion.name,
                           ranked_attributes, training_samples_indices,
                           validation_sample_indices, output_split_char,
                           output_file_descriptor, seed_num)


def calculate_value_to_split_index(values_in_each_split):
    value_to_split_index = {}
    for split_index, values in enumerate(values_in_each_split):
        for value in values:
            value_to_split_index[value] = split_index
    return value_to_split_index


def calculate_classification_for_each_split(train_dataset, training_samples_indices, attrib_index,
                                            value_to_split_index, num_splits):
    split_class_count = [[0] * train_dataset.num_classes for _ in range(num_splits)]
    for sample_index in training_samples_indices:
        sample_value = train_dataset.samples[sample_index][attrib_index]
        try:
            sample_split = value_to_split_index[sample_value]
        except KeyError:
            continue
        sample_class = train_dataset.sample_class[sample_index]
        split_class_count[sample_split][sample_class] += 1
    split_labels = [split_class_count[split_index].index(max(split_class_count[split_index]))
                    for split_index in range(num_splits)]
    return split_labels


def is_trivial(split_labels):
    label = split_labels[0]
    for curr_label in split_labels:
        if curr_label != label:
            return False
    return True


def calculate_accuracy(train_dataset, validation_sample_indices, attrib_index, value_to_split_index,
                       split_labels, most_common_int_class):
    num_correct_classifications = 0
    num_correct_classifications_wo_unkown = 0
    num_unkown = 0
    for sample_index in validation_sample_indices:
        sample_value = train_dataset.samples[sample_index][attrib_index]
        try:
            sample_split = value_to_split_index[sample_value]
        except KeyError:
            num_unkown += 1
            if most_common_int_class == train_dataset.sample_class[sample_index]:
                num_correct_classifications += 1
            continue
        sample_classified_label = split_labels[sample_split]
        if sample_classified_label == train_dataset.sample_class[sample_index]:
            num_correct_classifications += 1
            num_correct_classifications_wo_unkown += 1
    return num_correct_classifications, num_correct_classifications_wo_unkown, num_unkown


def calculate_largest_split(train_dataset, training_samples_indices, attrib_index,
                            value_to_split_index, num_splits):
    split_count = [0] * num_splits
    max_split_size = 0
    for sample_index in training_samples_indices:
        sample_value = train_dataset.samples[sample_index][attrib_index]
        try:
            sample_split = value_to_split_index[sample_value]
        except KeyError:
            continue
        split_count[sample_split] += 1
        if split_count[sample_split] > max_split_size:
            max_split_size = split_count[sample_split]
    return max_split_size


def get_percentage_in_largest_class(root_node, validation_samples_indices):
    most_common_class_train = root_node.most_common_int_class
    count = 0
    for sample_index in validation_samples_indices:
        if most_common_class_train == root_node.dataset.sample_class[sample_index]:
            count += 1
    return 100.0 * count / len(validation_samples_indices)


def save_fold_info(dataset_name, train_dataset, node, fold_number, criterion_name,
                   ranked_attributes, training_samples_indices, validation_sample_indices,
                   output_split_char, output_file_descriptor, seed_num):
    line_start = [dataset_name,
                  str(sum(train_dataset.valid_nominal_attribute)),
                  str(fold_number + 1),
                  str(get_percentage_in_largest_class(node, validation_sample_indices)),
                  criterion_name]
    for attribute in ranked_attributes:
        # ranked_attributes has form
        # [(attrib_index, curr_criteria_gain, [values_in_each_split], p_value, time_taken,
        #   should_accept, num_tests_needed, pref_position)]
        # or, in case of Fast Max Cut Chi Square With P Value M C:
        # [(attrib_index, curr_criteria_gain, [values_in_each_split], p_value, time_taken,
        #   should_accept, num_tests_needed, num_monte_carlo_done, pref_position)]
        attrib_index = attribute[0]
        values_in_each_split = attribute[2]
        num_distinct_values = len(set.union(*values_in_each_split))
        value_to_split_index = calculate_value_to_split_index(values_in_each_split)
        split_labels = calculate_classification_for_each_split(train_dataset,
                                                               training_samples_indices,
                                                               attrib_index,
                                                               value_to_split_index,
                                                               len(values_in_each_split))
        (num_correct_classifications,
         num_correct_classifications_wo_unkown,
         num_unkown) = calculate_accuracy(train_dataset,
                                          validation_sample_indices,
                                          attrib_index,
                                          value_to_split_index,
                                          split_labels,
                                          node.most_common_int_class)
        num_samples = len(validation_sample_indices)
        accuracy = 100.0 * num_correct_classifications/num_samples
        if num_samples == num_unkown:
            accuracy_wo_unkown = None
        else:
            accuracy_wo_unkown = (
                100.0 * num_correct_classifications_wo_unkown / (num_samples - num_unkown))
        largest_split_size = calculate_largest_split(train_dataset,
                                                     training_samples_indices,
                                                     attrib_index,
                                                     value_to_split_index,
                                                     len(values_in_each_split))
        if len(attribute) == 9:
            line_list = (line_start +
                         [train_dataset.attrib_names[attrib_index],
                          str(num_distinct_values),
                          str(attribute[1]),
                          str(attribute[3]),
                          str(accuracy),
                          str(num_samples),
                          str(accuracy_wo_unkown),
                          str(num_samples - num_unkown),
                          str(100.0 * largest_split_size / len(training_samples_indices)),
                          str(is_trivial(split_labels)),
                          str(attribute[4]),
                          str(attribute[-1] + 1),
                          str(attribute[5]),
                          str(attribute[6]),
                          str(datetime.datetime.now()),
                          str(attribute[7]),
                          str(seed_num + 1),
                          str(sum(train_dataset.valid_numeric_attribute)),
                          str(sum(train_dataset.valid_nominal_attribute)
                              + sum(train_dataset.valid_nominal_attribute)),
                          str(train_dataset.valid_numeric_attribute[attrib_index])])
        else:
            line_list = (line_start +
                         [train_dataset.attrib_names[attrib_index],
                          str(num_distinct_values),
                          str(attribute[1]),
                          str(attribute[3]),
                          str(accuracy),
                          str(num_samples),
                          str(accuracy_wo_unkown),
                          str(num_samples - num_unkown),
                          str(100.0 * largest_split_size / len(training_samples_indices)),
                          str(is_trivial(split_labels)),
                          str(attribute[4]),
                          str(attribute[-1] + 1),
                          str(attribute[5]),
                          str(attribute[6]),
                          str(datetime.datetime.now()),
                          str(None),
                          str(seed_num + 1),
                          str(sum(train_dataset.valid_numeric_attribute)),
                          str(sum(train_dataset.valid_nominal_attribute)
                              + sum(train_dataset.valid_nominal_attribute)),
                          str(train_dataset.valid_numeric_attribute[attrib_index])])
        print(output_split_char.join(line_list), file=output_file_descriptor)


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, output_csv_filepath, seed_num, output_split_char=',',
         load_numeric=False):
    with open(output_csv_filepath, 'a') as fout:
        fields_list = ['Dataset', 'Number of Valid Nominal Attributes', 'Fold #',
                       'Percentage in Largest Class (Validation Set per Fold)', 'Criterion',
                       'Attrib Name', 'Number of Distinct Values', 'Criterion Value', 'Max p-value',
                       'Accuracy', 'Number of Samples', 'Accuracy w/o missing values',
                       'Number of Samples without missing values', 'Largest split size (in %)',
                       'Is Trivial', 'Time taken [s]', 'Position in Favorite List', 'Should Accept',
                       'Number of Tests Needed', 'Date Time',
                       'Number of Monte Carlo Tests Needed in Weight Matrix (max = n*(n-1)/2)',
                       'Seed Index', 'Number of Valid Numeric Attributes',
                       'Total Number of Valid Attributes', 'Is Numeric']

        print(output_split_char.join(fields_list), file=fout)
        fout.flush()

        for dataset_number, filepath in enumerate(datasets_filepaths):
            if not os.path.exists(filepath) or not os.path.isfile(filepath):
                continue
            train_dataset = dataset.Dataset(filepath,
                                            key_attrib_indices[dataset_number],
                                            class_attrib_indices[dataset_number],
                                            split_chars[dataset_number],
                                            missing_value_strings[dataset_number],
                                            load_numeric=load_numeric)
            # print('-'*100)
            # print('Gini Index')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.GiniIndex(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Gini Twoing')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.GiniTwoing(),
            #                 fout,
            #                 seed_num,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Gini Twoing Monte Carlo')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.GiniTwoingMonteCarlo(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Twoing')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Twoing(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('ORT')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Ort(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('MPI')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Mpi(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExact(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Residue')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactResidue(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Chi Square')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactChiSquare(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Chi Square Heuristic')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactChiSquareHeuristic(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Gain Ratio')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.GainRatio(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCut(),
            #                 fout,
            #                 output_split_char)
            print('-'*100)
            print('Max Cut Naive')
            print()
            rank_attributes(dataset_names[dataset_number],
                            train_dataset,
                            criteria.MaxCutNaive(),
                            fout,
                            seed_num,
                            output_split_char)
            fout.flush()
            # print('-'*100)
            # print('Max Cut Naive With Local Search')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveWithLocalSearch(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Naive')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutNaive(),
            #                 fout,
            #                 seed_num,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Max Cut Naive Residue')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveResidue(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Naive Chi Square')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquare(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Naive Chi Square With Local Search')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquareWithLocalSearch(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Fast Max Cut Chi Square')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutChiSquare(),
            #                 fout,
            # #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Naive Chi Square Normalized With Local Search')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquareNormalizedWithLocalSearch(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            print('-'*100)
            print('Max Cut Naive Chi Square Normalized')
            print()
            rank_attributes(dataset_names[dataset_number],
                            train_dataset,
                            criteria.MaxCutNaiveChiSquareNormalized(),
                            fout,
                            seed_num,
                            output_split_char)
            fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Chi Square Normalized')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutChiSquareNormalized(),
            #                 fout,
            #                 seed_num,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Chi Square Normalized P Value')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutChiSquareNormalizedPValue(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Chi Square Normalized P Value M C')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutChiSquareNormalizedPValueMC(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Max Cut Naive Chi Square Heuristic')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquareHeuristic(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Monte Carlo')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutMonteCarlo(),
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Monte Carlo Residue')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutMonteCarloResidue(),
            #                 fout,
            #                 output_split_char)
            # fout.flush()


if __name__ == '__main__':
    DATASET_NAMES = []
    DATASETS_FILEPATHS = []
    KEY_ATTRIB_INDICES = []
    CLASS_ATTRIB_INDICES = []
    SPLIT_CHARS = []
    MISSING_VALUE_STRINGS = []

    # # TRAINING SET

    # # Mushroom
    # DATASET_NAMES.append('mushroom')
    # DATASETS_FILEPATHS.append(os.path.join('.', 'datasets', 'mushroom', 'training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(0)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Adult census income
    # DATASET_NAMES.append('adult census income')
    # DATASETS_FILEPATHS.append(
    #     os.path.join('.', 'datasets', 'adult census income', 'training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # KDD98 Target D

    # # 2 classes:
    # KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
    #     '.', 'datasets', 'kdd_multiclass')

    # DATASET_NAMES.append('kdd98_multiclass_2')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_2.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 3 classes:
    # DATASET_NAMES.append('kdd98_multiclass_3')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_3.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 4 classes:
    # DATASET_NAMES.append('kdd98_multiclass_4')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_4.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 5 classes:
    # DATASET_NAMES.append('kdd98_multiclass_5')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_5.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 9 classes:
    # DATASET_NAMES.append('kdd98_multiclass_9')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_9.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # # Connect4:
    # # CONNECT_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Connect4-3Aggregat')
    # # DATASET_NAMES.append('Connect4-3Aggregat')
    # # DATASETS_FILEPATHS.append(os.path.join(CONNECT_TRAIN_DB_FOLDER,
    # #                                        'Connect4-3Aggregat_training.csv'))
    # # KEY_ATTRIB_INDICES.append(None)
    # # CLASS_ATTRIB_INDICES.append(-1)
    # # SPLIT_CHARS.append(',')
    # # MISSING_VALUE_STRINGS.append('?')

    # # Nursery-Agregate:
    # NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Nursery-Agregate')
    # DATASET_NAMES.append('Nursery-Agregate_with_original')
    # DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
    #                                        'Nursery_original_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # CovTypeReduced:
    # COVTYPE_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'CovTypeReduced')
    # DATASET_NAMES.append('CovTypeReduced_with_original')
    # DATASETS_FILEPATHS.append(os.path.join(COVTYPE_TRAIN_DB_FOLDER,
    #                                        'CovType_original_no_num_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Car
    # DATASET_NAMES.append('Car_with_original')
    # DATASETS_FILEPATHS.append(
    #     os.path.join('.', 'datasets', 'Car', 'Car_training_orig_attributes.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Contraceptive
    # DATASET_NAMES.append('Contraceptive_with_original')
    # DATASETS_FILEPATHS.append(
    #     os.path.join('.', 'datasets', 'Contraceptive', 'cmc_with_aggreg_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    ###############################################################################################
    #TEST SET

    # Mushroom
    DATASET_NAMES.append('mushroom_full_test')
    DATASETS_FILEPATHS.append(os.path.join('.', 'datasets', 'mushroom', 'full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(0)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')


    # Adult census income
    DATASET_NAMES.append('adult census income full_test')
    DATASETS_FILEPATHS.append(
        os.path.join('.', 'datasets', 'adult census income', 'full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # KDD98 Target D

    # 2 classes:
    KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
        '.', 'datasets', 'kdd_multiclass')

    DATASET_NAMES.append('kdd98_multiclass_2_full_test')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_from_LRN_multiclass_2_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 3 classes:
    DATASET_NAMES.append('kdd98_multiclass_3_full_test')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_from_LRN_multiclass_3_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 4 classes:
    DATASET_NAMES.append('kdd98_multiclass_4_full_test')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_from_LRN_multiclass_4_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 5 classes:
    DATASET_NAMES.append('kdd98_multiclass_5_full_test')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_from_LRN_multiclass_5_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 9 classes:
    DATASET_NAMES.append('kdd98_multiclass_9_full_test')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_from_LRN_multiclass_9_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # # Connect4:
    # CONNECT_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Connect4-3Aggregat')
    # DATASET_NAMES.append('Connect4-3Aggregat')
    # DATASETS_FILEPATHS.append(os.path.join(CONNECT_TRAIN_DB_FOLDER,
    #                                        'Connect4-3Aggregat_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # Nursery-Agregate:
    NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Nursery-Agregate')
    DATASET_NAMES.append('Nursery-Agregate_with_original_full_test')
    DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
                                           'Nursery_original_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # CovTypeReduced:
    COVTYPE_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'CovTypeReduced')
    DATASET_NAMES.append('CovTypeReduced_with_original_full_test')
    DATASETS_FILEPATHS.append(os.path.join(COVTYPE_TRAIN_DB_FOLDER,
                                           'CovType_original_no_num_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Car
    DATASET_NAMES.append('Car_with_original_full_test')
    DATASETS_FILEPATHS.append(
        os.path.join('.', 'datasets', 'Car', 'Car_orig_attributes_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Contraceptive
    DATASET_NAMES.append('Contraceptive_with_original_full_test')
    DATASETS_FILEPATHS.append(
        os.path.join('.', 'datasets', 'Contraceptive', 'cmc_with_aggreg_full_test.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    OUTPUT_CSV_FILEPATH = os.path.join(
        '.',
        'outputs from datasets',
        'rank_TEST_7_max_cut_naive_and_chi_square_normalized_with_orig_attrib'
        '_seeds_1_mod_3_FULL_TEST_SET.csv')

    for seed_num in range(0, NUM_RANDOM_SEEDS_TO_USE, 3):
        main(DATASET_NAMES, DATASETS_FILEPATHS, KEY_ATTRIB_INDICES, CLASS_ATTRIB_INDICES,
             SPLIT_CHARS, MISSING_VALUE_STRINGS, OUTPUT_CSV_FILEPATH, seed_num + 1,
             load_numeric=False)
