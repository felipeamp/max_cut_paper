#!/usr/bin/python3
# -*- coding: utf-8 -*-


import datetime
import os
import random
# import sys
import timeit

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


def rank_attributes(dataset_name, train_dataset, criterion, max_height, output_file_descriptor,
                    seed_num, output_split_char, num_folds=10, use_stop_conditions=False):
    sample_indices_and_classes = list(enumerate(train_dataset.sample_class))
    seed = RANDOM_SEEDS[seed_num - 1]
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(sample_indices_and_classes)
    shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)
    original_valid_nominal_attributes = train_dataset.valid_nominal_attribute[:]
    original_valid_numeric_attributes = train_dataset.valid_numeric_attribute[:]

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
        tree = decision_tree.DecisionTree(criterion)
        num_valid_nominal_attribs = sum(original_valid_nominal_attributes)
        num_valid_numeric_attribs = sum(original_valid_numeric_attributes)
        num_valid_attribs = num_valid_nominal_attribs + num_valid_numeric_attribs
        num_attributes = len(original_valid_nominal_attributes)

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(original_valid_nominal_attributes,
                                                         original_valid_numeric_attributes)):
            if not is_valid_nominal_attrib and not is_valid_numeric_attrib:
                continue
            if not has_more_than_one_value(train_dataset,
                                           training_samples_indices,
                                           attrib_index):
                print("Attribute {} ({}) is valid but has"
                      " less than two different values.".format(
                          attrib_index,
                          train_dataset.attrib_names[attrib_index]))
                continue
            train_dataset.valid_nominal_attribute = [False] * num_attributes
            train_dataset.valid_nominal_attribute[attrib_index] = is_valid_nominal_attrib
            train_dataset.valid_numeric_attribute = [False] * num_attributes
            train_dataset.valid_numeric_attribute[attrib_index] = is_valid_numeric_attrib

            start_time = timeit.default_timer()
            ((_,
              num_correct_classifications,
              num_correct_classifications_wo_unkown,
              _,
              _,
              _,
              num_unkown,
              _),
             _) = tree.train_and_self_validate(train_dataset,
                                               training_samples_indices,
                                               validation_sample_indices,
                                               max_depth=max_height,
                                               min_samples_per_node=1,
                                               use_stop_conditions=use_stop_conditions)
            time_taken = timeit.default_timer() - start_time
            save_info(dataset_name, train_dataset, tree, fold_number, criterion.name,
                      num_valid_nominal_attribs, attrib_index,
                      training_samples_indices, validation_sample_indices, time_taken,
                      num_correct_classifications, num_correct_classifications_wo_unkown,
                      num_unkown, output_split_char, output_file_descriptor, seed_num,
                      num_valid_numeric_attribs, num_valid_attribs)

        train_dataset.valid_nominal_attribute = original_valid_nominal_attributes
        train_dataset.valid_numeric_attribute = original_valid_numeric_attributes


def has_more_than_one_value(dataset, valid_indices, attrib_index):
    values_seen = set()
    for sample_index in valid_indices:
        sample_value = dataset.samples[sample_index][attrib_index]
        if sample_value not in values_seen:
            values_seen.add(sample_value)
            if len(values_seen) >= 2:
                return True
    return False


def get_tree_details(tree_node):
    if tree_node.is_leaf:
        return (0,
                1,
                tree_node.num_valid_samples,
                tree_node.num_valid_samples,
                True,
                set((tree_node.most_common_int_class,)))

    height = 1
    num_leaves = 0
    smallest_num_samples_in_leaf = tree_node.num_valid_samples
    largest_num_samples_in_leaf = 0
    is_trivial = True
    seen_leaf_classes = set()
    for child_node in tree_node.nodes:
        (child_height,
         child_num_leaves,
         child_smallest_num_samples_in_leaf,
         child_largest_num_samples_in_leaf,
         child_is_trivial,
         child_seen_leaf_classes) = get_tree_details(child_node)

        if child_height + 1 > height:
            height = child_height + 1
        num_leaves += child_num_leaves
        if child_smallest_num_samples_in_leaf < smallest_num_samples_in_leaf:
            smallest_num_samples_in_leaf = child_smallest_num_samples_in_leaf
        if child_largest_num_samples_in_leaf > largest_num_samples_in_leaf:
            largest_num_samples_in_leaf = child_largest_num_samples_in_leaf
        if not child_is_trivial:
            is_trivial = False
        seen_leaf_classes |= child_seen_leaf_classes

    if is_trivial and len(seen_leaf_classes) > 1:
        is_trivial = False
    return (height, num_leaves, smallest_num_samples_in_leaf, largest_num_samples_in_leaf,
            is_trivial, seen_leaf_classes)


def get_num_distinct_values(train_dataset, attrib_index, training_samples_indices):
    distinct_values = set()
    num_distinct = 0
    for sample_index in training_samples_indices:
        sample_value = train_dataset.samples[sample_index][attrib_index]
        if sample_value not in distinct_values:
            distinct_values.add(sample_value)
            num_distinct += 1
    return num_distinct


def get_percentage_in_largest_class(root_node, validation_samples_indices):
    most_common_class_train = root_node.most_common_int_class
    count = 0
    for sample_index in validation_samples_indices:
        if most_common_class_train == root_node.dataset.sample_class[sample_index]:
            count += 1
    return 100.0 * count / len(validation_samples_indices)


def get_rarest_value_count(train_dataset, attrib_index, samples_indices):
    values_and_counters = {}
    for sample_index in samples_indices:
        sample_value = train_dataset.samples[sample_index][attrib_index]
        if sample_value in values_and_counters:
            values_and_counters[sample_value] += 1
        else:
            values_and_counters[sample_value] = 1
    return min(values_and_counters.values())


# def calculate_accuracy_one_level(root_node, validation_sample_indices, attrib_index):
#     num_correct_classifications = 0
#     num_correct_classifications_wo_unkown = 0
#     num_unkown = 0
#     node_split = root_node.node_split

#     mid_point = node_split.mid_point
#     if mid_point is None:
#         values_to_split = node_split.values_to_split

#     child_nodes = root_node.nodes
#     for sample_index in validation_sample_indices:
#         sample_value = root_node.dataset.samples[sample_index][attrib_index]
#         sample_class = root_node.dataset.sample_class[sample_index]
#         if mid_point is not None:
#             if sample_value is None: # Missing Numeric Value
#                 num_unkown += 1
#                 if root_node.most_common_int_class == sample_class:
#                     num_correct_classifications += 1
#                 continue
#             if sample_value <= mid_point:
#                 sample_split = 0 # left split
#             else:
#                 sample_split = 1 # right split
#         else:
#             try:
#                 sample_split = values_to_split[sample_value]
#             except KeyError:
#                 num_unkown += 1
#                 if root_node.most_common_int_class == sample_class:
#                     num_correct_classifications += 1
#                 continue
#         try:
#             child_node = child_nodes[sample_split]
#         except IndexError:
#             print("Split index ({}) does not exist in this Tree node."
#                   " Split indices only exist up to index {}.".format(
#                       sample_split, len(child_nodes) - 1))
#             sys.exit(1)
#         sample_classified_label = child_node.most_common_int_class
#         if sample_classified_label == sample_class:
#             num_correct_classifications += 1
#             num_correct_classifications_wo_unkown += 1
#     return num_correct_classifications, num_correct_classifications_wo_unkown, num_unkown


def save_info(dataset_name, train_dataset, tree, fold_number, criterion_name,
              num_valid_nominal_attribs, attrib_index, training_samples_indices,
              validation_sample_indices, time_taken, num_correct_classifications,
              num_correct_classifications_wo_unkown, num_unkown, output_split_char,
              output_file_descriptor, seed_num, num_valid_numeric_attribs, num_valid_attribs):

    (height,
     num_leaves,
     smallest_num_samples_in_leaf,
     largest_num_samples_in_leaf,
     is_trivial,
     _) = get_tree_details(tree.get_root_node())
    num_distinct_values = get_num_distinct_values(train_dataset,
                                                  attrib_index,
                                                  training_samples_indices)
    num_samples = len(validation_sample_indices)
    accuracy = 100.0 * num_correct_classifications / num_samples
    accuracy_wo_unkown = 100.0 * num_correct_classifications_wo_unkown / (num_samples - num_unkown)

    line_list = [dataset_name,
                 str(num_valid_nominal_attribs),
                 str(fold_number + 1),
                 str(get_percentage_in_largest_class(tree.get_root_node(),
                                                     validation_sample_indices)),
                 criterion_name,
                 train_dataset.attrib_names[attrib_index],
                 str(num_distinct_values),
                 str(None),
                 str(None),
                 str(accuracy),
                 str(num_samples),
                 str(accuracy_wo_unkown),
                 str(num_samples - num_unkown),
                 str(100.0 * largest_num_samples_in_leaf / len(training_samples_indices)),
                 str(is_trivial),
                 str(time_taken),
                 str(None),
                 str(height > 0),
                 str(None),
                 str(height),
                 str(num_leaves),
                 str(100.0 * smallest_num_samples_in_leaf / len(training_samples_indices)),
                 str(None),
                 str(datetime.datetime.now()),
                 str(seed_num + 1),
                 str(num_valid_numeric_attribs),
                 str(num_valid_attribs),
                 str(train_dataset.valid_numeric_attribute[attrib_index])]
    if train_dataset.valid_nominal_attribute[attrib_index]:
        rarest_value_in_root = get_rarest_value_count(train_dataset,
                                                      attrib_index,
                                                      training_samples_indices)
        line_list[-5] = str(100.0 * rarest_value_in_root / len(training_samples_indices))
    print(output_split_char.join(line_list), file=output_file_descriptor)


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, max_height, output_csv_filepath, seed_num, output_split_char=',',
         load_numeric=False, use_stop_conditions=False):
    with open(output_csv_filepath, 'a') as fout:
        fields_list = ['Dataset', 'Number of Valid Attributes', 'Fold #',
                       'Percentage in Largest Class (Validation Set per Fold)', 'Criterion',
                       'Attrib Name', 'Number of Distinct Values', 'Criterion Value', 'Max p-value',
                       'Accuracy', 'Number of Samples', 'Accuracy w/o missing values',
                       'Number of Samples without missing values', 'Largest leaf size (in %)',
                       'Is Trivial', 'Time taken [s]', 'Position in Favorite List', 'Should Accept',
                       'Number of Tests Needed', 'Tree Height', 'Number of Leaves',
                       'Smallest leaf size (in %)', 'Rarest value at Tree Root (in %)',
                       'Date Time', 'Seed Index', 'Number of Valid Numeric Attributes',
                       'Total Number of Valid Attributes',
                       'Is Numeric']

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
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            print('Gini Twoing')
            print()
            rank_attributes(dataset_names[dataset_number],
                            train_dataset,
                            criteria.GiniTwoing(),
                            max_height,
                            fout,
                            seed_num,
                            output_split_char,
                            use_stop_conditions=use_stop_conditions)
            fout.flush()
            # print('-'*100)
            # print('Twoing')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Twoing(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('ORT')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Ort(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('MPI')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.Mpi(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExact(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Residue')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactResidue(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Chi Square')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactChiSquare(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Exact Chi Square Heuristic')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutExactChiSquareHeuristic(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Gain Ratio')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.GainRatio(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            print('-'*100)
            print('Max Cut Naive')
            print()
            rank_attributes(dataset_names[dataset_number],
                            train_dataset,
                            criteria.MaxCutNaive(),
                            max_height,
                            fout,
                            seed_num,
                            output_split_char)
            fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Naive')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutNaive(),
            #                 max_height,
            #                 fout,
            #                 seed_num,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Max Cut Naive Chi Square')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquare(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # print('-'*100)
            # print('Max Cut Naive Chi Square Heuristic')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutNaiveChiSquareHeuristic(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            print('-'*100)
            print('Max Cut Naive Chi Square Normalized')
            print()
            rank_attributes(dataset_names[dataset_number],
                            train_dataset,
                            criteria.MaxCutNaiveChiSquareNormalized(),
                            max_height,
                            fout,
                            seed_num,
                            output_split_char,
                            use_stop_conditions=use_stop_conditions)
            fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Chi Square Normalized')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.FastMaxCutChiSquareNormalized(),
            #                 max_height,
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
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # fout.flush()
            # print('-'*100)
            # print('Max Cut Monte Carlo')
            # print()
            # rank_attributes(dataset_names[dataset_number],
            #                 train_dataset,
            #                 criteria.MaxCutMonteCarlo(),
            #                 max_height,
            #                 fout,
            #                 output_split_char)
            # fout.flush()


if __name__ == '__main__':
    MAX_HEIGHTS = [2]

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

    # # # 2 classes:
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
    # KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
    #     '.', 'datasets', 'kdd_multiclass')

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

    for max_height in MAX_HEIGHTS:
        OUTPUT_CSV_FILEPATH = os.path.join(
            '.',
            'outputs from datasets',
            'rank_TEST_8_max_cut_naive_and_chi_square_normalized_with_orig_attrib'
            '_seeds_1_mod_3_height_{}_FULL_TEST_SET.csv'.format(max_height))

        for seed_num in range(0, NUM_RANDOM_SEEDS_TO_USE, NUM_PROCS):
            main(DATASET_NAMES, DATASETS_FILEPATHS, KEY_ATTRIB_INDICES,
                 CLASS_ATTRIB_INDICES, SPLIT_CHARS, MISSING_VALUE_STRINGS,
                 max_height, OUTPUT_CSV_FILEPATH, seed_num + 1, load_numeric=False,
                 use_stop_conditions=False)
