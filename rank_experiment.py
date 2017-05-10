#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run cross-validation tests with decision trees, usually with maximum depth >= 2.
'''


import datetime
import itertools
import os
import math
import random
import sys
import timeit

import criteria
import dataset
import decision_tree

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


#: Initial seeds used in `random` and `numpy.random` modules, in order of `trial_number`.
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


def main(experiment_config):
    """Sets the configurations according to `experiment_config` and runs them.
    """
    raw_output_filepath = os.path.join(experiment_config["output folder"], 'raw_output.csv')
    with open(raw_output_filepath, 'w') as fout:
        init_raw_output_csv(fout, output_split_char=',')
        criteria_list = get_criteria(experiment_config["criteria"])

        if experiment_config["use enough depth"]:
            experiment_config["max depth"] = None

        if experiment_config["prunning parameters"]["use chi-sq test"]:
            max_p_value_chi_sq = experiment_config["prunning parameters"]["max chi-sq p-value"]
            decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = experiment_config[
                "prunning parameters"]["second most freq value min samples"]
        else:
            max_p_value_chi_sq = None
            decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = None

        decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
            "prunning parameters"]["use second most freq class min samples"]
        if decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
                "prunning parameters"]["second most freq class min samples"]
        else:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = None

        if experiment_config["use all datasets"]:
            datasets_configs = dataset.load_all_configs(experiment_config["datasets basepath"])
            datasets_configs.sort(key=lambda config: config["dataset name"])
        else:
            datasets_folders = [os.path.join(experiment_config["datasets basepath"], folderpath)
                                for folderpath in experiment_config["datasets folders"]]
            datasets_configs = [dataset.load_config(folderpath)
                                for folderpath in datasets_folders]
        if experiment_config["load one dataset at a time"]:
            for (dataset_config,
                 min_num_samples_allowed) in itertools.product(
                     datasets_configs,
                     experiment_config["prunning parameters"]["min num samples allowed"]):
                curr_dataset = dataset.Dataset(dataset_config["filepath"],
                                               dataset_config["key attrib index"],
                                               dataset_config["class attrib index"],
                                               dataset_config["split char"],
                                               dataset_config["missing value string"],
                                               experiment_config["use numeric attributes"])
                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_config["dataset name"],
                        curr_dataset,
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        num_folds=experiment_config["num folds"],
                        is_stratified=experiment_config["is stratified"],
                        use_numeric_attributes=experiment_config["use numeric attributes"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        output_file_descriptor=fout,
                        output_split_char=',')
        else:
            datasets = dataset.load_all_datasets(datasets_configs,
                                                 experiment_config["use numeric attributes"])
            for ((dataset_name, curr_dataset),
                 min_num_samples_allowed) in itertools.product(
                     datasets,
                     experiment_config["prunning parameters"]["min num samples allowed"]):
                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_name,
                        curr_dataset,
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        num_folds=experiment_config["num folds"],
                        is_stratified=experiment_config["is stratified"],
                        use_numeric_attributes=experiment_config["use numeric attributes"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        output_file_descriptor=fout,
                        output_split_char=',')


def init_raw_output_csv(raw_output_file_descriptor, output_split_char=','):
    """Writes the header to the raw output CSV file.
    """
    fields_list = ['Date Time',
                   'Dataset',
                   'Use Numeric attributes?',
                   'Attribute Name',
                   'Is the Attribute Numeric?',
                   'Number of Different Values (only for Nominal Attributes)',
                   'Total Number of Samples',
                   'Trial Number',
                   'Criterion',
                   'Maximum Depth Allowed',
                   'Total Number of folds',
                   'Fold Number',
                   'Is stratified?',

                   'Number of Samples Forcing a Leaf',
                   'Use Min Samples in Second Largest Class?',
                   'Min Samples in Second Largest Class',

                   'Uses Chi-Square Test',
                   'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',
                   'Minimum Number in Second Most Frequent Value',

                   'Number of Attributes in Dataset',
                   'Number of Valid Attributes in Dataset (m)',
                   'Number of Valid Nominal Attributes in Dataset (m_nom)',
                   'Number of Valid Numeric Attributes in Dataset (m_num)',

                   'Total Number of Inversions (pairs)',
                   'Total Number of Ties (pairs)',
                   'Total Number of Correct (pairs)',

                   'Criterion Value in Root Node',

                   'Total Time Taken for Current Fold [s]',

                   'Accuracy Percentage on Trivial Trees (with no splits)',

                   'Accuracy Percentage (with missing values)',
                   'Accuracy Percentage (without missing values)',
                   'Number of Samples Classified using Unkown Value',
                   'Percentage of Samples with Unkown Values for Accepted Attribute',

                   'Number of Nodes (after prunning)',
                   'Tree Depth (after prunning)',
                   'Number of Nodes Pruned']
    print(output_split_char.join(fields_list), file=raw_output_file_descriptor)
    raw_output_file_descriptor.flush()


def get_criteria(criteria_names_list):
    """Given a list of criteria names, returns a list of all there criteria (as `Criterion`'s).
    If a criterion name is unkown, the system will exit the experiment.
    """
    criteria_list = []
    for criterion_name in criteria_names_list:
        if criterion_name == "Twoing":
            criteria_list.append(criteria.Twoing())
        elif criterion_name == "GW Squared Gini":
            criteria_list.append(criteria.GWSquaredGini())
        elif criterion_name == "GW Chi Square":
            criteria_list.append(criteria.GWChiSquare())
        elif criterion_name == "LS Squared Gini":
            criteria_list.append(criteria.LSSquaredGini())
        elif criterion_name == "LS Chi Square":
            criteria_list.append(criteria.LSChiSquare())
        else:
            print('Unkown criterion name:', criterion_name)
            print('Exiting.')
            sys.exit(1)
    return criteria_list


def run(dataset_name, curr_dataset, criterion, min_num_samples_allowed, max_depth, num_trials,
        num_folds, is_stratified, use_numeric_attributes, use_chi_sq_test, max_p_value_chi_sq,
        output_file_descriptor, output_split_char=',', seed=None):
    """Runs `num_trials` experiments, each one doing a stratified cross-validation in `num_folds`
    folds. Saves the training and classification information in the `output_file_descriptor` file.
    """
    def _run_fold(dataset_name, curr_dataset, criterion, trial_number, min_num_samples_allowed,
                  max_depth, num_folds, is_stratified, use_numeric_attributes, use_chi_sq_test,
                  max_p_value_chi_sq, num_samples, original_valid_nominal_attributes,
                  original_valid_numeric_attributes, training_samples_indices,
                  validation_sample_indices, output_file_descriptor, output_split_char=','):
        print('Fold #{}'.format(fold_number + 1))
        print_information_per_attrib = {} # ...[attrib_index] = print_information
        accuracy_criterion_value = [] # ...[...] = (accuracy_with_missing_values, criterion_value)
        tree = decision_tree.DecisionTree(criterion)

        num_attributes = len(original_valid_nominal_attributes)
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(original_valid_nominal_attributes,
                                                         original_valid_numeric_attributes)):
            if not is_valid_nominal_attrib and not is_valid_numeric_attrib:
                continue

            # Let's pretend only the current attribute is valid.
            print()
            print('Current attribute: {} ({})'.format(
                curr_dataset.attrib_names[attrib_index], attrib_index))
            curr_dataset.valid_nominal_attribute = [False] * num_attributes
            curr_dataset.valid_nominal_attribute[attrib_index] = is_valid_nominal_attrib
            curr_dataset.valid_numeric_attribute = [False] * num_attributes
            curr_dataset.valid_numeric_attribute[attrib_index] = is_valid_numeric_attrib

            num_values = len(curr_dataset.attrib_int_to_value[attrib_index])
            if not num_values:
                continue

            if max_depth is None:
                curr_max_depth_allowed = 1 + math.ceil(math.log2(curr_dataset.num_classes))
            else:
                curr_max_depth_allowed = max_depth

            start_time = timeit.default_timer()
            ((_,
              num_correct_classifications_w_unkown,
              num_correct_classifications_wo_unkown,
              _,
              _,
              _,
              num_unkown,
              _),
             curr_max_depth_found,
             _,
             curr_num_nodes_prunned) = tree.train_and_test(
                 curr_dataset,
                 training_samples_indices,
                 validation_sample_indices,
                 max_depth=curr_max_depth_allowed,
                 min_samples_per_node=min_num_samples_allowed,
                 use_stop_conditions=use_chi_sq_test,
                 max_p_value_chi_sq=max_p_value_chi_sq)
            total_time_taken = timeit.default_timer() - start_time
            if (not tree.get_root_node().valid_nominal_attribute[attrib_index]
                    and not tree.get_root_node().valid_numeric_attribute[attrib_index]):
                continue
            try:
                curr_criterion_value = tree.get_root_node().node_split.criterion_value
            except AttributeError:
                continue

            trivial_accuracy = tree.get_trivial_accuracy(validation_sample_indices)
            accuracy_with_missing_values = (100.0 * num_correct_classifications_w_unkown
                                            / len(validation_sample_indices))
            try:
                accuracy_without_missing_values = (100.0 * num_correct_classifications_wo_unkown
                                                   / (len(validation_sample_indices) - num_unkown))
            except ZeroDivisionError:
                accuracy_without_missing_values = None

            percentage_unkown = 100.0 * num_unkown / len(validation_sample_indices)
            curr_num_nodes = tree.get_root_node().get_num_nodes()

            print_information_per_attrib[attrib_index] = [curr_criterion_value,
                                                          curr_max_depth_allowed,
                                                          num_values,
                                                          total_time_taken,
                                                          trivial_accuracy,
                                                          accuracy_with_missing_values,
                                                          accuracy_without_missing_values,
                                                          num_unkown,
                                                          percentage_unkown,
                                                          curr_num_nodes,
                                                          curr_max_depth_found,
                                                          curr_num_nodes_prunned]
            accuracy_criterion_value.append((accuracy_with_missing_values, curr_criterion_value))

        (num_inversions,
         num_ties,
         num_correct) = _count_inversions_and_ties(accuracy_criterion_value)

        num_valid_attributes = len(print_information_per_attrib)
        num_valid_numeric_attributes = sum(original_valid_numeric_attributes[attrib_index]
                                           for attrib_index in print_information_per_attrib)
        num_valid_nominal_attributes = num_valid_attributes - num_valid_numeric_attributes

        for attrib_index in sorted(print_information_per_attrib):
            save_info(dataset_name,
                      use_numeric_attributes,
                      curr_dataset.attrib_names[attrib_index],
                      original_valid_numeric_attributes[attrib_index],
                      num_samples,
                      trial_number + 1,
                      criterion.name,
                      num_folds,
                      fold_number + 1,
                      is_stratified,
                      min_num_samples_allowed,
                      use_chi_sq_test,
                      max_p_value_chi_sq,
                      num_attributes,
                      num_valid_attributes,
                      num_valid_nominal_attributes,
                      num_valid_numeric_attributes,
                      num_inversions,
                      num_ties,
                      num_correct,
                      *print_information_per_attrib[attrib_index],
                      output_file_descriptor,
                      output_split_char)


    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_attributes = len(curr_dataset.valid_nominal_attribute)
    if not use_numeric_attributes:
        curr_dataset.valid_numeric_attribute = [False] * num_attributes
    original_valid_nominal_attributes = curr_dataset.valid_nominal_attribute[:]
    original_valid_numeric_attributes = curr_dataset.valid_numeric_attribute[:]

    sample_indices_and_classes = list(enumerate(curr_dataset.sample_class))
    num_samples = len(sample_indices_and_classes)
    for trial_number in range(num_trials):
        print('*'*80)
        print('STARTING TRIAL #{}'.format(trial_number + 1))
        print()

        if seed is None:
            random.seed(RANDOM_SEEDS[trial_number])
            np.random.seed(RANDOM_SEEDS[trial_number])
        random.shuffle(sample_indices_and_classes)
        shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)


        if is_stratified:
            for (fold_number,
                 (training_randomized_indices,
                  validation_randomized_indices)) in enumerate(
                      StratifiedKFold(n_splits=num_folds).split(shuffled_sample_indices,
                                                                shuffled_sample_classes)):
                training_samples_indices = [shuffled_sample_indices[index]
                                            for index in training_randomized_indices]
                validation_sample_indices = [shuffled_sample_indices[index]
                                             for index in validation_randomized_indices]
                _run_fold(dataset_name, curr_dataset, criterion, trial_number,
                          min_num_samples_allowed, max_depth, num_folds, is_stratified,
                          use_numeric_attributes, use_chi_sq_test, max_p_value_chi_sq, num_samples,
                          original_valid_nominal_attributes, original_valid_numeric_attributes,
                          training_samples_indices, validation_sample_indices,
                          output_file_descriptor, output_split_char)
        else: # is NOT stratified
            for (fold_number,
                 (training_samples_indices,
                  validation_sample_indices)) in enumerate(
                      KFold(n_splits=num_folds).split(shuffled_sample_indices)):
                training_samples_indices = [shuffled_sample_indices[index]
                                            for index in training_randomized_indices]
                validation_sample_indices = [shuffled_sample_indices[index]
                                             for index in validation_randomized_indices]
                _run_fold(dataset_name, curr_dataset, criterion, trial_number,
                          min_num_samples_allowed, max_depth, num_folds, is_stratified,
                          use_numeric_attributes, use_chi_sq_test, max_p_value_chi_sq, num_samples,
                          original_valid_nominal_attributes, original_valid_numeric_attributes,
                          training_samples_indices, validation_sample_indices,
                          output_file_descriptor, output_split_char)

    # Resets the valid attributes lists to the original values to be used in any future experiment.
    curr_dataset.valid_nominal_attribute = original_valid_nominal_attributes
    curr_dataset.valid_numeric_attribute = original_valid_numeric_attributes


def _count_inversions_and_ties(accuracy_criterion_value):
    num_inversions = 0.0
    num_ties = 0.0
    num_correct = 0.0
    accuracy_criterion_value.sort(reverse=True)
    for index, (curr_accuracy, curr_criterion_value) in enumerate(accuracy_criterion_value):
        for fwd_accuracy, fwd_criterion_value in accuracy_criterion_value[index + 1:]:
            # Note that, since accuracy_criterion_value is sorted, curr_accuracy >= fwd_accuracy.
            if curr_accuracy == fwd_accuracy:
                num_ties += 1.
            elif fwd_criterion_value > curr_criterion_value:
                num_inversions += 1.
            elif fwd_criterion_value == curr_criterion_value:
                num_inversions += 0.5
                num_correct += 0.5
            else:
                num_correct += 1.
    return (num_inversions, num_ties, num_correct)


def save_info(dataset_name, use_numeric_attributes, attrib_name, is_numeric, num_samples,
              trial_number, criterion_name, num_folds, curr_fold_number, is_stratified,
              min_num_samples_allowed, use_chi_sq_test, max_p_value_chi_sq, num_attributes,
              num_valid_attributes, num_valid_nominal_attributes, num_valid_numeric_attributes,
              num_inversions, num_ties, num_correct, criterion_value, curr_max_depth_allowed,
              num_values, total_time_taken, trivial_accuracy, accuracy_with_missing_values,
              accuracy_without_missing_values, num_unkown, percentage_unkown, curr_num_nodes,
              curr_max_depth_found, curr_num_nodes_prunned, output_file_descriptor,
              output_split_char=','):
    """Saves the experiment's trial information in the CSV file.
    """
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(use_numeric_attributes),
                 attrib_name,
                 str(is_numeric),
                 str(num_values),
                 str(num_samples),
                 str(trial_number),
                 criterion_name,
                 str(curr_max_depth_allowed),
                 str(num_folds),
                 str(curr_fold_number),
                 str(is_stratified),

                 str(min_num_samples_allowed),
                 str(decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS),
                 str(decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),
                 str(decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE),

                 str(num_attributes),
                 str(num_valid_attributes),
                 str(num_valid_nominal_attributes),
                 str(num_valid_numeric_attributes),

                 str(num_inversions),
                 str(num_ties),
                 str(num_correct),

                 str(criterion_value),

                 str(total_time_taken),

                 str(trivial_accuracy),

                 str(accuracy_with_missing_values),
                 str(accuracy_without_missing_values),
                 str(num_unkown),
                 str(percentage_unkown),

                 str(curr_num_nodes),
                 str(curr_max_depth_found),
                 str(curr_num_nodes_prunned)]
    print(output_split_char.join(line_list), file=output_file_descriptor)
    output_file_descriptor.flush()
