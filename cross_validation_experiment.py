#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run cross-validation tests with decision trees, usually with maximum depth >= 2.
'''


import datetime
import itertools
import os
import random
import sys
import timeit

import criteria
import dataset
import decision_tree

import numpy as np


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

        if "starting seed index" not in experiment_config:
            starting_seed = 1
        else:
            starting_seed = experiment_config["starting seed index"]

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
                        starting_seed=starting_seed,
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
                        starting_seed=starting_seed,
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
                   'Total Number of Samples',
                   'Trial Number',
                   'Criterion',
                   'Maximum Depth Allowed',
                   'Number of folds',
                   'Is stratified?',
                   'Use Numeric Attributes?',

                   'Number of Samples Forcing a Leaf',
                   'Use Min Samples in Second Largest Class?',
                   'Min Samples in Second Largest Class',

                   'Uses Chi-Square Test',
                   'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',
                   'Minimum Number in Second Most Frequent Value',

                   'Average Number of Valid Attributes in Root Node (m)',
                   'Average Number of Valid Nominal Attributes in Root Node (m_nom)',
                   'Average Number of Valid Numeric Attributes in Root Node (m_num)',

                   'Total Time Taken for Cross-Validation [s]',

                   'Accuracy Percentage on Trivial Trees (with no splits)',

                   'Accuracy Percentage (with missing values)',
                   'Accuracy Percentage (without missing values)',
                   'Number of Samples Classified using Unkown Value',
                   'Percentage of Samples with Unkown Values for Accepted Attribute',

                   'Average Number of Values of Attribute Chosen at Root Node',
                   'Maximum Number of Values of Attribute Chosen at Root Node',
                   'Minimum Number of Values of Attribute Chosen at Root Node',

                   'Number of Trivial Splits [between 0 and number of folds]',

                   'Average Number of Nodes (after prunning)',
                   'Maximum Number of Nodes (after prunning)',
                   'Minimum Number of Nodes (after prunning)',

                   'Average Tree Depth (after prunning)',
                   'Maximum Tree Depth (after prunning)',
                   'Minimum Tree Depth (after prunning)',

                   'Average Number of Nodes Pruned']
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


def run(dataset_name, train_dataset, criterion, min_num_samples_allowed, max_depth, num_trials,
        starting_seed, num_folds, is_stratified, use_numeric_attributes, use_chi_sq_test,
        max_p_value_chi_sq, output_file_descriptor, output_split_char=',', seed=None):
    """Runs `num_trials` experiments, each one doing a stratified cross-validation in `num_folds`
    folds. Saves the training and classification information in the `output_file_descriptor` file.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for trial_number in range(num_trials):
        print('*'*80)
        print('STARTING TRIAL #{} WITH SEED #{}'.format(trial_number + 1, starting_seed))
        print()

        if seed is None:
            random.seed(RANDOM_SEEDS[trial_number + starting_seed - 1])
            np.random.seed(RANDOM_SEEDS[trial_number + starting_seed - 1])

        tree = decision_tree.DecisionTree(criterion=criterion)

        start_time = timeit.default_timer()
        (_,
         num_correct_classifications_w_unkown,
         num_correct_classifications_wo_unkown,
         _,
         _,
         _,
         num_unkown,
         _,
         _,
         num_nodes_prunned_per_fold,
         max_depth_per_fold,
         num_nodes_per_fold,
         num_valid_attributes_in_root,
         num_valid_nominal_attributes_in_root,
         num_valid_numeric_attributes_in_root,
         num_values_root_attribute_list,
         num_trivial_splits,
         trivial_accuracy_percentage) = tree.cross_validate(
             curr_dataset=train_dataset,
             num_folds=num_folds,
             max_depth=max_depth,
             min_samples_per_node=min_num_samples_allowed,
             is_stratified=is_stratified,
             print_tree=False,
             print_samples=False,
             use_stop_conditions=use_chi_sq_test,
             max_p_value_chi_sq=max_p_value_chi_sq)
        total_time_taken = timeit.default_timer() - start_time
        accuracy_with_missing_values = (100.0 * num_correct_classifications_w_unkown
                                        / train_dataset.num_samples)
        try:
            accuracy_without_missing_values = (100.0 * num_correct_classifications_wo_unkown
                                               / (train_dataset.num_samples - num_unkown))
        except ZeroDivisionError:
            accuracy_without_missing_values = None

        percentage_unkown = 100.0 * num_unkown / train_dataset.num_samples

        if num_values_root_attribute_list:
            (avg_num_values_root_attribute,
             max_num_values_root_attribute,
             min_num_values_root_attribute) = (np.mean(num_values_root_attribute_list),
                                               np.amax(num_values_root_attribute_list),
                                               np.amin(num_values_root_attribute_list))
        else:
            (avg_num_values_root_attribute,
             max_num_values_root_attribute,
             min_num_values_root_attribute) = (None, None, None)

        save_trial_info(dataset_name, train_dataset.num_samples, trial_number + starting_seed,
                        criterion.name, max_depth, num_folds, is_stratified, use_numeric_attributes,
                        min_num_samples_allowed, decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        use_chi_sq_test, max_p_value_chi_sq,
                        decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE,
                        np.mean(num_valid_attributes_in_root),
                        np.mean(num_valid_nominal_attributes_in_root),
                        np.mean(num_valid_numeric_attributes_in_root), total_time_taken,
                        trivial_accuracy_percentage, accuracy_with_missing_values,
                        accuracy_without_missing_values, num_unkown, percentage_unkown,
                        avg_num_values_root_attribute, max_num_values_root_attribute,
                        min_num_values_root_attribute, num_trivial_splits,
                        np.mean(num_nodes_per_fold), np.amax(num_nodes_per_fold),
                        np.amin(num_nodes_per_fold), np.mean(max_depth_per_fold),
                        np.amax(max_depth_per_fold), np.amin(max_depth_per_fold),
                        np.mean(num_nodes_prunned_per_fold), output_split_char,
                        output_file_descriptor)


def save_trial_info(dataset_name, num_total_samples, trial_number, criterion_name,
                    max_depth, num_folds, is_stratified, use_numeric_attributes,
                    min_num_samples_allowed, use_min_samples_second_largest_class,
                    min_samples_second_largest_class, use_chi_sq_test, max_p_value_chi_sq,
                    min_num_second_most_freq_value, avg_num_valid_attributes_in_root,
                    avg_num_valid_nominal_attributes_in_root,
                    avg_num_valid_numeric_attributes_in_root, total_time_taken,
                    trivial_accuracy_percentage, accuracy_with_missing_values,
                    accuracy_without_missing_values, num_unkown, percentage_unkown,
                    avg_num_values_root_attribute, max_num_values_root_attribute,
                    min_num_values_root_attribute, num_trivial_splits, avg_num_nodes, max_num_nodes,
                    min_num_nodes, avg_tree_depth, max_tree_depth, min_tree_depth,
                    avg_num_nodes_pruned, output_split_char, output_file_descriptor):
    """Saves the experiment's trial information in the CSV file.
    """
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_total_samples),
                 str(trial_number),
                 criterion_name,
                 str(max_depth),
                 str(num_folds),
                 str(is_stratified),
                 str(use_numeric_attributes),

                 str(min_num_samples_allowed),
                 str(use_min_samples_second_largest_class),
                 str(min_samples_second_largest_class),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),
                 str(min_num_second_most_freq_value),

                 str(avg_num_valid_attributes_in_root),
                 str(avg_num_valid_nominal_attributes_in_root),
                 str(avg_num_valid_numeric_attributes_in_root),

                 str(total_time_taken),

                 str(trivial_accuracy_percentage),

                 str(accuracy_with_missing_values),
                 str(accuracy_without_missing_values),
                 str(num_unkown),
                 str(percentage_unkown),

                 str(avg_num_values_root_attribute),
                 str(max_num_values_root_attribute),
                 str(min_num_values_root_attribute),
                 str(num_trivial_splits),

                 str(avg_num_nodes),
                 str(max_num_nodes),
                 str(min_num_nodes),

                 str(avg_tree_depth),
                 str(max_tree_depth),
                 str(min_tree_depth),

                 str(avg_num_nodes_pruned)]
    print(output_split_char.join(line_list), file=output_file_descriptor)
    output_file_descriptor.flush()
