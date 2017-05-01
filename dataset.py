#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains only the Dataset class.
"""

import copy
import json
import os
import sys
import timeit


class Dataset(object):
    """This class contains information about the loaded dataset used for training of decision
    trees.

    Includes auxiliary data obtained from the dataset. Sometimes used for cross-validation. May
    also include test data together with training data, so that all possible values are known for
    each attribute.

    Args:
        training_dataset_csv_filepath (str): System filepath to csv file containing the training
            dataset.
        key_attrib_index (int, optional): CSV column index of the sample keys. If 'None', samples
            will be numbered in order of appearance.
        class_attrib_index (int): CSV column index of the sample class.
        split_char (str, optional): Split char used in the CSV file. Defaults to ';'.
        missing_value_string (str): Indicates the current sample does not have this value.

    Attributes:
        attrib_names (:obj:'list' of :obj'str'): Names of each attribute in order of appearance.
        num_classes (int): number of different classes in dataset.
        num_samples (int): number of samples in dataset.
        sample_index_to_key (:obj:'list' of 'str'): Sample names in order of appearance in CSV.
        sample_key_to_index (:obj:'dict' from 'str' to 'int'): Sample index by key.
        samples (:obj:'list' of 'list' of 'int'): list of samples, each represented by a list of
            it's attributes values (represented by ints).
        attrib_int_to_value (:obj:'list' of 'list' of 'str'): Given an attribute index and it's int
            value representation, returns the string value represented by this int.
        attrib_value_to_int (:obj:'list' of 'list' of 'str'): Given an attribute index and it's
            string value, returns the int value that represents it.
        valid_nominal_attribute (:obj:'list' of 'bool'): Indicates if each attribute is valid and
            nominal.
        valid_numeric_attribute (:obj:'list' of 'bool'): Indicates if each attribute is valid and
            numeric.
        class_name_to_int (:obj:'dict' from 'str' to 'int'): Given a class name, returns the int
            value that represents it.
        class_int_to_name (:obj:'list' of 'str'): Given a class int value representation, returns
            the class name it represents.
        class_index_num_samples (:obj:'list' of 'int'): Number of samples for the class given by
            this int representation.
        number_samples_in_rarest_class ('int'): Number of samples in the rarest class in CSV.
        sample_class (:obj:'list' of 'int'): Class for each sample, by index.
        sample_costs (:obj:'list' of 'list' of 'float'): Misclassification cost for each sample, by
            class int.
        training_dataset_csv_filepath (str): System filepath to csv file containing the training
            dataset.
        key_attrib_index (int, optional): CSV column index of the sample keys. If 'None', samples
            will be numbered in order of appearance.
        class_attrib_index (int): CSV column index of the sample class.
        load_train_dataset_time_taken (float): Total time taken to load dataset and initialize this
            object.
    """

    def __init__(self, training_dataset_csv_filepath, key_attrib_index, class_attrib_index,
                 split_char, missing_value_string, load_numeric=False):
        # Init variables to default value
        self.attrib_names = []
        self.num_classes = 0
        self.num_samples = 0

        self.sample_index_to_key = [] # [sample_index] = key
        self.sample_key_to_index = {} # [key] = sample_index
        self.samples = [] # [sample_index][attrib_index] = sample_attrib_value
        self.sample_class = [] # [sample_index] = int_class
        self.sample_costs = [] # [sample_index][class_index] = misclassification_cost

        self.test_sample_index_to_key = [] # [sample_index] = key
        self.test_sample_key_to_index = {} # [key] = sample_index
        self.test_samples = [] # [sample_index][attrib_index] = sample_attrib_value
        self.test_sample_class = [] # [sample_index] = int_class
        self.test_sample_costs = [] # [sample_index][class_index] = misclassification_cost
        self.test_num_samples = 0
        self.test_dataset_csv_filepath = None

        self.attrib_int_to_value = [] # [attrib_index][int_value] = string_value
        self.attrib_value_to_int = [] # [attrib_index][string_value] = int_value
        self.valid_nominal_attribute = [] # [attrib_index] = boolean
        self.valid_numeric_attribute = [] # [attrib_index] = boolean

        self.class_name_to_int = {}
        self.class_int_to_name = []
        self.class_index_num_samples = []
        self.number_samples_in_rarest_class = 0

        self.training_dataset_csv_filepath = training_dataset_csv_filepath
        self.key_attrib_index = key_attrib_index
        self.class_attrib_index = class_attrib_index

        self.load_train_dataset_time_taken = None
        self.load_numeric = load_numeric

        self._load_train_dataset(split_char, missing_value_string)
        self._print_loaded_information()

    def _load_train_dataset(self, split_char, missing_value_string):
        # TESTED!
        """Loads the CSV and initialize auxiliary data.

        Args:
            split_char (str, optional): Split char used in the CSV file. Defaults to ';'.
            missing_value_string (str): Indicates the current sample does not have this value.
        """
        print()
        print('LOADING dataset...')
        if self.class_attrib_index is None:
            print('Error: No class attribute!')
            sys.exit(1)
        if self.key_attrib_index is None:
            print('No key attribute used! Numbering samples in order of appearance.')
        samples_counter = -1 # header is 0, first sample is 1
        wrong_samples = 0
        num_attributes = None
        with open(self.training_dataset_csv_filepath, 'r') as fin:
            start_time = timeit.default_timer()
            for line in fin:
                line_list = line.rstrip().split(split_char)
                if not line_list:
                    # empty line
                    continue
                samples_counter += 1
                if samples_counter == 0:
                    # header
                    num_attributes = len(line_list)
                    for attrib_index, attrib_name in enumerate(line_list):
                        self.valid_nominal_attribute.append(True)
                        self.valid_numeric_attribute.append(self.load_numeric)
                        self.attrib_value_to_int.append({})
                        self.attrib_int_to_value.append([])
                        self.attrib_names.append(attrib_name)
                    if self.key_attrib_index is not None:
                        self.valid_nominal_attribute[self.key_attrib_index] = False
                        self.valid_numeric_attribute[self.key_attrib_index] = False
                    self.valid_nominal_attribute[self.class_attrib_index] = False
                    self.valid_numeric_attribute[self.class_attrib_index] = False
                    continue

                # not header

                if len(line_list) != num_attributes:
                    # Sample with wrong number of attributes
                    wrong_samples += 1
                    print('\tSample {} with wrong number of attributes: {} instead of {}.'.format(
                        samples_counter,
                        len(line_list),
                        num_attributes))
                    print('\n\t{}\n'.format(line))
                    continue

                # Sample with correct number of attributes

                # Key
                sample_index = samples_counter - wrong_samples - 1
                key = '' # just to have it in scope in the second 'if' below
                if self.key_attrib_index is not None:
                    try:
                        key = line_list[self.key_attrib_index]
                    except IndexError:
                        print('Key attribute index '
                              '({}) is equal or larger than the number of attributes ({}).'.format(
                                  self.key_attrib_index,
                                  len(line_list)))
                        sys.exit(1)
                else:
                    sample_name_index = samples_counter - 1
                    key = 'sample_{}'.format(sample_name_index)
                if key in self.sample_key_to_index:
                    print('Repeated key: {}'.format(key))
                    sys.exit(1)

                # Class
                try:
                    sample_class_name = line_list[self.class_attrib_index]
                except IndexError:
                    print('Class attribute index '
                          '({}) is equal or larger than the number of attributes ({}).'.format(
                              self.class_attrib_index,
                              len(line_list)))
                    sys.exit(1)
                if sample_class_name in self.class_name_to_int:
                    sample_int_class = self.class_name_to_int[sample_class_name]
                else:
                    sample_int_class = len(self.class_int_to_name)
                    self.class_name_to_int[sample_class_name] = sample_int_class
                    self.class_int_to_name.append(sample_class_name)
                    self.class_index_num_samples.append(0)

                # Sample and attributes
                sample = copy.copy(line_list)
                is_correct = True
                for attrib_index, value in enumerate(line_list):
                    if self.valid_nominal_attribute[attrib_index]:
                        try:
                            float(value)
                            # If did not throw, then attribute is numeric, thus invalid.
                            self.valid_nominal_attribute[attrib_index] = False
                            self.attrib_value_to_int[attrib_index] = {}
                            self.attrib_int_to_value[attrib_index] = []
                        except ValueError:
                            if value == missing_value_string:
                                # This sample won't be saved in dataset
                                print('\tSample {} has a missing value in attribute {} ({})'.format(
                                    key,
                                    attrib_index,
                                    self.attrib_names[attrib_index]))
                                wrong_samples += 1
                                is_correct = False
                                break
                            elif self.valid_numeric_attribute[attrib_index]:
                                self.valid_numeric_attribute[attrib_index] = False
                            if value not in self.attrib_value_to_int[attrib_index]:
                                self.attrib_value_to_int[attrib_index][value] = len(
                                    self.attrib_int_to_value[attrib_index])
                                self.attrib_int_to_value[attrib_index].append(value)
                            sample[attrib_index] = self.attrib_value_to_int[attrib_index][value]
                    if self.valid_numeric_attribute[attrib_index]:
                        try:
                            sample[attrib_index] = float(value)
                            # If did not throw, then attribute is numeric.
                            if self.valid_nominal_attribute[attrib_index]:
                                self.valid_nominal_attribute[attrib_index] = False
                                self.attrib_value_to_int[attrib_index] = {}
                                self.attrib_int_to_value[attrib_index] = []
                        except ValueError:
                            if value == missing_value_string:
                                # This sample won't be saved in dataset
                                print('\tSample {} has a missing value in attribute {} ({})'.format(
                                    key,
                                    attrib_index,
                                    self.attrib_names[attrib_index]))
                                wrong_samples += 1
                                is_correct = False
                                break
                            self.valid_numeric_attribute[attrib_index] = False


                if is_correct:
                    # Save this sample in dataset
                    self.samples.append(sample)
                    self.sample_index_to_key.append(key)
                    self.sample_key_to_index[key] = sample_index
                    self.sample_class.append(sample_int_class)
                    self.class_index_num_samples[sample_int_class] += 1

        self.num_samples = samples_counter - wrong_samples
        self.num_classes = len(self.class_int_to_name)
        self.number_samples_in_rarest_class = min(self.class_index_num_samples)
        self.sample_costs = self._initialize_integer_costs(self.sample_class)
        self.load_train_dataset_time_taken = timeit.default_timer() - start_time
        if (num_attributes == 0
                or (sum(self.valid_nominal_attribute) == 0
                    and sum(self.valid_numeric_attribute) == 0)):
            print('Must have at least ONE valid attribute.')
            sys.exit(1)

    def _initialize_integer_costs(self, sample_class):
        # TESTED!
        """Initialize costs for each sample (1.0 for wrong class and 0.0 for the correct one).
        """
        sample_costs = []
        for curr_sample_class in sample_class:
            sample_costs.append([1.0] * self.num_classes)
            sample_costs[-1][curr_sample_class] = 0.0
        return sample_costs

    def _print_loaded_information(self):
        # TESTED!
        """Prints basic information of the loaded CSV.
        """
        print('Number of attributes: {}'.format(
            len(self.valid_nominal_attribute) + len(self.valid_numeric_attribute)))
        print('Number of nominal attributes: {}'.format(len(self.valid_nominal_attribute)))
        print('Number of numeric attributes: {}'.format(len(self.valid_numeric_attribute)))
        print('{} samples found!'.format(self.num_samples))
        print('{} classes found:'.format(self.num_classes))
        for class_index in range(self.num_classes):
            print('\tClass # {}: "{}" ({} samples)'.format(
                class_index,
                self.class_int_to_name[class_index],
                self.class_index_num_samples[class_index]))
        print('Time taken to load training dataset: {:.6f}s'.format(
            self.load_train_dataset_time_taken))

    def load_test_set_from_csv(self, test_dataset_csv_filepath, key_attrib_index,
                               class_attrib_index, split_char, missing_value_string):
        """Loads the CSV and initialize auxiliary data.

        Args:
            test_dataset_csv_filepath (str): path to the test dataset.
            key_attrib_index (int): column index of the samples' keys on the csv.
            class_attrib_index (int): column index of the samples' classes on the csv.
            split_char (str): char used to split columns in the csv.
            missing_value_string (str): string used to indicate that a sample does not have a value.
        """
        def _is_header_match(train_attrib_names, test_attrib_names):
            """Tests wether both headers are the same (up to lower/uppercase)."""
            if len(train_attrib_names) != len(test_attrib_names):
                return False
            for curr_train_attrib_name, curr_test_attrib_name in zip(train_attrib_names,
                                                                     test_attrib_names):
                if curr_train_attrib_name.lower() != curr_test_attrib_name.lower():
                    return False
            return True

        # First let's remove any previously loaded test set
        self.test_sample_index_to_key = []
        self.test_sample_key_to_index = {}
        self.test_samples = []
        self.test_sample_class = []
        self.test_sample_costs = []
        self.test_num_samples = 0
        self.test_dataset_csv_filepath = None

        if key_attrib_index != self.key_attrib_index:
            print('Test dataset key attribute ({}) is not equal to train'
                  ' dataset key attribute ({}).'.format(key_attrib_index,
                                                        self.key_attrib_index))
            sys.exit(1)

        if class_attrib_index != self.class_attrib_index:
            print('Test dataset class attribute ({}) is not equal to train'
                  ' dataset class attribute ({}).'.format(class_attrib_index,
                                                          self.class_attrib_index))
            sys.exit(1)

        samples_counter = -1 # header is 0, first sample is 1
        wrong_samples = 0
        with open(test_dataset_csv_filepath, 'r') as fin:
            start_time = timeit.default_timer()
            for line in fin:
                line_list = line.rstrip().split(split_char)
                if not line_list:
                    # empty line
                    continue
                samples_counter += 1
                if samples_counter == 0:
                    # header
                    if not _is_header_match(self.attrib_names, line_list):
                        print('Test dataset header is not equal to train dataset header.')
                        sys.exit(1)
                    continue

                # not header

                if len(line_list) != len(self.attrib_names):
                    # Sample with wrong number of attributes
                    wrong_samples += 1
                    print('\tSample {} with wrong number of attributes: {} instead of {}.'.format(
                        samples_counter,
                        len(line_list),
                        len(self.attrib_names)))
                    print('\n\t{}\n'.format(line))
                    continue

                # Sample with correct number of attributes

                # Key
                sample_index = samples_counter - wrong_samples - 1
                key = '' # just to have it in scope in the second 'if' below
                if key_attrib_index is not None:
                    key = line_list[key_attrib_index]
                else:
                    sample_name_index = samples_counter - 1
                    key = 'test_sample_{}'.format(sample_name_index)
                if key in self.test_sample_key_to_index:
                    print('Repeated key: {}'.format(key))
                    sys.exit(1)

                # Class
                sample_class_name = line_list[class_attrib_index]
                if sample_class_name in self.class_name_to_int:
                    sample_int_class = self.class_name_to_int[sample_class_name]
                else:
                    sample_int_class = len(self.class_int_to_name)
                    self.class_name_to_int[sample_class_name] = sample_int_class
                    self.class_int_to_name.append(sample_class_name)

                # Sample and attributes
                sample = copy.copy(line_list)
                for attrib_index, value in enumerate(line_list):
                    if self.valid_nominal_attribute[attrib_index]:
                        try:
                            float(value)
                            # If did not raise above, then attribute is numeric, thus invalid.
                            print('\tTest sample {} has numeric value ({}) in attribute {} ({})'
                                  ' (which is nominal).'.format(key,
                                                                value,
                                                                attrib_index,
                                                                self.attrib_names[attrib_index]))
                            sys.exit(1)
                        except ValueError:
                            if value == missing_value_string:
                                print('\tTest sample {} has missing value in attribute'
                                      ' {} ({}).'.format(key,
                                                         attrib_index,
                                                         self.attrib_names[attrib_index]))
                                sample[attrib_index] = -1
                                continue
                            if value not in self.attrib_value_to_int[attrib_index]:
                                self.attrib_value_to_int[attrib_index][value] = len(
                                    self.attrib_int_to_value[attrib_index])
                                self.attrib_int_to_value[attrib_index].append(value)
                            sample[attrib_index] = self.attrib_value_to_int[attrib_index][value]
                    elif self.valid_numeric_attribute[attrib_index]:
                        try:
                            sample[attrib_index] = float(value)
                        except ValueError:
                            if value == missing_value_string:
                                print('\tTest sample {} has missing value in attribute'
                                      ' {} ({}).'.format(key,
                                                         attrib_index,
                                                         self.attrib_names[attrib_index]))
                                sample[attrib_index] = None
                                continue
                            print('\tTest sample {} has nominal value ({}) in attribute {} ({})'
                                  ' (which is numeric).'.format(key,
                                                                value,
                                                                attrib_index,
                                                                self.attrib_names[attrib_index]))
                            sys.exit(1)

                # Save this sample in test dataset
                self.test_samples.append(sample)
                self.test_sample_index_to_key.append(key)
                self.test_sample_key_to_index[key] = sample_index
                self.test_sample_class.append(sample_int_class)

        self.test_sample_costs = self._initialize_integer_costs(self.test_sample_class)
        time_taken = timeit.default_timer() - start_time
        print('Time taken to load test dataset: {:.6f}s'.format(time_taken))
        self.test_num_samples = len(self.test_sample_index_to_key)
        self.test_dataset_csv_filepath = test_dataset_csv_filepath

    def _print_debug_info(self):
        # TESTED!
        print()
        print('DEBUG INFO:')
        print()
        print()

        print('self.attrib_names:')
        for int_key, attrib_name in enumerate(self.attrib_names):
            print('\t{} --> {}'.format(int_key, attrib_name))
        print()

        print('self.num_samples: {}'.format(self.num_samples))
        print()

        print('self.num_classes: {}'.format(self.num_classes))
        print()

        print('self.sample_index_to_key:')
        for int_key, key in enumerate(self.sample_index_to_key):
            print('\t{} --> {}'.format(int_key, key))
        print()

        print('self.sample_key_to_index:')
        for key, int_key in self.sample_key_to_index.items():
            print('\t{} --> {}'.format(key, int_key))
        print()

        print('self.samples:')
        for int_key, sample in enumerate(self.samples):
            print('\tSample # {}:'.format(int_key))
            for attrib_index, int_value in enumerate(sample):
                print('\t\t{} --> {}'.format(attrib_index, int_value))
        print()

        print('self.sample_class:')
        for int_key, sample_class in enumerate(self.sample_class):
            print('\t{} --> {}'.format(int_key, sample_class))
        print()

        print('self.sample_costs:')
        for int_key, cost_array in enumerate(self.sample_costs):
            print('\tSample # {}:'.format(int_key))
            for class_int, cost in enumerate(cost_array):
                print('\t\t{} --> {}'.format(class_int, cost))
        print()

        print('self.test_sample_index_to_key:')
        for int_key, key in enumerate(self.test_sample_index_to_key):
            print('\t{} --> {}'.format(int_key, key))
        print()

        print('self.test_sample_key_to_index:')
        for key, int_key in self.test_sample_key_to_index.items():
            print('\t{} --> {}'.format(key, int_key))
        print()

        print('self.test_samples:')
        for int_key, sample in enumerate(self.test_samples):
            print('\t{} --> {}'.format(int_key, sample))
        print()

        print('self.test_sample_class:')
        for int_key, sample_class in enumerate(self.test_sample_class):
            print('\t{} --> {}'.format(int_key, sample_class))
        print()

        print('self.test_sample_costs:')
        for int_key, cost_array in enumerate(self.test_sample_costs):
            print('\tTest Sample # {}:'.format(int_key))
            for class_int, cost in enumerate(cost_array):
                print('\t\t{} --> {}'.format(class_int, cost))
        print()

        print('self.test_dataset_csv_filepath: {}'.format(self.test_dataset_csv_filepath))
        print()

        print('self.valid_nominal_attribute:')
        for attrib_index, is_valid in enumerate(self.valid_nominal_attribute):
            print('\t{} --> {}'.format(attrib_index, is_valid))
        print()

        print('self.valid_numeric_attribute:')
        for attrib_index, is_valid in enumerate(self.valid_numeric_attribute):
            print('\t{} --> {}'.format(attrib_index, is_valid))
        print()

        print('self.attrib_int_to_value:')
        for attrib_index, int_to_values_array in enumerate(self.attrib_int_to_value):
            print('\tAtribute # {}:'.format(attrib_index))
            for value_int, value in enumerate(int_to_values_array):
                print('\t\t{} --> {}'.format(value_int, value))
        print()

        print('self.attrib_value_to_int:')
        for attrib_index, values_to_int_array in enumerate(self.attrib_value_to_int):
            print('\tAtribute # {}:'.format(attrib_index))
            for value, value_int in enumerate(values_to_int_array):
                print('\t\t{} --> {}'.format(value, value_int))
        print()

        print('self.number_samples_in_rarest_class: {}'.format(self.number_samples_in_rarest_class))
        print()

        print('self.training_dataset_csv_filepath: {}'.format(self.training_dataset_csv_filepath))
        print()

        print('self.key_attrib_index: {}'.format(self.key_attrib_index))
        print()

        print('self.class_attrib_index: {}'.format(self.class_attrib_index))
        print()

        print('self.load_train_dataset_time_taken: {}'.format(self.load_train_dataset_time_taken))
        print()

        print('self.class_int_to_name:')
        for class_int, class_name in enumerate(self.class_int_to_name):
            print('\t{} --> {}'.format(class_int, class_name))
        print()

        print('self.class_name_to_int:')
        for class_name, class_int in self.class_name_to_int.items():
            print('\t{} --> {}'.format(class_name, class_int))
        print()

        print('self.class_index_num_samples:')
        for class_int, num_samples in enumerate(self.class_index_num_samples):
            print('\t{} --> {}'.format(class_int, num_samples))
        print()


def load_config(folderpath):
    """Loads the configuration information for the dataset contained in the given folderpath in a
    dict.

    Args:
        folderpath (str): path to the dataset folder, which should contain `config.json` and
        `data.csv` files. `config.json` should have the following fields:
            - "dataset name" (str): the name of the current dataset;
            - "key attrib index" (int or null): the index containing the samples keys;
            - "class attrib index" (int): the index containing the samples classes. If negative,
                counts backwards.
            - "split char" (str): character or string used to separate columns in the data.csv file;
            - "missing value string" (str): string used to indicate a missing value for that sample
                and attribute.

    Returns:
        A dict with all the config.json key/values and also a "filepath" field, containing the
        data.csv path.
    """
    if not os.path.exists(folderpath):
        print('Folder "{}" does not exist.'.format(folderpath))
        print('Skipping this dataset.')
        return None

    config_filepath = os.path.join(folderpath, 'config.json')
    if not os.path.exists(config_filepath) or not os.path.isfile(config_filepath):
        print('"config.json" file does not exist in folder "{}".'.format(folderpath))
        print('Skipping this dataset.')
        return None

    data_filepath = os.path.join(folderpath, 'data.csv')
    if not os.path.exists(data_filepath) or not os.path.isfile(data_filepath):
        print('data.csv file does not exist in folder "{}".'.format(folderpath))
        print('Skipping this dataset.')
        return None

    print('Loading dataset configuration file for "{}".'.format(folderpath))
    with open(config_filepath, 'r') as config:
        config = json.load(config)
        mandatory_fields = ["dataset name", "key attrib index", "class attrib index", "split char",
                            "missing value string"]
        missing_fields = []
        for field in mandatory_fields:
            if field not in config:
                missing_fields.append(field)
        if missing_fields:
            print('Missing field(s) in {}:'.format(config_filepath))
            for field in missing_fields:
                print('\t{}'.format(field))
            print()
            print('Skipping this dataset.')
            return None
        if not isinstance(config["dataset name"], str):
            print('"dataset name" must be a string.')
            print('Skipping this dataset.')
            return None
        if (config["key attrib index"] is not None
                and not isinstance(config["key attrib index"], int)):
            print('"key attrib index" must have an integer value or be null.')
            print('Skipping this dataset.')
            return None
        if not isinstance(config["class attrib index"], int):
            print('"class attrib index" must have an integer value.')
            print('Skipping this dataset.')
            return None
        if not isinstance(config["split char"], str):
            print('"split char" must be a string.')
            print('Skipping this dataset.')
            return None
        if (config["missing value string"] is not None
                and not isinstance(config["missing value string"], str)):
            print('"missing value string" must be a string or null.')
            print('Skipping this dataset.')
            return None

        config["filepath"] = data_filepath
        return config


def load_all_configs(dataset_basepath):
    """Load information about every dataset available in the `dataset_basepath`.

    Returns:
        List of config dict information (see return type for `dataset.load_config`).
    """
    dataset_folders = [os.path.join(dataset_basepath, entry)
                       for entry in os.listdir(dataset_basepath)
                       if os.path.isdir(os.path.join(dataset_basepath, entry))]
    config_list = []
    for curr_folder in dataset_folders:
        config_list.append(load_config(curr_folder))
    return config_list


def load_all_datasets(datasets_configs, load_numeric):
    """Creates a Dataset object for every dataset available in the `datasets_configs` list.
    The argument `load_numeric` informs wether we should load numeric attributes or not.

    Returns:
        List of tuples (dataset_name, Dataset object).
    """
    datasets_configs.sort(key=lambda x: x["dataset name"].lower())
    datasets_list = []
    for dataset_config in datasets_configs:
        datasets_list.append((dataset_config["dataset name"],
                              Dataset(dataset_config["filepath"],
                                      dataset_config["key attrib index"],
                                      dataset_config["class attrib index"],
                                      dataset_config["split char"],
                                      dataset_config["missing value string"],
                                      load_numeric)))
    return datasets_list
