#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests.


"""

import abc
import itertools

import chol
import cvxpy as cvx
import numpy as np

#: Minimum gain allowed for Local Search methods to continue searching.
EPSILON = 0.000001


class Criterion(object):
    """Abstract base class for every criterion.
    """
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the criterion.
        """
        # returns (separation_attrib_index, splits_values, criterion_value)
        pass


#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       TWOING                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class Twoing(Criterion):
    """Twoing criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Twoing'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Twoing criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index][1])
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append((attrib_index,
                                               [best_left_values, best_right_values],
                                               best_total_gini_gain))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_twoing,
                 last_left_value,
                 first_right_value) = cls._twoing_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append((attrib_index,
                                               best_twoing,
                                               [{last_left_value}, {first_right_value}]))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @classmethod
    def _twoing_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_twoing = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                twoing_value = cls._get_twoing_value(class_num_left,
                                                     class_num_right,
                                                     num_left_samples,
                                                     num_right_samples)
                if twoing_value > best_twoing:
                    best_twoing = twoing_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_twoing, best_last_left_value, best_first_right_value)

    @staticmethod
    def _get_twoing_value(class_num_left, class_num_right, num_left_samples,
                          num_right_samples):
        sum_dif = 0.0
        for left_num, right_num in zip(class_num_left, class_num_right):
            class_num_tot = class_num_left + class_num_right
            if class_num_tot == 0:
                continue
            sum_dif += abs(left_num / num_left_samples - right_num / num_right_samples)

        num_total_samples = num_left_samples + num_right_samples
        frequency_left = num_left_samples / num_total_samples
        frequency_right = num_right_samples / num_total_samples

        twoing_value = (frequency_left * frequency_right / 4.0) * sum_dif ** 2
        return twoing_value

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                      GW SQUARED GINI                                      ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class GWSquaredGini(Criterion):
    """Square Gini criterion using Goemans and Williamson method for solving the Max Cut problem
    using a randomized approximation and a SDP formulation.
    """
    name = 'GW Squared Gini'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the GW Squared Gini
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_nominal_attrib:
                (new_to_orig_value_int,
                 new_contingency_table,
                 new_values_num_seen) = cls._remove_empty_values(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])

                (curr_cut_value,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(new_to_orig_value_int,
                                                              new_contingency_table,
                                                              new_values_num_seen)
                best_splits_per_attrib.append((attrib_index,
                                               [left_int_values, right_int_values],
                                               curr_cut_value))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, new_contingency_table,
                             new_values_num_seen):
        def _init_values_weights(new_contingency_table, new_values_num_seen):
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut).
            weights = np.zeros((new_values_num_seen.shape[0], new_values_num_seen.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(new_values_num_seen.shape[0]):
                for value_index_j in range(new_values_num_seen.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(new_contingency_table.shape[1]):
                        num_elems_value_j_diff_class = (
                            new_values_num_seen[value_index_j]
                            - new_contingency_table[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            new_contingency_table[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        weights = _init_values_weights(new_contingency_table, new_values_num_seen)
        frac_split_cholesky = cls._solve_max_cut(weights)
        left_new_values, right_new_values = cls._generate_random_partition(frac_split_cholesky)

        left_orig_values, right_orig_values = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                            left_new_values,
                                                                            right_new_values)
        cut_val = cls._calculate_split_value(left_new_values, right_new_values, weights)
        return cut_val, left_orig_values, right_orig_values


    @staticmethod
    def _solve_max_cut(weights):
        def _solve_sdp(weights):
            # See Max Cut approximation given by Goemans and Williamson, 1995.
            var = cvx.Semidef(weights.shape[0])
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(weights.shape[0]):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(weights)
        # The solution should already be symmetric, but let's just make sure the approximations
        # didn't change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        permutation_matrix, lower_triang_matrix, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that lower_triang_matrix.T is upper triangular, but
        # frac_split_cholesky = np.dot(lower_triang_matrix.T, permutation_matrix)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(lower_triang_matrix.T, permutation_matrix)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky):
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)

        left_new_values = set()
        right_new_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_new_values.add(new_value)
            else:
                right_new_values.add(new_value)
        return left_new_values, right_new_values

    @staticmethod
    def _get_split_in_orig_values(new_to_orig_value_int, left_new_values, right_new_values):
        # Let's get the original values on each side of this partition
        left_orig_values = set(new_to_orig_value_int[left_new_value]
                               for left_new_value in left_new_values)
        right_orig_values = set(new_to_orig_value_int[right_new_value]
                                for right_new_value in right_new_values)
        return left_orig_values, right_orig_values

    @staticmethod
    def _calculate_split_value(left_new_values, right_new_values, weights):
        cut_val = 0.0
        for value_left, value_right in itertools.product(left_new_values, right_new_values):
            cut_val += weights[value_left, value_right]
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GW CHI SQUARE                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GWChiSquare(Criterion):
    """Chi Square criterion using Goemans and Williamson method for solving the Max Cut problem
    using a randomized approximation and a SDP formulation.
    """
    name = 'GW Chi Square'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the GW Chi Square criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_nominal_attrib:
                (new_to_orig_value_int,
                 new_contingency_table,
                 new_values_num_seen) = cls._remove_empty_values(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])

                (curr_cut_value,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(new_to_orig_value_int,
                                                              new_contingency_table,
                                                              new_values_num_seen)
                best_splits_per_attrib.append((attrib_index,
                                               [left_int_values, right_int_values],
                                               curr_cut_value))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, new_contingency_table,
                             new_values_num_seen):
        def _init_values_weights(new_values_num_seen, new_contingency_table):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((new_values_num_seen.shape[0], new_values_num_seen.shape[0]),
                               dtype=np.float64)
            for value_index_i, num_samples_value_index_i in enumerate(new_values_num_seen):
                for value_index_j, num_samples_value_index_j in enumerate(new_values_num_seen):
                    if value_index_i >= value_index_j:
                        continue

                    # Let's calculate the weight of the (i,j)-th edge using the chi-square value.
                    num_samples_both_values = (num_samples_value_index_i
                                               + num_samples_value_index_j) # is always > 0.
                    for curr_class_index in range(new_contingency_table.shape[1]):
                        num_samples_both_values_curr_class = (
                            new_contingency_table[value_index_i, curr_class_index]
                            + new_contingency_table[value_index_j, curr_class_index])
                        if num_samples_both_values_curr_class == 0:
                            continue

                        expected_value_index_i_class = (
                            num_samples_value_index_i * num_samples_both_values_curr_class
                            / num_samples_both_values)
                        diff_index_i = (
                            new_contingency_table[value_index_i, curr_class_index]
                            - expected_value_index_i_class)

                        expected_value_index_j_class = (
                            num_samples_value_index_j * num_samples_both_values_curr_class
                            / num_samples_both_values)
                        diff_index_j = (
                            new_contingency_table[value_index_j, curr_class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))
                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                    if new_values_num_seen.shape[0] > 2:
                        weights[value_index_i, value_index_j] /= (new_values_num_seen.shape[0] - 1.)
                    weights[value_index_j, value_index_i] = weights[value_index_i, value_index_j]
            return weights

        weights = _init_values_weights(new_contingency_table, new_values_num_seen)
        frac_split_cholesky = cls._solve_max_cut(weights)
        left_new_values, right_new_values = cls._generate_random_partition(frac_split_cholesky)

        left_orig_values, right_orig_values = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                            left_new_values,
                                                                            right_new_values)
        cut_val = cls._calculate_split_value(left_new_values, right_new_values, weights)
        return cut_val, left_orig_values, right_orig_values

    @staticmethod
    def _solve_max_cut(weights):
        def _solve_sdp(weights):
            # See Max Cut approximation given by Goemans and Williamson, 1995.
            var = cvx.Semidef(weights.shape[0])
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(weights.shape[0]):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(weights)
        # The solution should already be symmetric, but let's just make sure the approximations
        # didn't change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        permutation_matrix, lower_triang_matrix, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that lower_triang_matrix.T is upper triangular, but
        # frac_split_cholesky = np.dot(lower_triang_matrix.T, permutation_matrix)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(lower_triang_matrix.T, permutation_matrix)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky):
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)

        left_new_values = set()
        right_new_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_new_values.add(new_value)
            else:
                right_new_values.add(new_value)
        return left_new_values, right_new_values

    @staticmethod
    def _get_split_in_orig_values(new_to_orig_value_int, left_new_values, right_new_values):
        # Let's get the original values on each side of this partition
        left_orig_values = set(new_to_orig_value_int[left_new_value]
                               for left_new_value in left_new_values)
        right_orig_values = set(new_to_orig_value_int[right_new_value]
                                for right_new_value in right_new_values)
        return left_orig_values, right_orig_values

    @staticmethod
    def _calculate_split_value(left_new_values, right_new_values, weights):
        cut_val = 0.0
        for value_left, value_right in itertools.product(left_new_values, right_new_values):
            cut_val += weights[value_left, value_right]
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                     LS Squared Gini                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class LSSquaredGini(Criterion):
    """Squared Gini criterion using a greedy local search for solving the Max Cut problem.
    """
    name = 'LS Squared Gini'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the LS Squared Gini
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (new_to_orig_value_int,
                 new_contingency_table,
                 new_values_num_seen) = cls._remove_empty_values(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                (curr_cut_value,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(new_to_orig_value_int,
                                                              new_contingency_table,
                                                              new_values_num_seen)
                best_splits_per_attrib.append((attrib_index,
                                               [left_int_values, right_int_values],
                                               curr_cut_value))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (cut_val,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append((attrib_index,
                                               cut_val,
                                               [{last_left_value}, {first_right_value}]))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, new_contingency_table,
                             new_values_num_seen):
        def _init_values_weights(new_contingency_table, new_values_num_seen):
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut).
            weights = np.zeros((new_values_num_seen.shape[0], new_values_num_seen.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(new_values_num_seen.shape[0]):
                for value_index_j in range(new_values_num_seen.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(new_contingency_table.shape[1]):
                        num_elems_value_j_diff_class = (
                            new_values_num_seen[value_index_j]
                            - new_contingency_table[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            new_contingency_table[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights


        weights = _init_values_weights(new_contingency_table, new_values_num_seen)
        # Initial partition generated through a greedy approach.
        (cut_val,
         left_new_values,
         right_new_values) = cls._generate_initial_partition(len(new_values_num_seen), weights)
        # Look for a better solution locally, changing the side of a single node or exchanging a
        # pair of nodes from different sides, while it increases the cut value.
        (cut_val_switched,
         left_new_values_switched,
         right_new_values_switched) = cls._switch_while_increase(cut_val,
                                                                 left_new_values,
                                                                 right_new_values,
                                                                 weights)
        if cut_val_switched > cut_val:
            cut_val = cut_val_switched
            (left_orig_values,
             right_orig_values) = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                left_new_values_switched,
                                                                right_new_values_switched)
        else:
            (left_orig_values,
             right_orig_values) = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                left_new_values,
                                                                right_new_values)
        return cut_val, left_orig_values, right_orig_values

    @classmethod
    def _generate_initial_partition(cls, num_values, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        for value in range(num_values):
            if not set_left_values: # first node goes to the left
                set_left_values.add(value)
                continue
            gain_assigning_right = sum(weights[value][left_value]
                                       for left_value in set_left_values)
            gain_assigning_left = sum(weights[value][right_value]
                                      for right_value in set_right_values)
            if gain_assigning_right >= gain_assigning_left:
                set_right_values.add(value)
                cut_val += gain_assigning_right
            else:
                set_left_values.add(value)
                cut_val += gain_assigning_left
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        found_improvement = True
        while found_improvement:
            found_improvement = False

            # Try to switch the side of a single node (`value`) to improve the cut value.
            for value in values_seen:
                new_cut_val = cls._split_gain_for_single_switch(curr_cut_val,
                                                                set_left_values,
                                                                set_right_values,
                                                                value,
                                                                weights)
                if new_cut_val - curr_cut_val >= EPSILON:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    found_improvement = True
                    break
            if found_improvement:
                continue

            # Try to switch a pair of nodes (`value1` and `value2`) from different sides to improve
            # the cut value.
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._split_gain_for_double_switch(curr_cut_val,
                                                                set_left_values,
                                                                set_right_values,
                                                                (value1, value2),
                                                                weights)
                if new_cut_val - curr_cut_val >= EPSILON:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    found_improvement = True
                    break
        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _split_gain_for_single_switch(curr_gain, left_new_values, right_new_values,
                                      new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in left_new_values:
            for value in left_new_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in right_new_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in left_new_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in right_new_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _split_gain_for_double_switch(curr_gain, left_new_values, right_new_values,
                                      new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in left_new_values:
            for value in left_new_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in right_new_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in left_new_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in right_new_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _get_split_in_orig_values(new_to_orig_value_int, left_new_values, right_new_values):
        # Let's get the original values on each side of this partition
        left_orig_values = set(new_to_orig_value_int[left_new_value]
                               for left_new_value in left_new_values)
        right_orig_values = set(new_to_orig_value_int[right_new_value]
                                for right_new_value in right_new_values)
        return left_orig_values, right_orig_values

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _best_cut_for_numeric(cls, sorted_values_and_classes, num_classes):
        # Initial state is having the first value of `sorted_values_and_classes` on the left and
        # everything else on the right.
        last_left_new_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        # Note that this cut with only the first sample on the left might not be valid: the value on
        # the left might also appears on the right of the split. Therefore we initialize with cut
        # value = -inf and only check if the current split is valid (and maybe update the
        # information about the best cut found) on the next loop iteration. Note that, by doing
        # this, we never test the split where the last sample is in the left, because there would be
        # no samples on the right.
        best_cut_value = float('-inf')
        best_last_left_new_value = None
        best_first_right_new_value = None

        # `curr_cut_value` holds the current cut value, even if it's not a valid cut.
        curr_cut_value = num_right_samples - class_num_right[last_left_class]

        for (first_right_new_value, first_right_class) in sorted_values_and_classes[1:]:
            if first_right_new_value != last_left_new_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_new_value = last_left_new_value
                best_first_right_new_value = first_right_new_value

            curr_cut_value -= num_left_samples - class_num_left[first_right_class]
            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
            curr_cut_value += num_right_samples - class_num_right[first_right_class]
            last_left_new_value = first_right_new_value

        return (best_cut_value, best_last_left_new_value, best_first_right_new_value)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       LS Chi Square                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class LSChiSquare(Criterion):
    """Chi Square criterion using a greedy local search for solving the Max Cut problem.
    """
    name = 'LS Chi Square'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the LS Chi Square criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (new_to_orig_value_int,
                 new_contingency_table,
                 new_values_num_seen) = cls._remove_empty_values(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                (curr_cut_value,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(new_to_orig_value_int,
                                                              new_contingency_table,
                                                              new_values_num_seen)
                best_splits_per_attrib.append((attrib_index,
                                               [left_int_values, right_int_values],
                                               curr_cut_value))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (cut_val,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes,
                     tree_node.class_index_num_samples)
                best_splits_per_attrib.append((attrib_index,
                                               cut_val,
                                               [{last_left_value}, {first_right_value}]))

        best_attribute_and_split = (None, [], float('-inf'))
        for curr_attrib_split in best_splits_per_attrib:
            if curr_attrib_split[2] > best_attribute_and_split[2]:
                best_attribute_and_split = curr_attrib_split
        return best_attribute_and_split

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, new_contingency_table,
                             new_values_num_seen):
        def _init_values_weights(new_values_num_seen, new_contingency_table):
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((new_values_num_seen.shape[0], new_values_num_seen.shape[0]),
                               dtype=np.float64)
            for value_index_i, num_samples_value_index_i in enumerate(new_values_num_seen):
                for value_index_j, num_samples_value_index_j in enumerate(new_values_num_seen):
                    if value_index_i >= value_index_j:
                        continue
                    num_samples_both_values = (num_samples_value_index_i
                                               + num_samples_value_index_j) # is always > 0.
                    for curr_class_index in range(new_contingency_table.shape[1]):
                        num_samples_both_values_curr_class = (
                            new_contingency_table[value_index_i, curr_class_index]
                            + new_contingency_table[value_index_j, curr_class_index])
                        if num_samples_both_values_curr_class == 0:
                            continue

                        expected_value_index_i_class = (
                            num_samples_value_index_i * num_samples_both_values_curr_class
                            / num_samples_both_values)
                        diff_index_i = (
                            new_contingency_table[value_index_i, curr_class_index]
                            - expected_value_index_i_class)

                        expected_value_index_j_class = (
                            num_samples_value_index_j * num_samples_both_values_curr_class
                            / num_samples_both_values)
                        diff_index_j = (
                            new_contingency_table[value_index_j, curr_class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))
                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                    if new_values_num_seen.shape[0] > 2:
                        weights[value_index_i, value_index_j] /= (new_values_num_seen.shape[0] - 1.)
                    weights[value_index_j, value_index_i] = weights[value_index_i, value_index_j]
            return weights


        weights = _init_values_weights(new_contingency_table, new_values_num_seen)
        # Initial partition generated through a greedy approach.
        (cut_val,
         left_new_values,
         right_new_values) = cls._generate_initial_partition(len(new_values_num_seen), weights)
        # Look for a better solution locally, changing the side of a single node or exchanging a
        # pair of nodes from different sides, while it increases the cut value.
        (cut_val_switched,
         left_new_values_switched,
         right_new_values_switched) = cls._switch_while_increase(cut_val,
                                                                 left_new_values,
                                                                 right_new_values,
                                                                 weights)
        if cut_val_switched > cut_val:
            cut_val = cut_val_switched
            (left_orig_values,
             right_orig_values) = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                left_new_values_switched,
                                                                right_new_values_switched)
        else:
            (left_orig_values,
             right_orig_values) = cls._get_split_in_orig_values(new_to_orig_value_int,
                                                                left_new_values,
                                                                right_new_values)
        return cut_val, left_orig_values, right_orig_values

    @classmethod
    def _generate_initial_partition(cls, num_values, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        for value in range(num_values):
            if not set_left_values: # first node goes to the left
                set_left_values.add(value)
                continue
            gain_assigning_right = sum(weights[value][left_value]
                                       for left_value in set_left_values)
            gain_assigning_left = sum(weights[value][right_value]
                                      for right_value in set_right_values)
            if gain_assigning_right >= gain_assigning_left:
                set_right_values.add(value)
                cut_val += gain_assigning_right
            else:
                set_left_values.add(value)
                cut_val += gain_assigning_left
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        found_improvement = True
        while found_improvement:
            found_improvement = False

            # Try to switch the side of a single node (`value`) to improve the cut value.
            for value in values_seen:
                new_cut_val = cls._split_gain_for_single_switch(curr_cut_val,
                                                                set_left_values,
                                                                set_right_values,
                                                                value,
                                                                weights)
                if new_cut_val - curr_cut_val >= EPSILON:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    found_improvement = True
                    break
            if found_improvement:
                continue

            # Try to switch a pair of nodes (`value1` and `value2`) from different sides to improve
            # the cut value.
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._split_gain_for_double_switch(curr_cut_val,
                                                                set_left_values,
                                                                set_right_values,
                                                                (value1, value2),
                                                                weights)
                if new_cut_val - curr_cut_val >= EPSILON:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    found_improvement = True
                    break
        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _split_gain_for_single_switch(curr_gain, left_new_values, right_new_values,
                                      new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in left_new_values:
            for value in left_new_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in right_new_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in left_new_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in right_new_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _split_gain_for_double_switch(curr_gain, left_new_values, right_new_values,
                                      new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in left_new_values:
            for value in left_new_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in right_new_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in left_new_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in right_new_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _get_split_in_orig_values(new_to_orig_value_int, left_new_values, right_new_values):
        # Let's get the original values on each side of this partition
        left_orig_values = set(new_to_orig_value_int[left_new_value]
                               for left_new_value in left_new_values)
        right_orig_values = set(new_to_orig_value_int[right_new_value]
                                for right_new_value in right_new_values)
        return left_orig_values, right_orig_values

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _best_cut_for_numeric(cls, sorted_values_and_classes, num_classes, class_index_num_samples):
        # Initial state is having the first value of `sorted_values_and_classes` on the left and
        # everything else on the right.
        last_left_new_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1
        num_samples = len(sorted_values_and_classes)

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        # Note that this cut with only the first sample on the left might not be valid: the value on
        # the left might also appears on the right of the split. Therefore we initialize with cut
        # value = -inf and only check if the current split is valid (and maybe update the
        # information about the best cut found) on the next loop iteration. Note that, by doing
        # this, we never test the split where the last sample is in the left, because there would be
        # no samples on the right.
        best_cut_value = float('-inf')
        best_last_left_new_value = None
        best_first_right_new_value = None

        for (first_right_new_value, first_right_class) in sorted_values_and_classes[1:]:
            # `curr_cut_value` holds the current cut value, even if it's not a valid cut.
            curr_cut_value = 0.0
            for class_index in range(num_classes):
                if class_index_num_samples[class_index] != 0:
                    expected_value_left_class = (
                        num_left_samples * class_index_num_samples[class_index] / num_samples)
                    diff_left = class_num_left[class_index] - expected_value_left_class
                    curr_cut_value += diff_left * (diff_left / expected_value_left_class)

                    expected_value_right_class = (
                        num_right_samples * class_index_num_samples[class_index] / num_samples)
                    diff_right = class_num_right[class_index] - expected_value_right_class
                    curr_cut_value += diff_right * (diff_right / expected_value_right_class)

            if first_right_new_value != last_left_new_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_new_value = last_left_new_value
                best_first_right_new_value = first_right_new_value
                last_left_new_value = first_right_new_value

            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1

        return (best_cut_value, best_last_left_new_value, best_first_right_new_value)
