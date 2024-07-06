# dingo : a python library for metabolic networks sampling and analysis
# dingo is part of GeomScale project

# Copyright (c) 2024 Ke Shi

# Licensed under GNU LGPL.3, see LICENCE file

import unittest
import os
import numpy as np
from dingo import MetabolicNetwork, PolytopeSampler
from dingo.pulp_based_impl import fba, fva, inner_ball, remove_redundant_facets

class TestFastMethods(unittest.TestCase):
    def test_inner_ball(self):

        m = 2
        n = 5

        A = np.zeros((2 * n, n), dtype="float")
        A[0:n] = np.eye(n)
        A[n:] -= np.eye(n, n, dtype="float")
        b = np.ones(2 * n, dtype="float")

        max_ball = inner_ball(A, b)

        self.assertTrue(abs(max_ball[1] - 1) < 1e-08)

    def test_fva(self):

        current_directory = os.getcwd()
        input_file_json = current_directory + "/ext_data/e_coli_core.json"

        model = MetabolicNetwork.from_json(input_file_json)

        res = fva(
            model.lb,
            model.ub,
            model.S,
            model.objective_function,
            model.parameters["opt_percentage"],
        )

        self.assertTrue(abs(res[3] - 0.8739215069684305) < 1e-08)
        self.assertEqual(res[0].size, 95)
        self.assertEqual(res[1].size, 95)

    def test_remove_redundant_facets(self):
        current_directory = os.getcwd()
        input_file_json = current_directory + "/ext_data/e_coli_core.json"

        model = MetabolicNetwork.from_json(input_file_json)

        A, b, Aeq, beq = remove_redundant_facets(
            model.lb,
            model.ub,
            model.S,
            model.objective_function,
            model.parameters["opt_percentage"],
        )

        self.assertEqual(A.shape[0], 25)
        self.assertEqual(A.shape[1], 95)
        self.assertEqual(Aeq.shape[0], 76)
        self.assertEqual(Aeq.shape[1], 95)

    def test_fast_fba(self):

        current_directory = os.getcwd()
        input_file_json = current_directory + "/ext_data/e_coli_core.json"

        model = MetabolicNetwork.from_json(input_file_json)

        res = fba(model._lb, model._ub, model._S, model._objective_function)

        self.assertTrue(abs(res[1] - 0.8739215069684305) < 1e-08)


if __name__ == "__main__":
    unittest.main()
