# dingo : a python library for metabolic networks sampling and analysis
# dingo is part of GeomScale project

# Copyright (c) 2024 Ke Shi

# Licensed under GNU LGPL.3, see LICENCE file

import pulp as pl
import numpy as np
import math
import sys

solver_name = "HiGHS"

def dot(c, x):
    """A function to get the dot product of variables vector x and coefficients vector c
    faster than pl.lpDot and pl.lpSum
    """
    return pl.LpAffineExpression(
        (x[j], c[j]) for j in range(len(c)) if abs(c[j]) > 1e-10
    )


def fba(lb, ub, S, c):
    """A Python function to perform fba using PuLP LP modeler
    Returns an optimal solution and its value for the following linear program:
    max c*v, subject to,
    Sv = 0, lb <= v <= ub

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the linear objective function, i.e., a n-dimensional vector
    """

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )
    if c.size != S.shape[1]:
        raise Exception(
            "The length of the lineart objective function must be equal to the number of reactions."
        )

    m = S.shape[0]
    n = S.shape[1]
    optimum_value = 0
    optimum_sol = np.zeros(n)

    try:
        # Create a model
        model = pl.LpProblem("FBA", pl.LpMaximize)

        # Create variables and set lb <= v <= ub
        v = [pl.LpVariable(f"v{i}", lowBound=lb[i], upBound=ub[i]) for i in range(n)]

        # Add the constraints Sv = 0
        for i in range(m):
            model += dot(S[i], v) == 0

        # Set the objective function
        model += dot(c, v)

        # Optimize model
        solver = pl.getSolver(solver_name, msg=0)
        model.solve(solver)

        # If optimized
        status = model.status
        if status == pl.LpStatusOptimal:
            optimum_value = pl.value(model.objective)
            optimum_sol = np.array([v[i].value() for i in range(n)])

        return optimum_sol, optimum_value

    except pl.PulpSolverError as e:
        print(f"Solver Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def fva(lb, ub, S, c, opt_percentage=100):
    """A Python function to perform fva using PuLP LP modeler
    Returns the value of the optimal solution for all the following linear programs:
    min/max v_i, for all coordinates i=1,...,n, subject to,
    Sv = 0, lb <= v <= ub

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    opt_percentage -- consider solutions that give you at least a certain
                      percentage of the optimal solution (default is to consider
                      optimal solutions only)
    """

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # declare the tolerance that gurobi and highs works properly (we found it experimentally)
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]

    max_biomass_flux_vector, max_biomass_objective = fba(lb, ub, S, c)

    min_fluxes = []
    max_fluxes = []

    adjusted_opt_threshold = (
        (opt_percentage / 100) * tol * math.floor(max_biomass_objective / tol)
    )

    try:
        # Create a model
        model = pl.LpProblem("FVA", pl.LpMinimize)

        # Create variables and set lb <= v <= ub
        v = [pl.LpVariable(f"v{i}", lowBound=lb[i], upBound=ub[i]) for i in range(n)]

        # Add the constraints Sv = 0
        for i in range(m):
            model += dot(S[i], v) == 0

        # add an additional constraint to impose solutions with at least `opt_percentage` of the optimal solution
        model += dot(c, v) >= adjusted_opt_threshold

        for i in range(n):
            # Set the objective function
            model.objective = v[i]

            # Optimize model
            solver = pl.getSolver(solver_name, msg=0)
            model.solve(solver)

            # If optimized
            if model.status == pl.LpStatusOptimal:
                # Get the min objective value
                min_objective = pl.value(model.objective)
                min_fluxes.append(min_objective)
            else:
                min_fluxes.append(lb[i])

            # Likewise, for the maximum
            model.objective = -v[i]

            # Optimize model
            solver = pl.getSolver(solver_name, msg=0)
            model.solve(solver)

            # Again if optimized
            if model.status == pl.LpStatusOptimal:
                # Get the max objective value
                max_objective = -pl.value(model.objective)
                max_fluxes.append(max_objective)
            else:
                max_fluxes.append(ub[i])

            # Make lists of fluxes numpy arrays
        min_fluxes = np.asarray(min_fluxes)
        max_fluxes = np.asarray(max_fluxes)

        return (
            min_fluxes,
            max_fluxes,
            max_biomass_flux_vector,
            max_biomass_objective,
        )

    except pl.PulpSolverError as e:
        print(f"Solver Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def inner_ball(A, b):
    """A Python function to compute the maximum inscribed ball in the given polytope using PuLP LP modeler
    Returns the optimal solution for the following linear program:
    max r, subject to,
    a_ix + r||a_i|| <= b, i=1,...,n

    Keyword arguments:
    A -- an mxn matrix that contains the normal vectors of the facets of the polytope row-wise
    b -- a m-dimensional vector
    """

    extra_column = []

    m = A.shape[0]
    n = A.shape[1]

    for i in range(A.shape[0]):
        entry = np.linalg.norm(A[i])
        extra_column.append(entry)

    column = np.asarray(extra_column)
    A_expand = np.c_[A, column]

    model = pl.LpProblem("Inner_Ball", pl.LpMaximize)

    # Create variables where x[n] is the radius
    x = [pl.LpVariable(f"x{i}") for i in range(n + 1)]

    # Add the constraints a_ix + r||a_i|| <= b
    for i in range(m):
        model += dot(A_expand[i], x) <= b[i]

    # Set the objective function
    model += x[n]

    # Optimize model
    solver = pl.getSolver(solver_name, msg=0)
    model.solve(solver)

    # Get the center point and the radius of max ball from the solution of LP
    point = [x[i].value() for i in range(n)]
    # its last element is the radius
    r = x[n].value()

    # And check whether the computed radius is negative
    if r < 0:
        print(
            "The radius calculated has negative value. The polytope is infeasible or something went wrong with the solver"
        )
    else:
        return point, r


def set_model(x, lb, ub,  Aeq, beq, A, b):
    """
    A helper function of remove_redundant_facets function
    Create a PuLP model with given PuLP variables, bounds, equality constraints, and inequality constraints
    but without an objective function.
    """

    model = pl.LpProblem()

    # Set the bounds for the variables
    for i in range(len(x)):
        x[i].lowBound = lb[i]
        x[i].upBound = ub[i]

    # Add the equality constraints
    for i in range(Aeq.shape[0]):
        model += dot(Aeq[i], x) == beq[i]

    # Add the inequality constraints
    for i in range(A.shape[0]):
        model += dot(A[i], x) <= b[i]

    return model


def remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    """A function to find and remove the redundant facets and to find
    the facets with very small offset and to set them as equalities

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    opt_percentage -- consider solutions that give you at least a certain
                      percentage of the optimal solution (default is to consider
                      optimal solutions only)
    """

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # TODO
    # declare the tolerance that gurobi works properly
    redundant_facet_tol = 1e-07
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]

    # [v,-v] <= [ub,-lb]
    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0)
    b = np.ascontiguousarray(b, dtype="float")

    beq = np.zeros(m)

    Aeq_res = S
    beq_res = np.array(beq)
    b_res = []
    A_res = np.empty((0, n), float)

    max_biomass_flux_vector, max_biomass_objective = fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    try:
        # Create variables for the model
        v = [pl.LpVariable(f"v{i}") for i in range(n)]

        # initialize
        indices_iter = range(n)
        removed = 1
        offset = 1
        facet_left_removed = np.zeros(n, dtype=bool)
        facet_right_removed = np.zeros(n, dtype=bool)

        # Loop until no redundant facets are found
        while removed > 0 or offset > 0:
            removed = 0
            offset = 0
            indices = indices_iter
            indices_iter = []

            Aeq = np.array(Aeq_res)
            beq = np.array(beq_res)

            A_res = np.empty((0, n), dtype=float)
            b_res = []

            model_iter = set_model(v, lb, ub, Aeq, beq, np.array([-c]), [val])

            for i in indices:

                redundant_facet_right = True
                redundant_facet_left = True

                for j in range(n):
                    v[j].lowBound = lb[j]
                    v[j].upBound = ub[j]

                # maximize v_i (right)
                model_iter.objective = -v[i]

                solver = pl.getSolver(solver_name, msg=0)
                model_iter.solve(solver)

                # if optimized
                if model_iter.status == pl.LpStatusOptimal:
                    # get the maximum objective value
                    max_objective = -pl.value(model_iter.objective)
                else:
                    max_objective = ub[i]

                # if this facet was not removed in a previous iteration
                if not facet_right_removed[i]:
                    # Relax the inequality
                    v[i].upBound = ub[i] + 1
                    
                    solver = pl.getSolver(solver_name, msg=0)
                    model_iter.solve(solver)
                    
                    # Reset the inequality
                    v[i].upBound = ub[i]

                    if model_iter.status == pl.LpStatusOptimal:
                        # Get the max objective value with relaxed inequality
                        max_objective2 = -pl.value(model_iter.objective)
                        if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                            redundant_facet_right = False
                        else:
                            removed += 1
                            facet_right_removed[i] = True

                # minimum v_i (left)
                model_iter.objective = v[i]
                solver = pl.getSolver(solver_name, msg=0)
                model_iter.solve(solver)

                # If optimized
                if model_iter.status == pl.LpStatusOptimal:
                    # Get the min objective value
                    min_objective = pl.value(model_iter.objective)
                else:
                    min_objective = lb[i]

                # if this facet was not removed in a previous iteration
                if not facet_left_removed[i]:
                    # Relax the inequality
                    v[i].lowBound = lb[i] - 1
                    
                    solver = pl.getSolver(solver_name, msg=0)
                    model_iter.solve(solver)
                    
                    # Reset the inequality
                    v[i].lowBound = lb[i]
                    
                    if model_iter.status == pl.LpStatusOptimal:
                        # Get the min objective value with relaxed inequality
                        min_objective2 = pl.value(model_iter.objective)
                        if (
                            np.abs(min_objective2 - min_objective) 
                            > redundant_facet_tol
                        ):
                            redundant_facet_left = False
                        else:
                            removed += 1
                            facet_left_removed[i] = True

                if (not redundant_facet_left) or (not redundant_facet_right):
                    width = abs(max_objective - min_objective)

                    # Check whether the offset in this dimension is small (and set an equality)
                    if width < redundant_facet_tol:
                        offset += 1
                        Aeq_res = np.vstack((Aeq_res, A[i]))
                        beq_res = np.append(beq_res, min(max_objective, min_objective))
                        # Remove the bounds on this dimension
                        ub[i] = sys.float_info.max
                        lb[i] = -sys.float_info.max
                    else:
                        # store this dimension
                        indices_iter.append(i)

                        if not redundant_facet_left:
                            # Not a redundant inequality
                            A_res = np.append(A_res, np.array([A[n + i]]), axis=0)
                            b_res.append(b[n + i])
                        else:
                            lb[i] = -sys.float_info.max

                        if not redundant_facet_right:
                            # Not a redundant inequality
                            A_res = np.append(A_res, np.array([A[i]]), axis=0)
                            b_res.append(b[i])
                        else:
                            ub[i] = sys.float_info.max
                else:
                    # Remove the bounds on this dimension
                    ub[i] = sys.float_info.max
                    lb[i] = -sys.float_info.max

        b_res = np.asarray(b_res, dtype="float")
        A_res = np.asarray(A_res, dtype="float")
        A_res = np.ascontiguousarray(A_res, dtype="float")

        return A_res, b_res, Aeq_res, beq_res

    except pl.PulpSolverError as e:
        print(f"Solver Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
