import pulp as pl
import numpy as np
import scipy.sparse as sp
import math
import time

solver_name = "COPT"
# solver_name = 'GUROBI'
# solver_name = 'HiGHS'
# solver_name = 'COIN_CMD'

# def __eq__(self, other):
#     if isinstance(other, (float, int)):
#         return pl.LpConstraint(self, pl.const.LpConstraintEQ, rhs=other)
#     else:
#         return pl.LpConstraint(self - other, pl.const.LpConstraintEQ)

# pl.LpAffineExpression.__eq__ = __eq__


def fba(lb, ub, S, c):
    """A Python function to perform fba using scipy.optimize LP solver `linprog`
    Returns an optimal solution and its value for the following linear program:
    max c*v, subject to,
    Sv = 0, lb <= v <= ub

    Keyword arguments:
    lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
    ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the linear objective function, i.e., a n-dimensional vector
    """

    # print("fbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

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

        model = pl.LpProblem("FBA", pl.LpMaximize)

        x = [pl.LpVariable(f"x{i}", lowBound=lb[i], upBound=ub[i]) for i in range(n)]

        for i in range(m):
            # model += (pl.lpDot(S[i], x) == 0, f"Mass balance {i}")
            # model += (pl.lpSum([S[i][j] * x[j] for j in range(n)]) == 0, f"Mass balance {i}")
            model += (
                pl.LpAffineExpression(
                    (x[j], S[i][j]) for j in range(n) if abs(S[i][j]) > 0.01
                )
                == 0,
                f"Mass balance {i}",
            )

        # model += (pl.lpDot(c,x), "Objective function")
        # model += (pl.lpSum([c[j] * x[j] for j in range(n)]), "Objective function")
        model += (
            pl.LpAffineExpression((x[j], c[j]) for j in range(n) if abs(c[j]) > 0.01),
            "Objective function",
        )
        # model.writeMPS("test.mps")
        solver = pl.getSolver(solver_name, msg=0)
        model.solve(solver)

        status = pl.LpStatus[model.status]

        # If optimized
        if status == "Optimal":
            optimum_value = pl.value(model.objective)
            optimum_sol = np.array([pl.value(x[i]) for i in range(n)])

        return optimum_sol, optimum_value

    except AttributeError:
        print("PuLP failed.")


def fva(lb, ub, S, c, opt_percentage=100):
    """A Python function to perform fva using gurobi LP solver
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

    # TODO
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]

    max_biomass_flux_vector, max_biomass_objective = fba(lb, ub, S, c)

    min_fluxes = []
    max_fluxes = []
    vopt = (opt_percentage / 100) * tol * math.floor(max_biomass_objective / tol)

    # try:

    model = pl.LpProblem("FVA", pl.LpMinimize)

    # Create variables
    x = [pl.LpVariable(f"x{i}", lowBound=lb[i], upBound=ub[i]) for i in range(n)]

    # Add the constraints
    # s = time.time()
    for i in range(m):
        # model += (pl.lpSum([S[i][j] * x[j] for j in range(n)]) == 0, f"Mass balance {i}")
        # model += (pl.lpDot(S[i], x) == 0, f"Mass balance {i}")
        # model += (pl.LpAffineExpression((x[j],S[i][j]) for j in range(n)) == 0, f"Mass balance {i}")
        model += (
            pl.LpAffineExpression(
                (x[j], S[i][j]) for j in range(n) if abs(S[i][j]) > 0.01
            )
            == 0,
            f"Mass balance {i}",
        )
    # t = time.time() - s
    # print("Time to add constraints: ", t)

    # model += pl.lpDot(c, x) >= vopt
    # model += (pl.lpSum([c[j] * x[j] for j in range(n)]) >= vopt, "Objective function")
    model += (
        pl.LpAffineExpression((x[j], c[j]) for j in range(n) if abs(c[j]) > 0.01)
        >= vopt
    )

    try:
        for i in range(n):
            model.objective = x[i]

            # Optimize model
            solver = pl.getSolver(solver_name, msg=0)
            model.solve(solver)

            status = pl.LpStatus[model.status]

            # If optimized
            if status == "Optimal":

                # Get the min objective value
                min_objective = pl.value(model.objective)
                min_fluxes.append(min_objective)
            else:
                min_fluxes.append(lb[i])

            # Likewise, for the maximum

            model.objective = -x[i]

            # Optimize model
            solver = pl.getSolver(solver_name, msg=0)
            model.solve(solver)

            status = pl.LpStatus[model.status]

            # Again if optimized
            if status == "Optimal":
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

    except pl.PulpError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except AttributeError:
        print("PuLP failed.")


def inner_ball(A, b):
    """A Python function to compute the maximum inscribed ball in the given polytope using gurobi LP solver
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
        entry = np.linalg.norm(A[i,])
        extra_column.append(entry)

    column = np.asarray(extra_column)
    A_expand = np.c_[A, column]

    model = pl.LpProblem("Inner Ball", pl.LpMaximize)

    # Create variables
    x = [pl.LpVariable(f"x{i}") for i in range(n + 1)]

    for i in range(m):
        model += (
            pl.LpAffineExpression((x[j], A_expand[i][j]) for j in range(n + 1)) <= b[i]
        )

    model += x[n]

    solver = pl.getSolver(solver_name, msg=0)
    model.solve(solver)

    point = []
    for i in range(n):
        point.append(pl.value(x[i]))

    r = pl.value(x[n])

    if r < 0:
        print(
            "The radius calculated has negative value. The polytope is infeasible or something went wrong with the solver"
        )
    else:
        return point, r
    
    
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

    # declare the tolerance that gurobi works properly (we found it experimentally)
    redundant_facet_tol = 1e-07
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)
    Aeq_res = S

    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")

    # call fba to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = np.empty((0, n), float)
    beq_res = np.array(beq)
    
    


