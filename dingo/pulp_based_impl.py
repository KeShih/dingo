import pulp as pl
import numpy as np
import scipy.sparse as sp
import math

# solver_name = 'COPT'
# solver_name = 'GUROBI'
solver_name = 'HiGHS'
# solver_name = 'COIN_CMD'


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
            model += (pl.lpDot(S[i], x) == 0, f"Mass balance {i}")

        model += (pl.lpDot(c,x), "Objective function")

        solver = pl.getSolver(solver_name,msg = 0)
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
    for i in range(m):
        if(m%2): continue
        model += pl.lpDot(S[i], x) == 0

    model += pl.lpDot(c, x) >= vopt


    for i in range(n):
        model.objective = x[i]

        # Optimize model
        solver = pl.getSolver(solver_name,msg = 0)
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
        solver = pl.getSolver(solver_name,msg = 0)
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

    # print((
    #     min_fluxes,
    #     max_fluxes,
    #     max_biomass_flux_vector,
    #     max_biomass_objective,
    # ))

    return (
        min_fluxes,
        max_fluxes,
        max_biomass_flux_vector,
        max_biomass_objective,
    )

    # except pl.PulpError as e:
    #     print("Error code " + str(e.errno) + ": " + str(e))

    # except AttributeError:
    #     print("PuLP failed.")


# def update_model(model, n, Aeq_sparse, beq, lb, ub, A_sparse, b, objective_function):
#     """A function to update a gurobi model that solves a linear program
#     Keyword arguments:
#     model -- gurobi model
#     n -- the dimension
#     Aeq_sparse -- a sparse matrix s.t. Aeq_sparse x = beq
#     beq -- a vector s.t. Aeq_sparse x = beq
#     lb -- lower bounds for the variables, i.e., a n-dimensional vector
#     ub -- upper bounds for the variables, i.e., a n-dimensional vector
#     A_sparse -- a sparse matrix s.t. A_sparse x <= b
#     b -- a vector matrix s.t. A_sparse x <= b
#     objective_function -- the objective function, i.e., a n-dimensional vector
#     """
#     model.remove(model.getVars())
#     model.update()
#     model.remove(model.getConstrs())
#     model.update()
#     x = model.addMVar(
#         shape=n,
#         vtype=GRB.CONTINUOUS,
#         name="x",
#         lb=lb,
#         ub=ub,
#     )
#     model.update()
#     model.addMConstr(Aeq_sparse, x, "=", beq, name="c")
#     model.update()
#     model.addMConstr(A_sparse, x, "<", b, name="d")
#     model.update()
#     model.setMObjective(None, objective_function, 0.0, None, None, x, GRB.MINIMIZE)
#     model.update()

#     return model

# def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
#     """A function to find and remove the redundant facets and to find
#     the facets with very small offset and to set them as equalities

#     Keyword arguments:
#     lb -- lower bounds for the fluxes, i.e., a n-dimensional vector
#     ub -- upper bounds for the fluxes, i.e., a n-dimensional vector
#     S -- the mxn stoichiometric matrix, s.t. Sv = 0
#     c -- the objective function to maximize
#     opt_percentage -- consider solutions that give you at least a certain
#                       percentage of the optimal solution (default is to consider
#                       optimal solutions only)
#     """
#     pass


# def fast_inner_ball(A, b):
#     """A Python function to compute the maximum inscribed ball in the given polytope using gurobi LP solver
#     Returns the optimal solution for the following linear program:
#     max r, subject to,
#     a_ix + r||a_i|| <= b, i=1,...,n

#     Keyword arguments:
#     A -- an mxn matrix that contains the normal vectors of the facets of the polytope row-wise
#     b -- a m-dimensional vector
#     """
