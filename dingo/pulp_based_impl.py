import pulp as pl
import numpy as np


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

    # A = np.zeros((2 * n, n), dtype="float")
    # A[0:n] = np.eye(n)
    # A[n:] -= np.eye(n, n, dtype="float")

    # b = np.concatenate((ub, -lb), axis=0)
    # b = np.asarray(b, dtype="float")
    # b = np.ascontiguousarray(b, dtype="float")

    beq = np.zeros(m)

    try:
        
        model = pl.LpProblem("FBA", pl.LpMaximize)
        
        x = [pl.LpVariable(f"x{i}", lowBound=lb[i], upBound=ub[i]) for i in range(n)]
        
        for i in range(m):
            model += (
                pl.lpDot(S[i], x) == 0,
                # pl.lpSum([S[i, j] * x[j] for j in range(n)]) == 0,
                f"Mass balance constraint {i}"
            )
        
        model += (
            pl.lpDot(c,x),
            "Objective function"
        )
        
        # solver = pl.PULP_CBC_CMD(msg=0)
        solver = pl.HiGHS(msg=0)
        model.solve(solver=solver)

        status = pl.LpStatus[model.status]
        
        # If optimized
        if status == "Optimal":
            optimum_value = pl.value(model.objective)
            optimum_sol = np.array([pl.value(x[i]) for i in range(n)])

        return optimum_sol, optimum_value

    except AttributeError:
        print("scipy.optimize.linprog failed.")
