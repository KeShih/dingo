import pyomo.environ as pyo
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
        model = pyo.AbstractModel()
        
        model.x = pyo.Var(range(n))
        
        model.obj = pyo.Objective(expr=pyo.summation(c, model.x))
        
        # constrains
        def mass_balance_rule(model, i):
            return sum(S[i, j] * model.x[j] for j in range(n)) == 0
        model.mass_balance = pyo.Constraint(range(m), rule=mass_balance_rule)
        
        
        
        
        
        

    except AttributeError:
        print("scipy.optimize.linprog failed.")
