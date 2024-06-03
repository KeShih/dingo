import numpy as np

# from optlang.gurobi_interface import Model, Variable, Constraint, Objective
# from optlang.glpk_interface import Model, Variable, Constraint, Objective
# from optlang.scipy_interface import Model, Variable, Constraint, Objective
# from optlang import Model, Variable, Constraint, Objective
from optlang.gurobi_interface import Model, Variable, Constraint, Objective

def opt_fba(lb, ub, S, c):
    """A Python function to perform fba using optlang
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

    beq = np.zeros(m)

    # A = np.zeros((2 * n, n), dtype="float")
    # A[0:n] = np.eye(n)
    # A[n:] -= np.eye(n, n, dtype="float")

    # b = np.concatenate((ub, -lb), axis=0)
    # b = np.asarray(b, dtype="float")
    # b = np.ascontiguousarray(b, dtype="float")

    c = np.asarray(c)

    beq = np.zeros(m)

    try:
        model = Model()

        model.configuration.verbosity = 0
        # model.configuration.lp_method = ""

        # Create variables
        x = np.array(
            [
                Variable(name="x{}".format(i), lb=lb[i], ub=ub[i], type="continuous")
                for i in range(n)
            ]
        )
        model.add(x)

        # Add constraints
        constraints1 = np.array([Constraint(0, lb=0, ub=0) for i in range(m)])
        model.add(constraints1)
        model.update()
        for i in range(m):
            constraints1[i].set_linear_coefficients({x[j]: S[i, j] for j in range(n)})

        # Add constraints for the inequalities of A
        # constraints2 = np.array(
        #     [Constraint(x[i], lb=lb[i], ub=ub[i]) for i in range(n)]
        # )
        # model.add(constraints2)
        

        # Set the objective function in the model
        objective_function = Objective((-c).dot(x), value=0, direction="min")

        model.objective = objective_function

        status = model.optimize()

        if status == "optimal":
            optimum_value = model.objective.value
            optimum_sol = np.array([var.primal for var in x])

        return optimum_sol, -optimum_value

    except AttributeError:
        print("optlang failed.")