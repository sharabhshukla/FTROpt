from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyomo.environ import *


@dataclass
class FTROpt:
    z_cost: pd.DataFrame
    z_rev: pd.DataFrame
    z_bcl: pd.DataFrame
    shift_factors: pd.DataFrame
    path_pref: pd.DataFrame

    def __post_init__(self):
        self.nodes = self.z_cost.columns.tolist()
        self.z_cost.index = self.nodes
        self.z_rev.index = self.nodes
        self.shift_factors.set_index("Name", inplace=True)

    def optimize_portfolio(self):
        model = ConcreteModel()

        nodes = self.z_cost.columns.tolist()
        model.nodes = Set(initialize=nodes)

        model.arcs = Set(initialize=[(i, j) for i in nodes for j in nodes if i != j], dimen=2)

        model.costs = Param(model.arcs, initialize=lambda model, i, j: self.z_cost.at[i, j], within=Reals)
        model.revs = Param(model.arcs, initialize=lambda model, i, j: self.z_rev.at[i, j], within=Reals)
        model.decomp_limits = Set(initialize=self.z_bcl['Mas_Cons'].tolist())
        model.path_pref = Param(model.arcs, initialize=lambda model, i, j: self.path_pref.at[i, j], within=Reals)
        model.portfolio_path_pref = Var(within=Reals)

        # Shift Factors
        def extract_shift_factors(model, node, constraint):
            return self.shift_factors.loc[self.shift_factors.index == node, constraint].iloc[0]

        model.shift_factors = Param(model.nodes, model.decomp_limits,
                                    initialize=extract_shift_factors,
                                    within=Reals)

        # Binding Constraints
        def z_bc_min_limits(model, constraint):
            return self.z_bcl.loc[self.z_bcl['Mas_Cons'] == constraint, 'MasMin'].iloc[0]

        def z_bc_max_limits(model, constraint):
            return self.z_bcl.loc[self.z_bcl['Mas_Cons'] == constraint, 'MasMax'].iloc[0]

        model.bc_max = Param(model.decomp_limits, initialize=lambda model, c: z_bc_max_limits(model, c),
                             within=Reals)
        model.bc_min = Param(model.decomp_limits, initialize=lambda model, c: z_bc_min_limits(model, c),
                             within=Reals)

        # Binary variable indicating if a path is selected
        model.arc_indicator = Var(model.arcs, within=Binary)

        # MW allocated to each arc, path between node i and node j
        model.allocated_mw = Var(model.arcs, bounds=(0, 50), within=NonNegativeReals)

        # Impose a max arc selection to 1500
        model.total_arcs_constraint = Constraint(expr=sum(model.arc_indicator[i, j] for i, j in model.arcs) <= 1500)

        # Decomposition constraints
        def decomposition_rule_min(model, constraint):
            return sum(
                model.allocated_mw[i, j] * (
                        model.shift_factors[i, constraint] - model.shift_factors[j, constraint]) for
                i, j in model.arcs) >= model.bc_min[constraint]

        model.decomp_min_limit = Constraint(model.decomp_limits, rule=decomposition_rule_min)

        def decomposition_rule_max(model, constraint):
            return sum(model.allocated_mw[i, j] * (
                    model.shift_factors[i, constraint] - model.shift_factors[j, constraint]) for i, j in
                       model.arcs) <= model.bc_max[constraint]

        model.decomp_max_limit = Constraint(model.decomp_limits, rule=decomposition_rule_max)

        def min_allocated_mw_constraint(model, i, j):
            return model.allocated_mw[i, j] <= model.arc_indicator[i, j] * 50

        model.min_allocated_mw_constraint = Constraint(model.arcs, rule=min_allocated_mw_constraint)

        # Objective: Maximize net revenue (revenue - cost)
        def objective_revenue(model):
            return sum(
                model.allocated_mw[i, j] * (model.revs[i, j] - model.costs[i, j]) for i, j
                in model.arcs)

        model.Objective = Objective(rule=objective_revenue, sense=maximize)

        def path_pref_objective(model):
            return model.portfolio_path_pref == sum(model.path_pref[i,j] * model.arc_indicator[i,j] for i,j in model.arcs)

        solver = SolverFactory("cplex", executable="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex")
        solver.options['mipgap'] = 0.01
        solver.options['threads'] = 8
        solver_status = solver.solve(model, tee=True)

        # Check solver status and termination condition
        if (solver_status.solver.status != SolverStatus.ok) or (
                solver_status.solver.termination_condition != TerminationCondition.optimal):
            raise Exception("Optimality target not achieved. Solver Status: {}, Termination Condition: {}".format(
                solver_status.solver.status, solver_status.solver.termination_condition))

        allocated_mw = np.zeros(shape=(len(self.nodes), len(self.nodes)))
        for i, nr in enumerate(self.nodes):
            for j, nc in enumerate(self.nodes):
                if nr != nc:
                    allocated_mw[i, j] = value(model.allocated_mw[nr, nc])

        optimal_mw_allocation = pd.DataFrame(data=allocated_mw, columns=self.nodes)
        optimal_mw_allocation.to_csv("./output/allocated_mw.csv", index=False)
