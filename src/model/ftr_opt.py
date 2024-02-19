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
        """
        Method to perform post-initialization operations.

        :return: None
        """
        self.nodes = self.z_cost.columns.tolist()
        self.z_cost.index = self.nodes
        self.z_rev.index = self.nodes
        self.shift_factors.set_index("Name", inplace=True)
        self.path_pref.index = self.path_pref.columns.tolist()

    def optimize_portfolio(self):
        """
        Optimize Portfolio

        This method optimizes the allocation of MW (MegaWatt) along different paths in order to maximize net revenue. It uses a linear optimization model to find the optimal solution.

        Parameters:
            None

        Returns:
            None

        """
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
            """
            :param model: The model to extract the shift factors from.
            :param node: The node to extract the shift factors for.
            :param constraint: The constraint to extract the shift factors for.
            :return: The shift factors for the given node and constraint.

            """
            return self.shift_factors.loc[self.shift_factors.index == node, constraint].iloc[0]

        model.shift_factors = Param(model.nodes, model.decomp_limits,
                                    initialize=extract_shift_factors,
                                    within=Reals)

        # Binding Constraints
        def z_bc_min_limits(model, constraint):
            """
            :param model: The model used for optimization.
            :param constraint: The constraint for which to find the minimum limit.
            :return: The minimum limit value for the given constraint.

            """
            return self.z_bcl.loc[self.z_bcl['Mas_Cons'] == constraint, 'MasMin'].iloc[0]

        def z_bc_max_limits(model, constraint):
            """
            Retrieves the maximum limit from the FTROpt.z_bcl data frame for a specific constraint.

            :param model: The FTROpt model object.
            :param constraint: The constraint for which the maximum limit needs to be retrieved.
            :return: The maximum limit value.
            """
            return self.z_bcl.loc[self.z_bcl['Mas_Cons'] == constraint, 'MasMax'].iloc[0]

        model.bc_max = Param(model.decomp_limits, initialize=lambda model, c: z_bc_max_limits(model, c),
                             within=Reals)
        model.bc_min = Param(model.decomp_limits, initialize=lambda model, c: z_bc_min_limits(model, c),
                             within=Reals)

        # Binary variable indicating if a arc is selected
        model.arc_indicator = Var(model.arcs, within=Binary)

        # MW allocated to each arc, path between node i and node j
        model.allocated_mw = Var(model.arcs, bounds=(0, 50), within=NonNegativeReals)

        # Impose a max arc selection to 1500
        model.total_arcs_constraint = Constraint(expr=sum(model.arc_indicator[i, j] for i, j in model.arcs) <= 1500)

        # Decomposition constraints
        def decomposition_rule_min(model, constraint):
            """
            :param model: The optimization model.
            :param constraint: The constraint for which the decomposition rule should be applied.
            :return: True if the sum of allocated_mw multiplied by the difference of shift factors for each arc in the model meets the minimum constraint value.
            """
            return sum(
                model.allocated_mw[i, j] * (
                        model.shift_factors[i, constraint] - model.shift_factors[j, constraint]) for
                i, j in model.arcs) >= model.bc_min[constraint]

        model.decomp_min_limit = Constraint(model.decomp_limits, rule=decomposition_rule_min)

        def decomposition_rule_max(model, constraint):
            """
            :param model: The optimization model
            :param constraint: The constraint for maximum shift factor difference
            :return: True if the decomposition rule is satisfied, False otherwise
            """
            return sum(model.allocated_mw[i, j] * (
                    model.shift_factors[i, constraint] - model.shift_factors[j, constraint]) for i, j in
                       model.arcs) <= model.bc_max[constraint]

        model.decomp_max_limit = Constraint(model.decomp_limits, rule=decomposition_rule_max)

        def min_allocated_mw_constraint(model, i, j):
            """
            Minimizes the allocation of MW (megawatts) constraint for a given model, arc indices i and j.

            :param model: The optimization model.
            :type model: FTROpt
            :param i: The index of the source arc.
            :type i: int
            :param j: The index of the destination arc.
            :type j: int
            :return: The constraint that ensures the allocated MW is less than or equal to the arc indicator multiplied by 50.
            :rtype: constraint
            """
            return model.allocated_mw[i, j] <= model.arc_indicator[i, j] * 50 # 50 is max MW limit for each arc

        model.min_allocated_mw_constraint = Constraint(model.arcs, rule=min_allocated_mw_constraint)

        def path_pref_objective(model):
            """
            :param model: The optimization model containing necessary variables and parameters for portfolio optimization.

            :return: The path preference objective value, which is calculated by summing the product of the arc weights (path_pref) and the arc indicators (arc_indicator) over all arcs in the model
            *.

            """
            return model.portfolio_path_pref == sum(model.path_pref[i, j] * model.arc_indicator[i, j] for i, j in model.arcs)

        model.path_pref_cons = Constraint(rule=path_pref_objective)

        # Step 1: Solve for the first objective (net revenue)
        def objective_revenue(model):
            """
            :param model: An instance of the FTROpt class representing the optimization portfolio model.
            :return: The objective revenue calculated based on the model's allocated_mw, revs, costs, and portfolio_path_pref attributes.
            The objective is weighted 0.9 for total revenue and 0.1 10% for path preferences

            """
            return (0.9*sum(model.allocated_mw[i, j] * (model.revs[i, j] - model.costs[i, j]) for i, j in model.arcs) + 0.1*
                    model.portfolio_path_pref)

        model.Objective = Objective(rule=objective_revenue, sense=maximize)

        solver = SolverFactory("cplex", executable="/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex")
        solver.options['mipgap'] = 0.01
        solver.options['threads'] = 8

        # Solve the model
        solver_status = solver.solve(model, tee=True)

        # Check solver status and store the optimal revenue value
        if (solver_status.solver.status == SolverStatus.ok) and (
                solver_status.solver.termination_condition == TerminationCondition.optimal):
            optimal_revenue = value(model.Objective)
        else:
            raise Exception("Optimal solution not found for the first objective.")

        allocated_mw = np.zeros(shape=(len(self.nodes), len(self.nodes)))
        for i, nr in enumerate(self.nodes):
            for j, nc in enumerate(self.nodes):
                if nr != nc:
                    allocated_mw[i, j] = value(model.allocated_mw[nr, nc])

        optimal_mw_allocation = pd.DataFrame(data=allocated_mw, columns=self.nodes)
        optimal_mw_allocation.to_csv("./output/allocated_mw.csv", index=False)
