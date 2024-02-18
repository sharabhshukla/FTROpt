## Introduction
The FTR (Financial Transmission Rights) Optimization model aims to maximize the net revenue from transmission rights within a power grid network. It considers costs, revenues, binding constraints, and shift factors associated with different transmission paths.

## Data Preparation

Data Inputs: The model starts with several Pandas DataFrames as inputs:
    - `z_cost`: Contains the cost associated with utilizing each potential transmission path.
    - `z_rev`: Contains the revenue potential of each transmission path.
    - `z_bcl`: Stores binding constraints like minimum and maximum limits for power transmission.
    - `shift_factors`: Indicates how power flows affect the network and constraints.
    - `path_pref`: Represents preferences for using certain paths, potentially based on non-financial factors.

## Model Definition

Pyomo Model Setup: A `ConcreteModel` from Pyomo is initialized to represent the optimization problem.
 Sets: Define the nodes in the network, arcs (possible paths between nodes), and decomposition limits from `z_bcl`.
 Parameters: Costs, revenues, path preferences, and shift factors are set up as parameters within the model, linked to the respective arcs and nodes.


## Variables
    - `arc_indicator`: Binary variables indicating whether a particular path (arc) is selected.
    - `allocated_mw`: Continuous variables representing the amount of power (in MW) allocated to each selected path.
    - `portfolio_path_pref`: A variable capturing the overall preference of the chosen portfolio of paths.


## Constraints
    - A total arc constraint limits the number of paths that can be selected.
    - Decomposition constraints ensure that the power flow complies with network and regulatory limits.
    - Additional constraints link the MW allocation to the binary path selection variables.


## Objective
    - The objective function seeks to maximize the net revenue, calculated as the difference between revenues and costs across all selected paths, adjusted by the portfolio path preference. The objective is weighted, a better alternative might be lexicographic objective optimization


## Solver Setup
    - The CPLEX solver is configured with specific options for the maximum permissible gap (`mipgap`) and the number of threads to use.
    - The model is solved, and the solver's status is checked to confirm an optimal solution has been found.


## Output Analysis
    - Upon successful optimization, the MW allocations for each path are extracted and saved to a CSV file for further analysis or operational planning.
