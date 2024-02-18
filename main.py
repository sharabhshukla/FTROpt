import pandas as pd
from src.model.ftr_opt import FTROpt


z_bcl_df = pd.read_csv("./data/raw/Z_BCLimits.csv")
z_cost_df = pd.read_csv("./data/raw/Z_Cost.csv")
z_rev_df = pd.read_csv("./data/raw/Z_Rev.csv")
shift_fctrs_df = pd.read_csv("./data/raw/Z_SF.csv")
path_pref_df = pd.read_csv("./data/raw/Z_MPUnSym.csv")


ftr_opt = FTROpt(
    z_cost=z_cost_df,
    z_bcl=z_bcl_df,
    z_rev=z_rev_df,
    shift_factors=shift_fctrs_df,
    path_pref=path_pref_df
)

ftr_opt.optimize_portfolio()