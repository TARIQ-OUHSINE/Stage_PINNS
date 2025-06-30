import pandas as pd

file_solid = 'Cristall_All.txt'
file_juice = 'Juice_All.txt'

data_solid = pd.read_csv(file_solid, delim_whitespace=True, header=None)
data_juice = pd.read_csv(file_juice, delim_whitespace=True, header=None)

data_solid.columns = ['P0', 't', 'y']
data_juice.columns = ['P0', 't', 'y']

data_solid.to_excel('S_f.xlsx', index=False)
data_juice.to_excel('S_j.xlsx', index=False)


donnees = {
    "C_j": float,
    "C_f": float,
    "T_1": float,
    "R_s": float,
    "CrisOff":  {"P0_j": float, "TB_j": float, "t": [], "P_moy": []},
    "CrisOn":  {"P0_j": float, "TB_j": float, "t": [], "P_moy": []},
    "JuiceOff": {"P0_j": float, "TB_j": float, "t": [], "P_moy": []},
    "JuiceOn": {"P0_j": float, "TB_j": float, "t": [], "P_moy": []},
}