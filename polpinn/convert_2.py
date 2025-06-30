import pandas as pd

# Lire le fichier texte en sautant les lignes de commentaires (commençant par %)
file_path = '../../FolderRun/FolderRun/LongTime_110_20_300_5000/JuiceOff.txt'
df = pd.read_csv(
    file_path,
    delim_whitespace=True,
    comment='%',
    header=None,
    skip_blank_lines=True
)

# Ajouter les bons noms de colonnes
df.columns = ["P0_Variable", "TB_Variable_(s)", "Time_(s)", "Variable_dépendante_P_(1)"]

# Sauvegarder en Excel
df.to_excel("JuiceOff.xlsx", index=False)
print("Conversion terminée avec succès.")
