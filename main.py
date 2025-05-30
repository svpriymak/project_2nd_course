import pandas as pd
import glob
import os

##впиши свой путь до датасета
folder_path = '/Users/svpriymak/Downloads/ozon'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dfs = {
    os.path.basename(file): pd.read_csv(file, on_bad_lines='skip', low_memory=False)
    for file in csv_files
}

print("Файлы в папке:")
for name in dfs:
    print("-", name)

file_to_view = 'export_ozon_Супермаркет Экспресс_2020-12-18_2021-01-18.csv'
if file_to_view in dfs:
    print(dfs[file_to_view].head())
else:
    print("Файл не найден.")