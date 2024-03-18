import pandas as pd 
import os
import numpy as np

def read_csv_from_directory(directory_path: str,
                            target: str,
                            columns_to_drop: list[str] = []) -> list[pd.DataFrame]:
  files: list[str] = [os.path.join(directory_path, file)
                      for file in os.listdir(directory_path)
                      if file.endswith(".csv")]
  
  data: list[pd.DataFrame] = []
  
  for pos, file in enumerate(files):
    dataframe = pd.read_csv(file, encoding='latin1', sep=";")
    
    
    if pos == 0:
      columns_to_drop += [column for column in dataframe.columns if column.startswith("Unnamed")]
    
    dataframe[file.split("/")[-1][:-4].split("-")[-1]] = dataframe[target]
    data.append(dataframe.drop(columns=columns_to_drop + [target]))
    
  return data

def join_dataframes(dataframes: list[pd.DataFrame], column_to_join: str) -> pd.DataFrame:
  dataframes = [dataframe.set_index(column_to_join) for dataframe in dataframes]
  return dataframes[0].join(dataframes[1:])

def pre_proccess_df(dataframes: list[pd.DataFrame]):
  
  for dataframe in dataframes:
    date_string = dataframe.apply(lambda row: f"{row['Data']} {row['Hora']}", axis=1)
    dataframe["timestamp"] = pd.to_datetime(date_string, format='%d/%m/%Y %H:%M:%S')
    dataframe.drop(columns=["Data", "Hora"], inplace=True)
    dataframe.replace('             ', np.nan, inplace=True)
    dataframe.dropna(inplace=True)
  
  return dataframes