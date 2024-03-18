import requests
import os
import xmltodict
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_hydrographic_basin(basin: int = None, sub_basin: int = None) -> dict:
  if basin is not None and sub_basin is not None:
    url: str = f"{os.environ.get('ANA_BASE_URL')}HidroBaciaSubBacia?codBacia={basin}&codSubBacia={sub_basin}"
  elif basin is not None:
    url: str = f"{os.environ.get('ANA_BASE_URL')}HidroBaciaSubBacia?codBacia={basin}&codSubBacia="
  elif sub_basin is not None:
    url: str = f"{os.environ.get('ANA_BASE_URL')}HidroBaciaSubBacia?codBacia=&codSubBacia={sub_basin}"
  else:
    url: str = f"{os.environ.get('ANA_BASE_URL')}HidroBaciaSubBacia"
  
  print(url)
  
  response: requests.Response = requests.get(url)
  
  return xmltodict.parse(response.text)

def get_telemetric_stations(status: int = 0, origin: int = 0 ) -> dict:
  url: str = f"{os.environ.get('ANA_BASE_URL')}ListaEstacoesTelemetricas?statusEstacoes={status}&origem={origin}"
  response: requests.Response = requests.get(url)
  
  return xmltodict.parse(response.text)

def pre_proccess_data(data: dict) -> dict:
  data.pop("@msdata:rowOrder")
  data.pop("@diffgr:id")
  return data

def filter_by_uf(data: dict, uf: str) -> list:
  tables: list[dict] = data["DataSet"]["diffgr:diffgram"]["Estacoes"]["Table"]
  filtered_data: list[dict] = []

  for table in tables:
    try:
      if table["Municipio-UF"][-2:] == uf: 
        filtered_data.append(pre_proccess_data(table))
    except:
      pass
    
  return filtered_data

def pre_proccess_sub_basins(basin: dict) -> dict:
  tables: list[dict] = basin["DataSet"]["diffgr:diffgram"]['SubBacia']["Table"]
  basin_id: str = tables[0]["codBacia"]
  basin_name: str = tables[0]["nmBacia"]
  
  basin_obj: dict = {
    "id": basin_id,
    "name": basin_name
  }
  
  data: list = []
  
  for table in tables:
    data.append(pre_proccess_data(table))
    
  return {
    "basin": basin_obj,
    "sub_basins": data
  }
  

def telemetric_stations_to_df(data: list[dict]) -> pd.DataFrame:
  return pd.DataFrame(data)