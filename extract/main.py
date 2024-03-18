import extract_data.service_ana as sa
from pprint import pprint

print(sa.telemetric_stations_to_df(sa.filter_by_uf(sa.get_telemetric_stations(0,0), "ES")))