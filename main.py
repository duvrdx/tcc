import pandas as pd
from algorithm.NAS import MLPDirectorsCutNAS
import warnings
import argparse
import numpy as np

def main() -> None:
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(description='Train a Neural Network using NAS')
  parser.add_argument('--logs', type=str, help='Logs directory')
  parser.add_argument('--charts', type=str, help='Charts directory')
  parser.add_argument('--data', type=str, help='Data file')
  parser.add_argument('--target', type=str, help='Target column')
  parser.add_argument('--epochs', type=int, help='Number of epochs')
  parser.add_argument('--aval', type=int, help='Number of avaliations')
  
  args = parser.parse_args()
  
  df = pd.read_csv(f"{args.data}", index_col=0)
  df.replace('             ', np.nan, inplace=True)
  df.dropna(inplace=True)
  df.sort_index(inplace=True)

  y = df[f"{args.target}"]
  X = df.drop(columns=[f"{args.target}"])

  nas = MLPDirectorsCutNAS(X, y, log_dir=args.logs, charts_dir=args.charts)
  nas.apply_pca()
  
  nas.performance_estimation_strategy(args.aval, args.epochs)

if __name__ == "__main__":
  main()