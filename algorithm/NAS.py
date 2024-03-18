import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import algorithm.config as cfg
from just_logger.logging import Logger 
from datetime import datetime

class MLPDirectorsCutNAS:
  def __init__(self, X: pd.DataFrame, y: pd.DataFrame, log_dir: str, charts_dir: str):
    self.initial_timestamp: datetime = datetime.now()
    self.logger: Logger = Logger(f"./{log_dir}/{self.initial_timestamp}.log")
    self.chart_dir: str = charts_dir
    self.X: pd.DataFrame = X
    self.X_normalized: pd.DataFrame = None
    self.y: pd.DataFrame = y
    
    self.layer_size: list[int] =cfg.layer_size
    self.activation: list[str] = cfg.activation
    self.hidden_layer: list[int] = cfg.hidden_layer
    self.alpha: list[int] = cfg.alpha
    self.batch_size: list[int] = cfg.batch_size
    self.learning_rate: list[int] = cfg.learning_rate
    self.max_iter: list[int] = cfg.max_iter
    
  def _get_random_architecture(self) -> dict:
    return {
        "hidden_layer": np.random.choice(self.hidden_layer),
        "activation": np.random.choice(self.activation),
        "alpha": np.random.choice(self.alpha),
        "batch_size": np.random.choice(self.batch_size),
        "learning_rate": np.random.choice(self.learning_rate),
        "max_iter": np.random.choice(self.max_iter)
    }
  
  def train_model_mlp(self, architeture: dict[str, str], run_test=False) -> tuple[float, float]:
    
    mlp = MLPRegressor(hidden_layer_sizes=(architeture["hidden_layer"]), activation=architeture["activation"],
                            solver='adam', alpha=architeture["alpha"], batch_size=architeture["batch_size"],
                            learning_rate_init=architeture["learning_rate"], max_iter=architeture["max_iter"])
    
    mlp.fit(self.X_train, self.y_train.values)
    
    X = self.X_normalized if run_test else self.X_test
    y = self.y if run_test else self.y_test

    prediction = mlp.predict(X)
    mse = (np.square(y - prediction)).mean(axis=None)
        
    return (prediction, np.sqrt(mse), r2_score(y.values, prediction))
  
  def apply_pca(self) -> None:
    pca = PCA(0.98)
    scaler = StandardScaler()
    self.X_normalized = pd.DataFrame(pca.fit_transform(scaler.fit_transform(self.X)))
    # Base de treino e teste
    self.X_train, self.X_test, \
    self.y_train, self.y_test = train_test_split(self.X_normalized, self.y, random_state=42)
  
    return self
  
  def search_space_sklearn(self, architetures: int) -> list[dict]:
    return [self._get_random_architecture() for _ in range(architetures)]
  
  def search_strategy_mlp_skl(self, aval: int, epochs: int, rmse_base: float = 10000, r2_base=0) -> dict[str, str]:
    best_architeture: dict = None
    
    for i in range(epochs):
        print(f'Epoch: {i}')
        
        for architeture in tqdm(self.search_space_sklearn(aval), desc="Avalition"):

            _, rmse, r2 = self.train_model_mlp(architeture)
            
            if rmse < rmse_base and r2 > r2_base:
                rmse_base = rmse
                r2_base = r2
                
                best_architeture = {
                      "hidden_layer": architeture["hidden_layer"],
                      "activation": architeture["activation"],
                      "alpha": architeture["alpha"],
                      "batch_size": architeture["batch_size"],
                      "learning_rate": architeture["learning_rate"],
                      "max_iter": architeture["max_iter"]
                }
                self.logger.info(f'Found new Best Architeture at Epoch {i}: {best_architeture}')
                self.logger.info(f'RMSE: {rmse:.5f}')
                self.logger.info(f'R2: {r2:.5f}')
        print('-'* 40)
        
    return best_architeture
  
  def performance_estimation_strategy(self, aval: int, epochs: int) -> None:
      
      best_architeture = self.search_strategy_mlp_skl(aval, epochs)
      
      if not best_architeture:
        print("No best architeture found. Exiting...")
        return
      
      prediction, rmse, r2 = self.train_model_mlp(best_architeture, True)
      
      self.logger.info(f'Final Best Architeture: {best_architeture}')
      self.logger.info(f'RMSE MLP (Full Dataset): {rmse:.5f}')
      self.logger.info(f'R2 MLP (Full Dataset): {r2:.5f}')
      
      print(f'Final Best Architeture: {best_architeture}')      
      print(f'RMSE MLP (Full Dataset): {rmse:.5f}')
      print(f'R2 MLP (Full Dataset): {r2:.5f}')

      self.plot_real_vs_pred_with_residuals('MLP', self.y, prediction, prediction - self.y.values)  

      print("Finished without errors 😄")
      print(f'Elapsed time: {datetime.now() - self.initial_timestamp}')
      
  def plot_real_vs_pred_with_residuals(self, model: str, y_real, y_pred, loss):

    plt.figure(figsize=(20, 8))

    # Gráfico de Séries Temporais Linear
    plt.plot(y_real.index, y_real.values, label='Real')
    plt.plot(y_real.index, y_pred, label='Predito')
    plt.xlabel('Tempo')
    plt.ylabel('Valores')
    plt.title(f'Real x Predito - {model}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{self.chart_dir}/temporal_series_{model}_{self.initial_timestamp}.png')

    plt.figure(figsize=(20, 8))
    # Histograma de Resíduos
    plt.hist(loss, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de Resíduos - {model}')
    plt.xlabel('Frequência')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{self.chart_dir}/histogram_residuals_{model}_{self.initial_timestamp}.png')

        