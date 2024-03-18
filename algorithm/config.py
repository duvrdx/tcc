optimizers = ['SGD', 'RMSprop', 'Adam','Adadelta','Adamax']
layers = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
layer_size = [32, 64, 128, 256, 512]
# loss = ['mse','mae','map'] # Vou optar por loss = MSE
activation = ['identity', 'logistic', 'relu', 'tanh'] #,'softmax',]
# metricas = ['rmse','mse','mae'] # Vou setar a métrica do algoritmo igual a do artigo

# Exclusivos Para o modelo MLP Regressor sklearn
hidden_layer = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
alpha = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
batch_size = [200, 250, 300] #[50, 100, 150, 200, 250, 300]
learning_rate = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.000001]
max_iter = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]