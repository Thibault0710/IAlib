import numpy as np
from scipy.optimize import minimize

# Fonction logistique (sigmoïde)
def sigmoid(X, theta, beta):
    z = np.dot(X, theta) + beta
    return 1 / (1 + np.exp(-z))

# Fonction de perte pour la régression logistique (moindres carrés)
def loss_function(params, X, y):
    theta = params[:-1]
    beta = params[-1]
    y_pred = sigmoid(X, theta, beta)
    return np.mean((y_pred - y) ** 2)

# Données d'entrée X et valeurs cibles Y
X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [0.0, 6.0]])
Y = np.array([0.5, 0.5, 0.5, 0.5])

# Initialisation aléatoire des coefficients
initial_params = np.random.randn(X.shape[1] + 1)

# Minimisation de la fonction de perte
result = minimize(loss_function, initial_params, args=(X, Y))

# Coefficients optimaux du modèle
coefficients = result.x[:-1]
beta = result.x[-1]

print("Coefficients de la régression logistique:", coefficients)
print("Beta:", beta)
