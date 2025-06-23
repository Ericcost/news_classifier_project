# # Descente de gradient pour minimiser f(x) = (x-3)^2

# # Définition de la fonction à minimiser
# def f(x):
#     return (x - 3) ** 2

# # Définition de la dérivée de la fonction (gradient)
# def grad_f(x):
#     return 2 * (x - 3)

# # Initialisation du point de départ
# x = 0.0  # On commence à x=0

# # Taux d'apprentissage (learning rate)
# lr = 0.1  # Contrôle la taille des pas de mise à jour

# # Nombre d'itérations
# n_iter = 20

# for i in range(n_iter):
#     grad = grad_f(x)         # Calcul du gradient au point courant
#     x = x - lr * grad        # Mise à jour de x dans la direction opposée au gradient
#     print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")  # Affichage du progrès

# # À la fin, x devrait être proche de 3, le minimum de la fonction

# Descente de gradient pour minimiser f(x, y) = (x-2)^2 + (y+3)^2

# # Définition de la fonction à minimiser
# def f(x, y):
#     return (x - 2) ** 2 + (y + 3) ** 2

# # Définition du gradient (dérivées partielles par rapport à x et y)
# def grad_f(x, y):
#     df_dx = 2 * (x - 2)      # Dérivée partielle par rapport à x
#     df_dy = 2 * (y + 3)      # Dérivée partielle par rapport à y
#     return df_dx, df_dy

# # Initialisation des paramètres
# x, y = 0.0, 0.0  # Point de départ

# # Taux d'apprentissage
# lr = 0.1

# # Nombre d'itérations
# n_iter = 20

# for i in range(n_iter):
#     grad_x, grad_y = grad_f(x, y)  # Calcul du gradient pour x et y
#     x = x - lr * grad_x            # Mise à jour de x
#     y = y - lr * grad_y            # Mise à jour de y
#     print(f"Iteration {i+1}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {f(x, y):.4f}")

# # À la fin, x devrait être proche de 2 et y proche de -3 (le minimum de la fonction)

import numpy as np

# 1. Initialiser les poids aléatoirement selon N(0, sigma^2)
# Ici, on choisit une distribution normale centrée en 0 avec une variance sigma^2.
# Cela permet de commencer l'optimisation à partir d'un point aléatoire, ce qui est important
# pour éviter certains problèmes comme rester bloqué dans des symétries ou des minima locaux.
# Dans le contexte du deep learning, cette étape est cruciale car une mauvaise initialisation
# peut empêcher le réseau d'apprendre correctement.
sigma = 0.1
W = np.random.normal(0, sigma, size=2)  # Exemple avec 2 paramètres (W[0] et W[1])

# Exemple de fonction de coût J(W) = (W[0] - 2)^2 + (W[1] + 3)^2
# Cette fonction représente une "erreur" ou "coût" que l'on cherche à minimiser.
# Ici, J(W) atteint son minimum lorsque W[0]=2 et W[1]=-3.
# En deep learning, la fonction de coût mesure à quel point les prédictions du modèle
# sont éloignées de la vérité, et on ajuste les paramètres (ici W) pour la rendre la plus petite possible.
def J(W):
    return (W[0] - 2) ** 2 + (W[1] + 3) ** 2

# Gradient de J par rapport à W
# Ici, on calcule le vecteur gradient de la fonction de coût J par rapport aux paramètres W.
# Le gradient indique la direction dans laquelle la fonction J augmente le plus rapidement.
# Pour minimiser J, on doit donc se déplacer dans la direction opposée au gradient.
# Pour notre fonction J(W) = (W[0] - 2)^2 + (W[1] + 3)^2 :
# - La dérivée partielle par rapport à W[0] est 2*(W[0] - 2)
# - La dérivée partielle par rapport à W[1] est 2*(W[1] + 3)
# On retourne le gradient sous forme de vecteur numpy, ce qui facilitera les calculs vectoriels.
def grad_J(W):
    dJ_dW0 = 2 * (W[0] - 2)      # Dérivée partielle de J par rapport à W[0]
    dJ_dW1 = 2 * (W[1] + 3)      # Dérivée partielle de J par rapport à W[1]
    return np.array([dJ_dW0, dJ_dW1])  # On retourne le gradient sous forme de vecteur

# 2. Boucle jusqu'à convergence
# Ici, on définit les hyperparamètres de la descente de gradient :
# - learning_rate (taux d'apprentissage) : contrôle la taille des pas lors de la mise à jour des poids.
#   Un taux trop grand peut empêcher la convergence, un taux trop petit rend l'apprentissage très lent.
# - n_iter (nombre maximal d'itérations) : on fixe un nombre maximal d'itérations pour éviter les boucles infinies.
# - tolerance (tolérance de convergence) : si la mise à jour des poids devient très petite (inférieure à la tolérance),
#   on considère que l'algorithme a convergé vers un minimum et on arrête la boucle.
learning_rate = 0.1
n_iter = 100
tolerance = 1e-3


# Boucle principale de la descente de gradient
for i in range(n_iter):
    grad = grad_J(W)  # 3. Calculer le gradient de la fonction de coût J par rapport aux poids W
    # Le gradient indique la direction dans laquelle la fonction de coût augmente le plus rapidement.
    # Pour minimiser J, on doit donc se déplacer dans la direction opposée au gradient.

    W_new = W - learning_rate * grad  # 4. Mettre à jour les poids en suivant la direction opposée au gradient
    # Ici, on applique la règle de mise à jour : W = W - learning_rate * gradient
    # Le learning_rate (taux d'apprentissage) contrôle la taille du pas de mise à jour.

    if np.linalg.norm(W_new - W) < tolerance:  # Critère de convergence
        # On vérifie si la mise à jour des poids est suffisamment petite (inférieure à la tolérance fixée).
        # Si c'est le cas, cela signifie que l'algorithme a convergé vers un minimum et on peut arrêter la boucle.
        break

    W = W_new  # On met à jour les poids pour l'itération suivante

    print(f"Iteration {i+1}: W = {W}, J(W) = {J(W):.4f}")
    # Affichage de l'état actuel : les poids et la valeur de la fonction de coût à chaque itération
    # Cela permet de visualiser la progression de l'algorithme vers le minimum.

# 5. Retourner les poids finaux
print("Poids finaux:", W)
# À la fin de la boucle, on affiche les poids finaux trouvés par la descente de gradient.
# Ces poids devraient être proches du minimum de la fonction de coût J, c'est-à-dire W[0] ≈ 2 et W[1] ≈ -3.

import numpy as np

# Poids finaux trouvés par la descente de gradient
W = np.array([1.99726522, -2.9964832])

# Exemple de nouvelle entrée
x1 = 2.0
x2 = -1.0

# Prédiction : y = W[0]*x1 + W[1]*x2
y_pred = W[0] * x1 + W[1] * x2

print(f"Prédiction pour x1={x1}, x2={x2} : y = {y_pred:.4f}")