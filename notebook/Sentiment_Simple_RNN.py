import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split

import sys
import os

# Ajouter le dossier parent (news_classifier_project/) au path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Maintenant tu peux importer ton module
from nlp.preprocessing import Preprocessing
from nlp.simple_rnn import SimpleRNN
from torch.utils.data import TensorDataset, DataLoader

# Hyperparamètres
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 3  # À adapter selon ton jeu de données
MAX_LEN = 20  # À adapter selon preprocessing

### Chargement des données
path = '/Users/ericcosterousse/.cache/kagglehub/datasets/abdelmalekeladjelet/sentiment-analysis-dataset/versions/1'
df = pd.read_csv(os.path.join(path, 'sentiment_data.csv'))
df.drop('Unnamed: 0', axis=1, inplace=True)
print("Étape 1 - Données chargées :")
print(df.head())

### Prétraitement des commentaires
if 'Comment' not in df.columns:
    raise ValueError("La colonne 'Comment' n'existe pas dans le DataFrame.")
preprocess = Preprocessing()
df['Cleaned_Comment'] = df['Comment'].astype(str).apply(preprocess.preprocess)
print("\nÉtape 2 - Commentaires nettoyés :")
print(df[['Comment', 'Cleaned_Comment']].head())

# Tokenisation et création du vocabulaire
tokenized = df['Cleaned_Comment'].apply(lambda x: x.split())
print("\nÉtape 3 - Exemple de tokens :")
print(tokenized.head())

word_counter = Counter([token for row in tokenized for token in row])
most_common = word_counter.most_common(VOCAB_SIZE - 2)
word2idx = {'<PAD>': 0, '<UNK>': 1}
for i, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = i
print("\nÉtape 4 - Taille du vocabulaire :", len(word2idx))
print("Exemples de mots du vocabulaire :", list(word2idx.items())[:10])

# Encodage et padding
def encode_and_pad(tokens, word2idx, max_len):
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in tokens]
    if len(indices) > max_len:
        return indices[:max_len]
    return indices + [word2idx['<PAD>']] * (max_len - len(indices))

df['encoded'] = tokenized.apply(lambda tokens: encode_and_pad(tokens, word2idx, MAX_LEN))
print("\nÉtape 5 - Exemple d'encodage et padding :")
print("Texte :", df['Cleaned_Comment'].iloc[0])
print("Encodé :", df['encoded'].iloc[0])

## Séparation train/test avec indices
X = torch.tensor(df['encoded'].tolist(), dtype=torch.long)
y = torch.tensor(df['Sentiment'].tolist(), dtype=torch.long)
indices = np.arange(len(df))

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
print("\nÉtape 6 - Split train/test :")

# Création de la couche embedding externe (séparée du modèle)
embedding_layer = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=EMBEDDING_DIM)

# Création du modèle SimpleRNN
model = SimpleRNN(
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
)
print("\nÉtape 7 - Modèle RNN créé :")
print(model)

# Fonction de perte
criterion = nn.CrossEntropyLoss()

# Optimiseur SGD sur model + embedding_layer
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(embedding_layer.parameters()),
    lr=1e-3
)

# Test sur un mini-batch avec embeddings pré-calculés
x_batch = X_train[:4]
y_batch = y_train[:4]
embedded_batch = embedding_layer(x_batch)  # (batch_size, seq_len, embedding_dim)
out = model(embedded_batch)
print("\nÉtape 8 - Sortie du modèle sur un mini-batch :")
print("Output shape :", out.shape)
print("Output logits :", out)

loss = criterion(out, y_batch)
print("\nÉtape 9 - Calcul de la perte :")
print("Loss value :", loss.item())

# Boucle d'entraînement (mini-test)
model.train()
optimizer.zero_grad()
embedded_batch = embedding_layer(x_batch)
out = model(embedded_batch)
loss = criterion(out, y_batch)
loss.backward()
optimizer.step()
print("\nÉtape 10 - Un pas d'entraînement effectué.")

# Précision sur le batch
preds = torch.argmax(out, dim=1)
acc = (preds == y_batch).float().mean()
print("Batch accuracy :", acc.item())

# Boucle d'entraînement complète
EPOCHS = 10
BATCH_SIZE = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\nÉtape 11 - Début de la boucle d'entraînement complète :")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        embedded_xb = embedding_layer(xb)  # calcul embeddings ici
        out = model(embedded_xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(out, dim=1)
        acc = (preds == yb).float().mean()
        epoch_loss += loss.item() * xb.size(0)
        epoch_acc += acc.item() * xb.size(0)
    epoch_loss /= len(train_dataset)
    epoch_acc /= len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

print("\nÉtape 12 - Boucle d'entraînement terminée.")

# Évaluation sur le jeu de test avec DataLoader
print("\nÉtape 12.1 - Évaluation sur le jeu de test avec DataLoader...")
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        embedded_xb = embedding_layer(xb)  # Important : calculer embeddings pour le test aussi !
        out = model(embedded_xb)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())

print("Prédictions sur le jeu de test terminées.")

# Ajout des prédictions dans le DataFrame test pour comparaison
df_test = df.iloc[test_indices].copy()
df_test['prediction'] = all_preds

# Calcul du pourcentage d'erreur sur le jeu de test
nb_erreurs = (df_test['prediction'] != df_test['Sentiment']).sum()
pourcentage_erreur = nb_erreurs / len(df_test) * 100
print(f"\nPourcentage d'erreur sur le jeu de test : {pourcentage_erreur:.2f}%")

# Affichage d'un échantillon avec prédiction et vrai label
print("\nQuelques exemples du jeu de test avec prédictions :")
print(df_test[['Cleaned_Comment', 'Sentiment', 'prediction']].head())

# Exemple de prédiction sur un nouveau tweet
nouveau_tweet = "Ce film était incroyable, j'ai adoré !"
tweet_clean = preprocess.preprocess(nouveau_tweet)
print("\nÉtape 13 - Prédiction sur un nouveau tweet :")
print("Tweet original :", nouveau_tweet)
print("Tweet nettoyé :", tweet_clean)
tweet_tokens = tweet_clean.split()
print("Tokens :", tweet_tokens)
tweet_encoded = encode_and_pad(tweet_tokens, word2idx, MAX_LEN)
print("Encodé :", tweet_encoded)
tweet_tensor = torch.tensor([tweet_encoded], dtype=torch.long)

model.eval()
with torch.no_grad():
    embedded_tweet = embedding_layer(tweet_tensor)  # calcul embeddings pour la prédiction
    output = model(embedded_tweet)
    prediction = torch.argmax(output, dim=1).item()
print("Prédiction (classe) :", prediction)

