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
from nlp.recurrent_neuronal_network import RecurrentNeuronalNetwork
from torch.utils.data import TensorDataset, DataLoader

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

print("Distribution des classes :")
print(df['Sentiment'].value_counts())


if 'Comment' not in df.columns:
    raise ValueError("La colonne 'Comment' n'existe pas dans le DataFrame.")
preprocess = Preprocessing()
df['Cleaned_Comment'] = df['Comment'].astype(str).apply(preprocess.preprocess)
print("\nÉtape 2 - Commentaires nettoyés :")
print(df[['Comment', 'Cleaned_Comment']].head())

# Taille cible par classe
target_per_class = 50000

# Assure-toi que tu as bien les colonnes
assert 'Sentiment' in df.columns

# Échantillonnage aléatoire dans chaque classe
df = pd.concat([
    df[df['Sentiment'] == 0].sample(n=target_per_class, random_state=42),
    df[df['Sentiment'] == 1].sample(n=target_per_class, random_state=42),
    df[df['Sentiment'] == 2].sample(n=target_per_class, random_state=42)
])

# Mélange final
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Vérification
print("Distribution des classes après équilibrage :")
print(df['Sentiment'].value_counts())


print("\nÉtape 2.1 - Création des tokens issus des commentaires nettoyés :")
# Tokenisation et création du vocabulaire
tokenized = df['Cleaned_Comment'].apply(lambda x: x.split())
print("\nÉtape 3 - Exemple de tokens :")
print(tokenized.head())

# Création du vocabulaire
VOCAB_SIZE = 5000  # Taille du vocabulaire
word_counter = Counter([token for row in tokenized for token in row])
most_common = word_counter.most_common(VOCAB_SIZE - 2)
word2idx = {'<PAD>': 0, '<UNK>': 1}
for i, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = i
print("\nÉtape 4 - Taille du vocabulaire (correspondant à un dictionnaire { mot: fréquence }) :", len(word2idx))
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

model = RecurrentNeuronalNetwork(
    input_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
)
print("\nÉtape 7 - Modèle RNN créé :")
print(model)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['Sentiment']),
    y=df['Sentiment']
)
print("Poids des classes (inverse des fréquences) :", class_weights)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.Adam(
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
EPOCHS = 20
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
    if (epoch + 1) % 5 == 0:
        print(f"\n→ Exemple de prédiction après {epoch+1} époques :")
        # Exemple de tweet pour la prédiction intermédiaire
        exemple_tweet = "i love natural language processing, it's fascinating!"
        exemple_clean = preprocess.preprocess(exemple_tweet)
        exemple_tokens = exemple_clean.split()
        exemple_encoded = encode_and_pad(exemple_tokens, word2idx, MAX_LEN)
        tweet_tensor = torch.tensor([exemple_encoded], dtype=torch.long)
        embedded_tweet = embedding_layer(tweet_tensor)
        output = model(embedded_tweet)
        print("Logits :", output)
        print("Prédiction :", torch.argmax(output, dim=1).item())


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
nouveau_tweet = "i love natural language processing, it's fascinating!"
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

nouveaux_tweets_annotes = {
    "I absolutely hated this service, total waste of time.": 0,
    "This movie was amazing, I loved it!": 2,
    "The food was lukewarm, but edible.": 1,
    "Customer support was rude and unhelpful.": 0,
    "Had a fantastic experience, will come back!": 2,
    "Honestly, nothing special.": 1,
    "Really disappointed with the product.": 0,
    "Fast and efficient delivery.": 2,
    "Not great, I expected better.": 0,
    "The staff was polite.": 1,
    "Everything was perfect, thank you!": 2,
    "I'm not impressed.": 0,
    "It's okay, nothing more.": 1,
    "Great atmosphere and good music.": 2,
    "They answered the phone rudely.": 0,
    "Just average, nothing else to say.": 1,
    "It was a real pleasure to go there!": 2,
    "Product arrived broken.": 0,
    "It was a nice evening.": 2,
    "Neither good nor bad, just average.": 1,
    "Very disappointing, I do not recommend.": 0,
    "Neutral experience, no strong opinion.": 1,
    "Wonderful welcome, very friendly!": 2,
    "The website keeps crashing...": 0,
    "Overall good, but nothing exceptional.": 1
}

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for tweet, true_label in nouveaux_tweets_annotes.items():
        tweet_clean = preprocess.preprocess(tweet)
        tweet_tokens = tweet_clean.split()
        tweet_encoded = encode_and_pad(tweet_tokens, word2idx, MAX_LEN)
        tweet_tensor = torch.tensor([tweet_encoded], dtype=torch.long)

        embedded_tweet = embedding_layer(tweet_tensor)
        output = model(embedded_tweet)
        prediction = torch.argmax(output, dim=1).item()

        y_true.append(true_label)
        y_pred.append(prediction)

        print(f"Tweet : {tweet}")
        print(f"→ Vrai : {true_label} | Prédit : {prediction}\n")


# Afficher la matrice de confusion
cm = confusion_matrix(y_true, y_pred)
labels = ["Négatif (0)", "Neutre (1)", "Positif (2)"]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de confusion - Modèle de sentiment")
plt.show()

# Rapport détaillé
print("\nClassification Report :")
print(classification_report(y_true, y_pred, target_names=labels))
