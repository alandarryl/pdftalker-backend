# PDF Talker : backend

Importe un PDF, indexe son contenu, puis pose-lui des questions en langage naturel. Les réponses citent les pages sources et s'affichent en streaming.

Deux modes de fonctionnement, au choix **à chaque document et à chaque question** :

| Mode | Embeddings (recherche) | Génération (réponse) | Avantage |
|---|---|---|---|
| **Local** | Ollama (`bge-m3`) | Ollama (`qwen2.5:1.5b`) | Les documents ne quittent jamais la machine |
| **Cloud** | Gemini (`gemini-embedding-001`) | Gemini (`gemini-3.8-flash`) | Rapide et plus précis |

On peut mélanger : indexer en local et générer avec Gemini, ou l'inverse.

---

## 1. Principe : le RAG

Un modèle ne peut pas « lire » un PDF de 300 pages d'un coup (contexte limité, coût, hallucinations). On utilise donc le **RAG** (Retrieval-Augmented Generation) :

```
        INDEXATION (une fois par document)
PDF ─► extraction du texte par page ─► découpage en chunks ─► embeddings ─► fichier .npy

        QUESTION (à chaque message)
question ─► embedding ─► similarité cosinus avec tous les chunks ─► top-k passages
         ─► prompt (consignes + passages + question) ─► LLM ─► réponse + pages sources
```

1. **Extraction** : PyMuPDF lit le texte page par page (les numéros de page sont conservés pour les citations).
2. **Chunking** : chaque page est découpée en morceaux d'environ 1 200 caractères, avec 200 caractères de chevauchement, en coupant de préférence sur un saut de ligne ou une fin de phrase.
3. **Embeddings** : chaque chunk devient un vecteur (une liste de nombres qui représente son sens).
4. **Recherche** : la question est vectorisée **avec le même modèle**, puis comparée à tous les chunks par similarité cosinus. On garde les `top_k` meilleurs.
5. **Génération** : le LLM reçoit uniquement ces passages et doit répondre à partir d'eux, en citant les pages.

---

## 2. Architecture

```
pdftalker-backend/
├── .env                      # clés et réglages (jamais commité)
├── .gitignore
├── requirements.txt
├── data/                     # généré automatiquement (jamais commité)
│   ├── uploads/              # PDF d'origine : <id>.pdf
│   ├── processed/            # chunks et métadonnées : <id>.json
│   └── vectors/              # index : <id>.npy et <id>.meta.json
└── app/
    ├── main.py               # application FastAPI, CORS, routeurs
    ├── config.py             # lecture du .env et réglages
    ├── api/
    │   ├── documents.py      # upload, chunks, indexation, statut
    │   └── chat.py           # /chat et /chat/stream
    ├── providers/
    │   ├── __init__.py       # get(nom) : choisit ollama ou gemini
    │   ├── errors.py         # ProviderError commune
    │   ├── ollama.py         # embed, chat, chat_stream
    │   └── gemini.py         # embed, chat, chat_stream
    └── services/
        ├── pdf_parser.py     # extraction + détection de PDF scanné
        ├── chunking.py       # découpage en chunks
        └── vector_store.py   # index numpy + recherche cosinus
```

### L'abstraction des providers

Ollama et Gemini exposent la **même interface** :

```python
embed(texts: list[str], task: str = "document") -> list[list[float]]
chat(messages: list[dict]) -> str
chat_stream(messages: list[dict])   # générateur de morceaux de texte
```

`providers.get("ollama" | "gemini")` renvoie le bon module. Ajouter un fournisseur (OpenAI, Claude, Mistral…) revient à écrire un fichier avec ces trois fonctions et à l'ajouter dans `providers/__init__.py`.

Le paramètre `task` (`"document"` ou `"query"`) est utilisé par Gemini, qui optimise les vecteurs différemment pour un passage et pour une question. Ollama l'ignore.

### Le stockage vectoriel

L'index est un simple fichier **numpy** : une matrice `(nombre_de_chunks, dimension)` de vecteurs normalisés. La recherche est un produit matriciel (`matrice @ vecteur_question`), ce qui revient à la similarité cosinus. Pour quelques centaines de chunks, c'est instantané.

À côté, `<id>.meta.json` mémorise le **provider, le modèle et la dimension** utilisés pour créer l'index. Cela permet :
- de vectoriser la question avec le bon provider automatiquement,
- de détecter un index créé avec un autre modèle (erreur claire au lieu d'un plantage).

> **Pourquoi pas ChromaDB ?** Il était prévu au départ, mais son module natif plantait sur la machine de développement (`access violation` Windows lors de l'écriture). Numpy n'a aucune dépendance native, et l'interface `vector_store` reste la même : on pourra la remplacer par Qdrant ou pgvector quand il y aura des milliers de documents.

---

## 3. Installation (Windows PowerShell)

Prérequis : Python 3.12, [Ollama](https://ollama.com) (pour le mode local).

```powershell
cd pdftalker-backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell refuse d'activer le venv :

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`requirements.txt` :

```
fastapi
uvicorn[standard]
python-multipart
pymupdf
httpx
numpy
python-dotenv
```

Sous Linux/macOS : `source .venv/bin/activate` à la place du script `.ps1`.

### Modèles Ollama

```powershell
ollama pull bge-m3          # embeddings (excellent en français, ~1,2 Go)
ollama pull qwen2.5:1.5b    # génération locale (léger)
```

`qwen2.5:3b` ou `7b` donnent de meilleures réponses si la machine le permet. Évite les variantes `-coder`, spécialisées dans le code.

### Fichier `.env`

À créer dans `pdftalker-backend/` (sans espaces autour du `=`, sans guillemets) :

```
GEMINI_API_KEY=ta_cle
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_EMBED_MODEL=gemini-embedding-001

# Optionnel
CHAT_PROVIDER=ollama
CHAT_MODEL=qwen2.5:1.5b
EMBED_MODEL=bge-m3
OLLAMA_URL=http://localhost:11434
```

La clé s'obtient sur [Google AI Studio](https://aistudio.google.com). Le `.env` n'est lu qu'**au démarrage** : redémarre le serveur après chaque modification.

Les noms de modèles Gemini changent souvent. Pour lister ceux disponibles avec ta clé :

```powershell
$key = (Get-Content .env | Select-String "GEMINI_API_KEY").ToString().Split("=",2)[1].Trim()
(Invoke-RestMethod -Uri "https://generativelanguage.googleapis.com/v1beta/models" -Headers @{"x-goog-api-key"=$key}).models | Where-Object { $_.supportedGenerationMethods -contains "generateContent" } | Select-Object -ExpandProperty name
```

### `.gitignore`

```
.venv/
.env
data/
__pycache__/
body.json
sortie.txt
```

---

## 4. Lancer

```powershell
uvicorn app.main:app --reload --reload-dir app
```

- `--reload-dir app` limite la surveillance au code. Sans lui, le serveur se recharge en boucle à cause du venv.
- Documentation interactive : <http://localhost:8000/docs>

> **Attention** : en mode `--reload`, enregistrer un fichier de `app/` redémarre le serveur et **tue l'indexation en cours**. Pour une indexation longue en local, lance `uvicorn app.main:app` sans `--reload`.

---

## 5. API

### Documents

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/documents` | Upload d'un PDF (max 50 Mo). Extrait, découpe, renvoie `id`, nombre de pages et de chunks, `looks_scanned` |
| `GET` | `/documents/{id}` | Métadonnées |
| `GET` | `/documents/{id}/chunks?page=10&limit=20` | Inspecter les chunks (utile pour vérifier l'extraction) |
| `POST` | `/documents/{id}/index?provider=ollama\|gemini` | Lance l'indexation **en arrière-plan** (réponse 202 immédiate) |
| `GET` | `/documents/{id}/status` | Progression de l'indexation |
| `GET` | `/health` | Vérification que le serveur répond |

États renvoyés par `/status` :

```json
{"state": "running", "provider": "gemini", "done": 120, "total": 275}
{"state": "done", "provider": "gemini", "model": "gemini-embedding-001", "dim": 768}
{"state": "error", "error": "..."}
{"state": "not_indexed"}
```

Tant que l'indexation est `running`, **l'ancien index (s'il existe) reste utilisé** : il n'est remplacé qu'à la toute fin. Ne compare donc pas de résultats avant `state: done`.

### Chat

`POST /chat` (réponse complète) et `POST /chat/stream` (streaming) prennent le même corps :

```json
{
  "doc_id": "635ba4418e2d4d2c8127e73fea92d714",
  "question": "Que fait le guérisseur à l'orteil du narrateur ?",
  "top_k": 6,
  "chat_provider": "gemini"
}
```

- `top_k` : nombre de passages envoyés au LLM (1 à 12, défaut 6)
- `chat_provider` : `"ollama"` ou `"gemini"` (sinon valeur de `CHAT_PROVIDER` dans `.env`)

Réponse de `/chat` :

```json
{
  "answer": "Le guérisseur tient le petit orteil gauche... (p. 10).",
  "sources": [{"page": 10, "score": 0.763, "extract": "3.\nJe ne savais pas..."}]
}
```

### Streaming (SSE)

`/chat/stream` renvoie des Server-Sent Events :

```
event: sources
data: [{"page": 10, "score": 0.763, "extract": "..."}]

event: token
data: "Le guérisseur tient le petit"

event: token
data: " orteil gauche..."

event: done
data: {}
```

- `sources` arrive en premier : le front peut afficher les pages tout de suite.
- `token` : morceaux de texte à concaténer.
- `error` : envoyé si le fournisseur échoue en cours de route.
- Les erreurs détectées **avant** le flux (document non indexé, provider inconnu) sont de vraies erreurs HTTP (400, 409, 502).

### Codes d'erreur

| Code | Signification |
|---|---|
| 400 | Identifiant ou provider invalide, fichier non PDF |
| 404 | Document introuvable |
| 409 | Document non indexé, ou index créé avec un autre modèle d'embedding |
| 413 | Fichier trop gros |
| 422 | PDF illisible ou sans texte |
| 502 | Erreur du fournisseur (Ollama injoignable, Gemini en erreur) |

---

## 6. Tester en ligne de commande

```powershell
# 1. Indexer (local)
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/documents/<id>/index?provider=ollama"
# ... ou avec Gemini
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/documents/<id>/index?provider=gemini"

# 2. Suivre l'avancement
while ($true) { Invoke-RestMethod http://127.0.0.1:8000/documents/<id>/status | Out-String; Start-Sleep 10 }

# 3. Streaming (curl.exe, avec le .exe)
chcp 65001
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "@body.json"
```

Swagger et `Invoke-RestMethod` n'affichent pas un flux progressivement : utilise `curl.exe -N`.

---

## 7. Résultats observés

Tests sur un roman de 139 pages (275 chunks), CPU seul, sans GPU.

| | Ollama / bge-m3 | Gemini |
|---|---|---|
| Indexation | ~10 min (≈0,5 chunk/s) | ~2 min (lots de 10 + pause) |
| Question test : rang de la bonne page | 3e | 1re (et pages voisines pertinentes) |
| Question hors sujet | « Je ne trouve pas cette information » | idem |

`nomic-embed-text` a été testé puis abandonné : trop peu discriminant en français littéraire (scores quasi identiques pour tous les passages).

La génération avec `qwen2.5:1.5b` répond juste mais rédige parfois maladroitement ; Gemini rédige nettement mieux. Ce sont des tests sur un seul document : des signes encourageants, pas une évaluation.

---

## 8. Choix de conception à connaître

- **Indexation asynchrone** : sur CPU, l'embedding de 275 chunks dure plusieurs minutes. L'endpoint répond immédiatement (202) et une tâche de fond avance, avec un compteur `done/total`. Ce sera la base de la barre de progression du front.
- **Gemini et les quotas** : lots de 10 textes avec 4 s de pause (les lots de 50 dépassaient le quota gratuit, erreur 429). Les erreurs 429, 500 et 503 sont réessayées avec des pauses croissantes (2, 4, 8 s).
- **Réessais du streaming** : uniquement tant qu'aucun token n'a été envoyé, pour ne jamais dupliquer du texte.
- **Anti prompt injection** : le contenu du document est placé dans `<document>…</document>`, et le prompt système indique que c'est une donnée, pas des instructions. C'est une atténuation, pas une garantie.
- **Réponse « je ne sais pas »** : le prompt impose une phrase précise quand l'information est absente des passages. Testé avec Ollama et Gemini.
- **Confidentialité** : en mode Gemini, le texte des passages (et de tout le document à l'indexation) est envoyé à Google. N'indexe pas de documents sensibles avec Gemini ; l'offre gratuite peut servir à améliorer les produits de Google. Le mode 100 % Ollama garde tout en local.

---

## 9. Limites connues

- **Un seul index par document** : réindexer avec un autre provider écrase le précédent.
- **État d'indexation en mémoire** : un redémarrage du serveur perd la progression d'une indexation en cours (relancer `/index`). `/status` retombe alors sur l'état du fichier d'index.
- **Chunking par page** : une phrase à cheval sur deux pages est coupée. Compromis volontaire pour garder des citations de page exactes.
- **PDF scannés** : détectés (`looks_scanned`) mais pas encore traités. L'OCR est à faire.
- **Tableaux, colonnes, en-têtes** : l'extraction a été validée sur un roman (cas facile). Elle reste à tester sur des documents administratifs.
- **Pas de seuil de pertinence** : on envoie toujours les `top_k` meilleurs passages, même s'ils sont mauvais. Les scores sont bas quand rien ne correspond (≈0,4 contre ≈0,75), un seuil est envisageable mais à calibrer sur plusieurs documents.
- **Pas d'historique de conversation** : chaque question est indépendante.
- **Pas d'authentification**, pas de suppression de documents.
- **Un jeu d'évaluation manque** : 10 questions avec la page attendue permettraient de mesurer chaque changement (chunking, modèle, `top_k`).

---

## 10. Problèmes rencontrés et solutions

| Symptôme | Cause | Solution |
|---|---|---|
| `source` introuvable | Commande Linux sous PowerShell | `.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError` alors que le paquet est installé | Plusieurs venvs mélangés | Un seul `.venv`, `python -m pip install` |
| Serveur qui redémarre en boucle | `--reload` surveille le venv | `--reload-dir app` |
| Serveur qui meurt sans message pendant l'indexation | ChromaDB (`access violation`) | Remplacé par numpy |
| `Failed to fetch` sur `/index` | Requête de plusieurs minutes | Indexation en arrière-plan |
| 500 sur `/chat` après changement de modèle d'embedding | Dimensions incompatibles (768 vs 1024) | Contrôle de dimension → 409 clair |
| `/status` répond `done` à tort | Ancien `.npy` encore présent | `.meta.json` mémorise provider et modèle |
| Résultats « Gemini » identiques à ceux d'avant | Test fait avant la fin de l'indexation | Attendre `state: done` |
| `Erreur Ollama 404 model not found` | Modèle retiré d'Ollama | `ollama pull <modèle>` |
| Gemini 404 « no longer available » | Nom de modèle périmé | Mettre le nom actuel dans `.env` |
| Gemini 429 en indexation | Quota (lots trop gros) | Lots de 10 + pause |
| Gemini 503 | Surcharge temporaire | Réessais automatiques |
| `'builtin_function_or_method' has no attribute 'sleep'` | `from time import time` | `import time` |
| Accents illisibles dans la console | Console non UTF-8 | `chcp 65001` |
| `Duplicate Operation ID` | Routeur inclus deux fois | Un seul `include_router` par routeur |

---

## 11. Feuille de route

1. **Frontend Next.js** : upload, choix du mode, barre de progression, chat en streaming, sources cliquables.
2. Visionneuse PDF avec passages surlignés (citations cliquables).
3. Un index par provider et par document ; choix du mode à l'import.
4. Recherche hybride (vectorielle + mots-clés BM25) et reranking.
5. OCR pour les PDF scannés.
6. Historique des conversations, multi-documents.
7. Résumés automatiques, extraction structurée.
8. Authentification, suppression, Docker (`docker compose up`).
9. Jeu d'évaluation automatique.