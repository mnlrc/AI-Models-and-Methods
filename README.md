# Projet 1: Recherche adversariale
## Se lancer dans le projet
Les dépendances du projet se trouvent dans le fichier `pyproject.toml`. Nous vous conseillons d'utiliser `uv` pour installer les dépendances dans un environnement virtuel.

```bash
pip install uv # Si uv n'est pas déjà installé
uv sync # Pour installer les dépendances
source .venv/bin/activate # Pour activer l'environnement virtuel
```

## Environnement virtuel
Votre IDE peut être configuré pour utiliser un environnement virtuel spécifique. Par exemple, dans VSCode, vous pouvez sélectionner l'interpréteur Python en appuyant sur `Ctrl+Shift+P` et en recherchant `Python: Select Interpreter`, puis en choisissant l'environnement virtuel `.venv/bin/python`.

Vous pouvez aussi directement utiliser `uv` pour ne pas vous encombrer des détails d'un environnement virtuel:
```bash
uv run python src/main.py # Pour exécuter un script Python
uv run pytest             # Pour lancer les tests
```

## Lancer les tests
Pour lancer les tests, utilisez la commande suivante (qui utilise `pytest`) :
```bash
pytest
```