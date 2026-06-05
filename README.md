# Cross Approximation Project

Ce projet étudie des méthodes de **Cross Approximation** pour construire des approximations de rang faible de matrices denses, en particulier des matrices synthétiques et des matrices noyau construites à partir de jeux de données réels.

Les expériences comparent principalement :

- **fpCA** : full pivoting Cross Approximation
- **ppCA** : partial pivoting Cross Approximation
- **ppCA adaptive** : version adaptative avec critère d'arrêt
- **SVD** : référence optimale pour l'erreur en norme de Frobenius

## Structure du projet

```text
.
├── approximation.py      # Algorithmes de Cross Approximation
├── error_analysis.py     # Calcul des erreurs relatives et mesures de temps
├── data.py               # Chargement des données et matrices noyau
├── synthetic.py          # Génération de matrices synthétiques
├── tests.py              # Expériences principales et génération de figures PDF
├── tests_pdf.py          # Version alternative avec arguments de ligne de commande
├── plots/                # Figures générées
└── numerical_experiments_updated.tex
```

## Installation

Le projet contient déjà un environnement virtuel `.venv`. Depuis le dossier du projet, active-le avec :

```bash
source .venv/bin/activate
```

Puis lance le script avec :

```bash
python tests.py
```

Si tu n'utilises pas l'environnement virtuel, Python peut ne pas trouver certaines bibliothèques comme `sklearn`.

## Dépendances

Les principales bibliothèques utilisées sont :

- `numpy`
- `matplotlib`
- `scikit-learn`

Si besoin, elles peuvent être installées avec :

```bash
pip install numpy matplotlib scikit-learn
```

## Lancer les expériences

Pour exécuter toutes les expériences principales :

```bash
python tests.py
```

Les figures sont sauvegardées dans le dossier :

```text
plots/
```

Pour lancer la version PDF avec options :

```bash
python tests_pdf.py
```

Il est aussi possible de sauter le test California dans `tests_pdf.py` :

```bash
python tests_pdf.py --skip-california
```

## Expériences incluses

### Iris

Le projet teste deux matrices noyau sur le jeu de données Iris :

- noyau gaussien
- noyau linéaire

### Matrices synthétiques

Les matrices synthétiques permettent de contrôler la décroissance des valeurs singulières :

- décroissance exponentielle
- décroissance polynomiale
- matrice PSD bruitée de rang faible

### California Housing

Le test California utilise le jeu de données California Housing. Les variables sont standardisées avec `StandardScaler`, puis une matrice noyau gaussienne est construite avec :

```python
sigma = np.sqrt(x.shape[1])
```

Ce choix évite que le noyau soit trop proche de l'identité, ce qui rendrait l'erreur lente à diminuer pour des rangs faibles.

## Résultats

Les graphes comparent l'erreur relative en norme de Frobenius en fonction du rang. La courbe SVD sert de référence, car elle donne la meilleure approximation de rang faible pour cette norme.

Les résultats montrent que :

- fpCA est généralement plus stable mais plus coûteuse
- ppCA est plus rapide mais dépend davantage du choix des pivots
- ppCA adaptive permet d'arrêter l'approximation selon la taille des mises à jour
- le choix des paramètres du noyau, en particulier `sigma`, influence fortement la décroissance de l'erreur

## Commandes Git utiles

Voir les fichiers modifiés :

```bash
git status
```

Ajouter les changements :

```bash
git add README.md approximation.py tests.py tests_pdf.py plots
```

Créer un commit :

```bash
git commit -m "Update cross approximation experiments"
```

Envoyer sur GitHub :

```bash
git push origin main
```
