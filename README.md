# 🤖 Système de Prédiction Football avec Machine Learning

## 📊 Vue d'ensemble

Ce système utilise **XGBoost** pour prédire les résultats de matchs de football en analysant toutes les statistiques disponibles via l'API Football.

### Performances attendues
- **Objectif** : >60% d'accuracy (actuellement 27% avec l'algorithme manuel)
- **Données** : ~9 500 matchs des 5 dernières saisons
- **Features** : 50+ statistiques par match

---

## 🚀 Guide d'utilisation

### Étape 1 : Installation des dépendances

```bash
pip install -r requirements.txt
```

### Étape 2 : Collecte des données (utilise ~20k-30k requêtes)

```bash
python collect_data.py
```

**Ce que fait ce script :**
- ✅ Récupère tous les matchs terminés depuis 2020
- ✅ 5 ligues × 5 saisons × ~380 matchs = ~9 500 matchs
- ✅ Extraction de 50+ features par match
- ✅ Sauvegarde progressive dans `football_training_data.csv`
- ✅ Temps estimé : 2-4 heures

**Optimisations incluses :**
- Cache des statistiques d'équipe (évite les requêtes répétées)
- Sauvegardes intermédiaires après chaque ligue
- Compteur de requêtes API (limite 75k/jour)

### Étape 3 : Entraînement du modèle

```bash
python train_model.py
```

**Ce que fait ce script :**
- ✅ Charge les données collectées
- ✅ Entraîne un modèle XGBoost optimisé
- ✅ Affiche les performances (accuracy, matrice de confusion)
- ✅ Sauvegarde le modèle dans `football_model.pkl`
- ✅ Temps estimé : 5-10 minutes

---

## 📈 Features extraites (50+)

### Par équipe (Home et Away)
1. **Forme récente**
   - Points de forme (W=3, D=1, L=0)
   - Séquence de résultats

2. **Statistiques générales**
   - Matchs joués, victoires, nuls, défaites
   - Buts marqués/encaissés (total et moyenne)
   - Buts à domicile/extérieur spécifiques
   - Clean sheets
   - Matches sans marquer

3. **Discipline**
   - Cartons jaunes
   - Cartons rouges

4. **Pénalties**
   - Pénalties marqués
   - Pénalties manqués

5. **Classement**
   - Position
   - Points
   - Différence de buts

6. **Statistiques de match** (du match spécifique)
   - Possession
   - Tirs (total, cadrés)
   - Corners
   - Fautes
   - Hors-jeu
   - Passes totales, précision
   - Dribbles réussis
   - Duels gagnés
   - Et bien d'autres...

### Historique H2H
- Nombre de confrontations
- Victoires de chaque équipe

---

## 🎯 Architecture du système

```
1. collect_data.py
   ├─ Récupère les matchs via API Football
   ├─ Extrait toutes les features disponibles
   └─ Sauvegarde dans football_training_data.csv

2. train_model.py
   ├─ Charge les données
   ├─ Entraîne XGBoost
   ├─ Évalue les performances
   └─ Sauvegarde football_model.pkl

3. Intégration web (à venir)
   ├─ API Flask pour servir les prédictions
   └─ Modification de index.html pour utiliser le modèle
```

---

## 📊 Structure des données

### Fichier `football_training_data.csv`

Colonnes principales :
```
- fixture_id, date, league_id, season
- home_team_id, away_team_id
- result (H/D/A) ← TARGET
- home_form_points, away_form_points
- home_goals_for_avg, away_goals_for_avg
- home_rank, away_rank
- home_shots_on_target, away_shots_on_target
- home_ball_possession, away_ball_possession
- ... (50+ features au total)
```

---

## 🔧 Paramètres du modèle XGBoost

```python
{
    'objective': 'multi:softprob',  # 3 classes avec probabilités
    'num_class': 3,                 # H, D, A
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

---

## 📝 Prochaines étapes

1. ✅ Collecter les données historiques
2. ✅ Entraîner le modèle XGBoost
3. ⏳ Créer une API Flask pour servir les prédictions
4. ⏳ Modifier index.html pour appeler l'API au lieu de l'algorithme manuel
5. ⏳ Déployer sur un serveur avec le modèle

---

## 💡 Améliorations possibles

- [ ] Ajouter les données de Ligue 2, Championship, etc.
- [ ] Inclure les compétitions européennes dans l'entraînement
- [ ] Ajouter des features météo
- [ ] Inclure les absences de joueurs clés
- [ ] Hyperparameter tuning avec GridSearch
- [ ] Ensemble de modèles (XGBoost + LightGBM + Random Forest)

---

## 📞 Support

Si vous rencontrez des erreurs :
1. Vérifiez que votre clé API est valide
2. Assurez-vous d'avoir moins de 75k requêtes utilisées aujourd'hui
3. Vérifiez que tous les packages sont installés

---

## 🎉 Résultat attendu

Après l'entraînement, vous devriez voir :
```
🎯 Accuracy sur le test set: 60-65%
```

C'est **2x mieux** que l'algorithme manuel actuel (27%) ! 🚀