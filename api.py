from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Permettre les requêtes depuis le frontend

# Charger le modèle au démarrage
print("🤖 Chargement du modèle ML...")
model_data = joblib.load('football_model_v2.pkl')
model = model_data['model']
label_encoder = model_data['label_encoder']
feature_columns = model_data['feature_columns']
print("✅ Modèle chargé avec succès !")

API_KEY = "7a191233823be5517a7135efde992711"
API_BASE_URL = "https://v3.football.api-sports.io"

def make_api_request(endpoint, params=None):
    """Faire une requête à l'API Football"""
    headers = {'x-apisports-key': API_KEY}
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Erreur API: {e}")
        return None

def extract_features_from_api(match_data, home_stats, away_stats, standings, h2h_data, fixture_stats):
    """Extraire les features d'un match pour la prédiction"""
    features = {}
    
    # === STATISTIQUES D'ÉQUIPE DOMICILE ===
    if home_stats:
        # Forme
        features['home_form_points'] = sum([3 if c=='W' else 1 if c=='D' else 0 
                                            for c in home_stats.get('form', '')])
        
        # Matchs
        fixtures_home = home_stats.get('fixtures', {})
        features['home_matches_played'] = fixtures_home.get('played', {}).get('total', 0)
        features['home_wins'] = fixtures_home.get('wins', {}).get('total', 0)
        features['home_draws'] = fixtures_home.get('draws', {}).get('total', 0)
        features['home_losses'] = fixtures_home.get('loses', {}).get('total', 0)
        
        # Buts
        goals_home = home_stats.get('goals', {})
        features['home_goals_for_total'] = goals_home.get('for', {}).get('total', {}).get('total', 0)
        features['home_goals_for_avg'] = float(goals_home.get('for', {}).get('average', {}).get('total', 0) or 0)
        features['home_goals_against_total'] = goals_home.get('against', {}).get('total', {}).get('total', 0)
        features['home_goals_against_avg'] = float(goals_home.get('against', {}).get('average', {}).get('total', 0) or 0)
        features['home_goals_for_home'] = goals_home.get('for', {}).get('total', {}).get('home', 0)
        features['home_goals_against_home'] = goals_home.get('against', {}).get('total', {}).get('home', 0)
        
        # Clean sheets
        features['home_clean_sheets'] = home_stats.get('clean_sheet', {}).get('total', 0)
        features['home_failed_to_score'] = home_stats.get('failed_to_score', {}).get('total', 0)
        
        # Cartons
        cards_home = home_stats.get('cards', {})
        features['home_yellow_cards'] = cards_home.get('yellow', {}).get('0-15', {}).get('total', 0) or 0
        features['home_red_cards'] = cards_home.get('red', {}).get('0-15', {}).get('total', 0) or 0
        
        # Pénalties
        penalty_home = home_stats.get('penalty', {})
        features['home_penalty_scored'] = penalty_home.get('scored', {}).get('total', 0)
        features['home_penalty_missed'] = penalty_home.get('missed', {}).get('total', 0)
    
    # === STATISTIQUES D'ÉQUIPE EXTÉRIEUR ===
    if away_stats:
        # Forme
        features['away_form_points'] = sum([3 if c=='W' else 1 if c=='D' else 0 
                                            for c in away_stats.get('form', '')])
        
        # Matchs
        fixtures_away = away_stats.get('fixtures', {})
        features['away_matches_played'] = fixtures_away.get('played', {}).get('total', 0)
        features['away_wins'] = fixtures_away.get('wins', {}).get('total', 0)
        features['away_draws'] = fixtures_away.get('draws', {}).get('total', 0)
        features['away_losses'] = fixtures_away.get('loses', {}).get('total', 0)
        
        # Buts
        goals_away = away_stats.get('goals', {})
        features['away_goals_for_total'] = goals_away.get('for', {}).get('total', {}).get('total', 0)
        features['away_goals_for_avg'] = float(goals_away.get('for', {}).get('average', {}).get('total', 0) or 0)
        features['away_goals_against_total'] = goals_away.get('against', {}).get('total', {}).get('total', 0)
        features['away_goals_against_avg'] = float(goals_away.get('against', {}).get('average', {}).get('total', 0) or 0)
        features['away_goals_for_away'] = goals_away.get('for', {}).get('total', {}).get('away', 0)
        features['away_goals_against_away'] = goals_away.get('against', {}).get('total', {}).get('away', 0)
        
        # Clean sheets
        features['away_clean_sheets'] = away_stats.get('clean_sheet', {}).get('total', 0)
        features['away_failed_to_score'] = away_stats.get('failed_to_score', {}).get('total', 0)
        
        # Cartons
        cards_away = away_stats.get('cards', {})
        features['away_yellow_cards'] = cards_away.get('yellow', {}).get('0-15', {}).get('total', 0) or 0
        features['away_red_cards'] = cards_away.get('red', {}).get('0-15', {}).get('total', 0) or 0
        
        # Pénalties
        penalty_away = away_stats.get('penalty', {})
        features['away_penalty_scored'] = penalty_away.get('scored', {}).get('total', 0)
        features['away_penalty_missed'] = penalty_away.get('missed', {}).get('total', 0)
    
    # === CLASSEMENT ===
    if standings and len(standings) > 0:
        table = standings[0].get('league', {}).get('standings', [[]])[0]
        
        home_standing = next((t for t in table if t['team']['id'] == match_data['home_team_id']), None)
        away_standing = next((t for t in table if t['team']['id'] == match_data['away_team_id']), None)
        
        if home_standing:
            features['home_rank'] = home_standing['rank']
            features['home_points'] = home_standing['points']
            features['home_goal_diff'] = home_standing['goalsDiff']
        
        if away_standing:
            features['away_rank'] = away_standing['rank']
            features['away_points'] = away_standing['points']
            features['away_goal_diff'] = away_standing['goalsDiff']
    
    # === HEAD TO HEAD ===
    if h2h_data:
        features['h2h_total_matches'] = len(h2h_data)
        home_wins_h2h = sum(1 for m in h2h_data 
                           if (m['teams']['home']['id'] == match_data['home_team_id'] and m['teams']['home']['winner'])
                           or (m['teams']['away']['id'] == match_data['home_team_id'] and m['teams']['away']['winner']))
        features['h2h_home_wins'] = home_wins_h2h
        features['h2h_away_wins'] = len(h2h_data) - home_wins_h2h
    
    # === FEATURES DÉRIVÉES (CRITIQUES) ===
    # Ces features capturent les différences de niveau entre les équipes
    
    if 'home_rank' in features and 'away_rank' in features and features['home_rank'] and features['away_rank']:
        features['rank_difference'] = features['away_rank'] - features['home_rank']
    
    if 'home_points' in features and 'away_points' in features:
        features['points_difference'] = features['home_points'] - features['away_points']
        if features['away_points'] > 0:
            features['points_ratio'] = features['home_points'] / (features['away_points'] + 1)
    
    if 'home_form_points' in features and 'away_form_points' in features:
        features['form_difference'] = features['home_form_points'] - features['away_form_points']
        if features['away_form_points'] > 0:
            features['form_ratio'] = features['home_form_points'] / (features['away_form_points'] + 1)
    
    if 'home_goals_for_avg' in features and 'away_goals_for_avg' in features:
        features['attack_difference'] = features['home_goals_for_avg'] - features['away_goals_for_avg']
        if features['away_goals_for_avg'] > 0:
            features['attack_ratio'] = features['home_goals_for_avg'] / (features['away_goals_for_avg'] + 0.1)
    
    if 'home_goals_against_avg' in features and 'away_goals_against_avg' in features:
        features['defense_difference'] = features['away_goals_against_avg'] - features['home_goals_against_avg']
    
    if 'home_wins' in features and 'away_wins' in features:
        features['wins_difference'] = features['home_wins'] - features['away_wins']
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour faire une prédiction"""
    try:
        data = request.json
        
        home_team_id = data['home_team_id']
        away_team_id = data['away_team_id']
        league_id = data['league_id']
        
        print(f"🎯 Prédiction demandée : {home_team_id} vs {away_team_id} (Ligue {league_id})")
        
        # Déterminer si c'est une coupe
        cup_leagues = [66, 65, 45, 48, 143, 81, 137]
        is_cup = league_id in cup_leagues
        
        # Mapping des ligues principales
        main_leagues = {
            66: 61, 65: 61, 45: 39, 48: 39, 143: 140, 81: 78, 137: 135
        }
        
        stats_league_id = main_leagues[league_id] if is_cup else league_id
        
        # Récupérer les données de l'API
        print("📊 Récupération des statistiques...")
        
        home_stats_data = make_api_request('teams/statistics', {
            'team': home_team_id,
            'league': stats_league_id,
            'season': 2025
        })
        
        away_stats_data = make_api_request('teams/statistics', {
            'team': away_team_id,
            'league': stats_league_id,
            'season': 2025
        })
        
        h2h_data = make_api_request('fixtures/headtohead', {
            'h2h': f"{home_team_id}-{away_team_id}",
            'last': 10
        })
        
        standings_data = make_api_request('standings', {
            'league': stats_league_id,
            'season': 2025
        })
        
        home_stats = home_stats_data.get('response') if home_stats_data else None
        away_stats = away_stats_data.get('response') if away_stats_data else None
        h2h = h2h_data.get('response', []) if h2h_data else []
        standings = standings_data.get('response', []) if standings_data else []
        
        # Extraire les features
        match_data = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'league_id': league_id
        }
        
        features = extract_features_from_api(
            match_data, home_stats, away_stats, standings, h2h, None
        )
        
        # Créer un DataFrame avec toutes les colonnes attendues
        X = pd.DataFrame([features])
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Garder seulement les colonnes du modèle dans le bon ordre
        X = X[feature_columns]
        
        print("🤖 Prédiction en cours...")
        
        # Faire la prédiction
        probas = model.predict_proba(X)[0]
        classes = label_encoder.classes_
        
        # Construire la réponse
        result = {
            'homeWin': float(probas[list(classes).index('H')] * 100),
            'draw': float(probas[list(classes).index('D')] * 100),
            'awayWin': float(probas[list(classes).index('A')] * 100),
            'confidence': float(np.max(probas) * 100),
            'prediction': classes[np.argmax(probas)],
            'model': 'XGBoost ML',
            'accuracy': 81.94
        }
        
        print(f"✅ Prédiction : H={result['homeWin']:.1f}% D={result['draw']:.1f}% A={result['awayWin']:.1f}%")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# IDs des équipes hôtes de la Coupe du Monde 2026 (avantage du terrain)
WC_HOST_TEAM_IDS = {2384, 5529, 16}  # USA, Canada, Mexico

# Points FIFA approximatifs (2025) des 48 équipes qualifiées
# Sert de référence sur le "niveau" réel de l'équipe, indépendamment de ses
# derniers adversaires (une équipe qui écrase des voisins faibles ne doit pas
# paraître aussi forte qu'une grande nation européenne)
WC_FIFA_POINTS = {
    1: 1735,    # Belgium
    2: 1850,    # France
    3: 1715,    # Croatia
    5: 1450,    # Sweden
    6: 1776,    # Brazil
    7: 1701,    # Uruguay
    8: 1709,    # Colombia
    9: 1854,    # Spain
    10: 1819,   # England
    11: 1480,   # Panama
    12: 1652,   # Japan
    13: 1630,   # Senegal
    15: 1635,   # Switzerland
    16: 1641,   # Mexico
    17: 1530,   # South Korea
    20: 1530,   # Australia
    22: 1637,   # Iran
    23: 1450,   # Saudi Arabia
    25: 1716,   # Germany
    26: 1886,   # Argentina
    27: 1750,   # Portugal
    28: 1507,   # Tunisia
    31: 1715,   # Morocco
    32: 1487,   # Egypt
    770: 1480,  # Czech Republic
    775: 1580,  # Austria
    777: 1560,  # Türkiye
    1090: 1500, # Norway
    1108: 1500, # Scotland
    1113: 1450, # Bosnia & Herzegovina
    1118: 1748, # Netherlands
    1501: 1470, # Ivory Coast
    1504: 1410, # Ghana
    1508: 1380, # Congo DR
    1531: 1380, # South Africa
    1532: 1470, # Algeria
    1533: 1380, # Cape Verde Islands
    1548: 1410, # Jordan
    1567: 1390, # Iraq
    1568: 1410, # Uzbekistan
    1569: 1410, # Qatar
    2380: 1450, # Paraguay
    2382: 1622, # Ecuador
    2384: 1672, # USA
    2386: 1280, # Haiti
    4673: 1280, # New Zealand
    5529: 1577, # Canada
    5530: 1300, # Curaçao
}

def get_national_team_form(team_id, last=10):
    """Récupère la forme récente d'une équipe nationale (toutes compétitions confondues)"""
    data = make_api_request('fixtures', {'team': team_id, 'last': last})
    fixtures = data.get('response', []) if data else []

    played = len(fixtures)
    goals_for = 0
    goals_against = 0
    form_points = 0

    for f in fixtures:
        home_id = f['teams']['home']['id']
        gh = f['goals']['home'] or 0
        ga = f['goals']['away'] or 0

        if home_id == team_id:
            scored, conceded = gh, ga
        else:
            scored, conceded = ga, gh

        goals_for += scored
        goals_against += conceded

        if scored > conceded:
            form_points += 3
        elif scored == conceded:
            form_points += 1

    if played == 0:
        return {'goals_for_avg': 1.0, 'goals_against_avg': 1.0, 'form_points': 0, 'played': 0}

    return {
        'goals_for_avg': goals_for / played,
        'goals_against_avg': goals_against / played,
        'form_points': form_points,
        'played': played
    }


@app.route('/predict_wc', methods=['POST'])
def predict_wc():
    """Prédiction simplifiée pour les matchs de Coupe du Monde (équipes nationales)"""
    try:
        data = request.json
        home_team_id = data['home_team_id']
        away_team_id = data['away_team_id']

        print(f"🏆 Prédiction CDM : équipe {home_team_id} vs équipe {away_team_id}")

        home_form = get_national_team_form(home_team_id)
        away_form = get_national_team_form(away_team_id)

        # Historique des confrontations directes (2 derniers matchs)
        h2h_data = make_api_request('fixtures/headtohead', {
            'h2h': f"{home_team_id}-{away_team_id}",
            'last': 2
        })
        h2h_matches = []
        for f in (h2h_data.get('response', []) if h2h_data else []):
            h2h_matches.append({
                'date': f['fixture']['date'][:10],
                'home': f['teams']['home']['name'],
                'away': f['teams']['away']['name'],
                'home_goals': f['goals']['home'],
                'away_goals': f['goals']['away'],
                'competition': f['league']['name']
            })

        home_fifa = WC_FIFA_POINTS.get(home_team_id, 1400)
        away_fifa = WC_FIFA_POINTS.get(away_team_id, 1400)

        home_strength = home_fifa / 20 \
            + (home_form['goals_for_avg'] - home_form['goals_against_avg']) * 5 \
            + home_form['form_points'] * 0.3
        away_strength = away_fifa / 20 \
            + (away_form['goals_for_avg'] - away_form['goals_against_avg']) * 5 \
            + away_form['form_points'] * 0.3

        diff = home_strength - away_strength

        # Avantage du terrain pour les pays hôtes
        if home_team_id in WC_HOST_TEAM_IDS:
            diff += 8
        if away_team_id in WC_HOST_TEAM_IDS:
            diff -= 8

        home_win = max(8, min(82, 40 + diff * 1.2))
        draw = max(10, min(30, 27 - abs(diff) * 0.25))
        away_win = 100 - home_win - draw

        if away_win < 8:
            deficit = 8 - away_win
            away_win = 8
            home_win -= deficit

        total = home_win + draw + away_win
        home_win, draw, away_win = (home_win / total * 100, draw / total * 100, away_win / total * 100)

        probas = {'H': home_win, 'D': draw, 'A': away_win}
        prediction = max(probas, key=probas.get)

        result = {
            'homeWin': float(home_win),
            'draw': float(draw),
            'awayWin': float(away_win),
            'confidence': float(probas[prediction]),
            'prediction': prediction,
            'model': 'Forme récente (Coupe du Monde 2026)',
            'homeForm': home_form,
            'awayForm': away_form,
            'homeFifaPoints': home_fifa,
            'awayFifaPoints': away_fifa,
            'h2h': h2h_matches
        }

        print(f"✅ Prédiction CDM : H={home_win:.1f}% D={draw:.1f}% A={away_win:.1f}%")

        return jsonify(result)

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/wc_groups', methods=['GET'])
def wc_groups():
    """Classements actuels des groupes de la Coupe du Monde 2026"""
    try:
        data = make_api_request('standings', {'league': 1, 'season': 2026})
        standings = data['response'][0]['league']['standings'] if data and data.get('response') else []

        groups = {}
        for grp in standings:
            if not grp:
                continue
            gname = grp[0]['group'].replace('Group ', '')
            if len(gname) != 1:
                continue  # ignore "Ranking of third-placed teams"
            groups[gname] = []
            for t in grp:
                groups[gname].append({
                    'id': t['team']['id'],
                    'name': t['team']['name'],
                    'logo': t['team']['logo'],
                    'rank': t['rank'],
                    'points': t['points'],
                    'played': t['all']['played'],
                    'win': t['all']['win'],
                    'draw': t['all']['draw'],
                    'lose': t['all']['lose'],
                    'gf': t['all']['goals']['for'],
                    'ga': t['all']['goals']['against']
                })

        return jsonify({'groups': groups})
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/wc_results', methods=['GET'])
def wc_results():
    """Résultats des matchs déjà joués de la Coupe du Monde 2026"""
    try:
        data = make_api_request('fixtures', {'league': 1, 'season': 2026, 'status': 'FT-AET-PEN'})
        fixtures = data.get('response', []) if data else []

        results = []
        for f in fixtures:
            results.append({
                'id': f['fixture']['id'],
                'date': f['fixture']['date'][:10],
                'round': f['league']['round'],
                'home': f['teams']['home']['name'],
                'home_id': f['teams']['home']['id'],
                'away': f['teams']['away']['name'],
                'away_id': f['teams']['away']['id'],
                'home_goals': f['goals']['home'],
                'away_goals': f['goals']['away']
            })

        return jsonify({'results': results})
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/wc_knockout', methods=['GET'])
def wc_knockout():
    """Matchs à élimination directe de la Coupe du Monde 2026 (huitièmes, quarts, demis, finale)"""
    try:
        data = make_api_request('fixtures', {'league': 1, 'season': 2026})
        fixtures = data.get('response', []) if data else []

        knockout = []
        for f in fixtures:
            round_name = f['league']['round']
            if 'Group Stage' in round_name:
                continue
            knockout.append({
                'id': f['fixture']['id'],
                'date': f['fixture']['date'],
                'venue': f['fixture']['venue'].get('city') or f['fixture']['venue'].get('name'),
                'round': round_name,
                'status': f['fixture']['status']['short'],
                'home': f['teams']['home']['name'] or 'À déterminer',
                'home_id': f['teams']['home']['id'],
                'away': f['teams']['away']['name'] or 'À déterminer',
                'away_id': f['teams']['away']['id'],
                'home_goals': f['goals']['home'],
                'away_goals': f['goals']['away']
            })

        return jsonify({'knockout': knockout})
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint pour vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'accuracy': 80.56
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 API DE PRÉDICTION FOOTBALL ML")
    print("="*60)
    print("📊 Modèle : XGBoost")
    print("🎯 Accuracy : 80.56%")
    print("🌐 URL : http://localhost:5002")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5002)