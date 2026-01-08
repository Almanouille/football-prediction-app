// api/predict.js
const API_KEY = '7a191233823be5517a7135efde992711';
const API_BASE = 'https://v3.football.api-sports.io';

export default async function handler(req, res) {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Méthode non autorisée' });
    }

    const { homeTeamId, awayTeamId, leagueId } = req.body;

    if (!homeTeamId || !awayTeamId || !leagueId) {
        return res.status(400).json({ error: 'Données manquantes' });
    }

    try {
        // Récupérer les statistiques des deux équipes
        const [homeStatsResponse, awayStatsResponse, h2hResponse] = await Promise.all([
            fetchTeamStats(homeTeamId, leagueId),
            fetchTeamStats(awayTeamId, leagueId),
            fetchH2H(homeTeamId, awayTeamId)
        ]);

        // Calculer la prédiction
        const prediction = calculatePrediction(
            homeStatsResponse,
            awayStatsResponse,
            h2hResponse
        );

        res.status(200).json(prediction);

    } catch (error) {
        console.error('Error making prediction:', error);
        res.status(500).json({ error: 'Erreur lors de la prédiction', details: error.message });
    }
}

async function fetchTeamStats(teamId, leagueId) {
    try {
        // Essayer 2025 d'abord
        let response = await fetch(
            `${API_BASE}/teams/statistics?league=${leagueId}&season=2025&team=${teamId}`,
            {
                headers: {
                    'x-apisports-key': API_KEY,
                },
            }
        );
        let data = await response.json();

        // Si pas de données, essayer 2024
        if (!data.response || Object.keys(data.response).length === 0) {
            response = await fetch(
                `${API_BASE}/teams/statistics?league=${leagueId}&season=2024&team=${teamId}`,
                {
                    headers: {
                        'x-apisports-key': API_KEY,
                    },
                }
            );
            data = await response.json();
        }

        return data.response || {};
    } catch (error) {
        console.error('Error fetching team stats:', error);
        return {};
    }
}

async function fetchH2H(team1, team2) {
    try {
        const response = await fetch(
            `${API_BASE}/fixtures/headtohead?h2h=${team1}-${team2}&last=5`,
            {
                headers: {
                    'x-apisports-key': API_KEY,
                },
            }
        );
        const data = await response.json();
        return data.response || [];
    } catch (error) {
        console.error('Error fetching H2H:', error);
        return [];
    }
}

function calculatePrediction(homeStats, awayStats, h2h) {
    let homeScore = 50;
    let awayScore = 50;

    // Analyse de la forme (si disponible)
    if (homeStats.form && awayStats.form) {
        const homeForm = homeStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
        const awayForm = awayStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
        
        const formDiff = homeForm - awayForm;
        homeScore += formDiff * 2.5;
        awayScore -= formDiff * 2.5;
    }

    // Avantage domicile
    homeScore += 12;

    // Statistiques de buts
    if (homeStats.goals && awayStats.goals) {
        const homeGoalsAvg = parseFloat(homeStats.goals.for?.average?.home || 0);
        const awayGoalsAvg = parseFloat(awayStats.goals.for?.average?.away || 0);
        const homeDefAvg = parseFloat(homeStats.goals.against?.average?.home || 0);
        const awayDefAvg = parseFloat(awayStats.goals.against?.average?.away || 0);

        homeScore += (homeGoalsAvg - awayDefAvg) * 8;
        awayScore += (awayGoalsAvg - homeDefAvg) * 8;
    }

    // Historique H2H
    if (h2h.length > 0) {
        const homeWins = h2h.filter(m => m.teams.home.winner || m.teams.away.winner).length;
        homeScore += homeWins * 3;
        awayScore += (h2h.length - homeWins) * 3;
    }

    // Normaliser les scores en pourcentages
    const total = homeScore + awayScore;
    let homeWinProb = Math.round((homeScore / total) * 100);
    let awayWinProb = Math.round((awayScore / total) * 100);
    
    // Ajuster pour que le total fasse 100%
    if (homeWinProb + awayWinProb > 100) {
        awayWinProb = 100 - homeWinProb;
    }
    
    const drawProb = Math.max(15, Math.min(30, 100 - homeWinProb - awayWinProb));
    
    // Réajuster si nécessaire
    const adjustment = (homeWinProb + awayWinProb + drawProb - 100) / 2;
    homeWinProb = Math.max(0, Math.round(homeWinProb - adjustment));
    awayWinProb = Math.max(0, Math.round(awayWinProb - adjustment));

    // Calculer l'indice de confiance
    const maxProb = Math.max(homeWinProb, awayWinProb, drawProb);
    const confidence = Math.min(92, Math.max(55, maxProb + 15));

    // Calculer les valeurs pour l'analyse
    const homeForm = homeStats.form ? homeStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0) : 0;
    const awayForm = awayStats.form ? awayStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0) : 0;
    const homeAttack = homeStats.goals?.for?.average?.home || 0;
    const awayAttack = awayStats.goals?.for?.average?.away || 0;

    return {
        homeWin: homeWinProb,
        draw: drawProb,
        awayWin: awayWinProb,
        confidence: confidence,
        prediction: homeWinProb > awayWinProb && homeWinProb > drawProb ? 'home' : 
                   awayWinProb > homeWinProb && awayWinProb > drawProb ? 'away' : 'draw',
        analysis: {
            homeForm: homeForm,
            awayForm: awayForm,
            homeAttack: homeAttack,
            awayAttack: awayAttack
        }
    };
}