// api/matches.js
const API_KEY = '7a191233823be5517a7135efde992711';
const API_BASE = 'https://v3.football.api-sports.io';

export default async function handler(req, res) {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    const { league } = req.query;

    if (!league) {
        return res.status(400).json({ error: 'League ID requis' });
    }

    try {
        // Essayer la saison 2025 d'abord
        let response = await fetch(
            `${API_BASE}/fixtures?league=${league}&season=2025&next=10`,
            {
                headers: {
                    'x-apisports-key': API_KEY,
                },
            }
        );

        let data = await response.json();

        // Si erreur ou pas de résultats, essayer 2024
        if (!data.response || data.response.length === 0) {
            response = await fetch(
                `${API_BASE}/fixtures?league=${league}&season=2024&next=10`,
                {
                    headers: {
                        'x-apisports-key': API_KEY,
                    },
                }
            );
            data = await response.json();
        }

        if (data.errors && Object.keys(data.errors).length > 0) {
            return res.status(500).json({ error: 'Erreur API Football', details: data.errors });
        }

        if (!data.response || data.response.length === 0) {
            return res.status(404).json({ error: 'Aucun match trouvé' });
        }

        // Formater les données pour le frontend
        const matches = data.response.map(match => ({
            id: match.fixture.id,
            date: match.fixture.date,
            leagueId: match.league.id,
            homeName: match.teams.home.name,
            awayName: match.teams.away.name,
            homeLogo: match.teams.home.logo,
            awayLogo: match.teams.away.logo,
            homeTeamId: match.teams.home.id,
            awayTeamId: match.teams.away.id,
            homeForm: null, // Sera récupéré lors de la prédiction
            awayForm: null
        }));

        res.status(200).json({ matches });

    } catch (error) {
        console.error('Error fetching matches:', error);
        res.status(500).json({ error: 'Erreur serveur', details: error.message });
    }
}