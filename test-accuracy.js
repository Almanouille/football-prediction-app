// test-accuracy.js
// Script pour tester l'accuracy du modèle de prédiction

const API_KEY = '7a191233823be5517a7135efde992711';
const API_BASE = 'https://v3.football.api-sports.io';

// Fonction pour faire un délai entre les requêtes
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Fonction pour récupérer les matchs terminés
async function getFinishedMatches(leagueId, season) {
    const response = await fetch(
        `${API_BASE}/fixtures?league=${leagueId}&season=${season}&status=FT`,
        {
            headers: {
                'x-apisports-key': API_KEY
            }
        }
    );
    const data = await response.json();
    return data.response || [];
}

// Fonction pour récupérer les stats d'une équipe
async function getTeamStats(teamId, leagueId, season) {
    const response = await fetch(
        `${API_BASE}/teams/statistics?league=${leagueId}&season=${season}&team=${teamId}`,
        {
            headers: {
                'x-apisports-key': API_KEY
            }
        }
    );
    const data = await response.json();
    return data.response || {};
}

// Fonction pour récupérer H2H
async function getH2H(team1, team2) {
    const response = await fetch(
        `${API_BASE}/fixtures/headtohead?h2h=${team1}-${team2}&last=5`,
        {
            headers: {
                'x-apisports-key': API_KEY
            }
        }
    );
    const data = await response.json();
    return data.response || [];
}

// Fonction pour récupérer le classement
async function getStandings(leagueId, season) {
    const response = await fetch(
        `${API_BASE}/standings?league=${leagueId}&season=${season}`,
        {
            headers: {
                'x-apisports-key': API_KEY
            }
        }
    );
    const data = await response.json();
    return data.response || [];
}

// Algorithme de prédiction AMÉLIORÉ avec 11 features
function calculatePrediction(homeStats, awayStats, h2h, standings, match) {
    let homeScore = 50;
    let awayScore = 50;

    // 1. Forme récente (15%)
    if (homeStats.form && awayStats.form) {
        const homeForm = homeStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
        const awayForm = awayStats.form.split('').reduce((acc, r) => acc + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
        
        const formDiff = homeForm - awayForm;
        homeScore += formDiff * 1.0;
        awayScore -= formDiff * 1.0;
    }

    // 2. Avantage domicile (8%)
    homeScore += 6;

    // 3. Statistiques de buts (12%)
    if (homeStats.goals && awayStats.goals) {
        const homeGoalsAvg = parseFloat(homeStats.goals.for?.average?.home || 0);
        const awayGoalsAvg = parseFloat(awayStats.goals.for?.average?.away || 0);
        const homeDefAvg = parseFloat(homeStats.goals.against?.average?.home || 0);
        const awayDefAvg = parseFloat(awayStats.goals.against?.average?.away || 0);

        homeScore += (homeGoalsAvg - awayDefAvg) * 3;
        awayScore += (awayGoalsAvg - homeDefAvg) * 3;
    }

    // 4. Tirs cadrés (15%)
    if (homeStats.shots && awayStats.shots) {
        const homeShotsOnTarget = parseFloat(homeStats.shots?.for?.on_target?.home || 0);
        const awayShotsOnTarget = parseFloat(awayStats.shots?.for?.on_target?.away || 0);
        const homeDefShotsOnTarget = parseFloat(homeStats.shots?.against?.on_target?.home || 0);
        const awayDefShotsOnTarget = parseFloat(awayStats.shots?.against?.on_target?.away || 0);
        
        homeScore += (homeShotsOnTarget - awayDefShotsOnTarget) * 0.5;
        awayScore += (awayShotsOnTarget - homeDefShotsOnTarget) * 0.5;
    }

    // 5. Corners (10%)
    if (homeStats.corners && awayStats.corners) {
        const homeCorners = parseFloat(homeStats.corners?.for?.home || 0);
        const awayCorners = parseFloat(awayStats.corners?.for?.away || 0);
        const homeDefCorners = parseFloat(homeStats.corners?.against?.home || 0);
        const awayDefCorners = parseFloat(awayStats.corners?.against?.away || 0);
        
        homeScore += (homeCorners - awayDefCorners) * 0.3;
        awayScore += (awayCorners - homeDefCorners) * 0.3;
    }

    // 6. Possession (8%)
    if (homeStats.ball_possession && awayStats.ball_possession) {
        const homePossession = parseFloat(homeStats.ball_possession?.home?.replace('%', '') || 50);
        const awayPossession = parseFloat(awayStats.ball_possession?.away?.replace('%', '') || 50);
        
        const possessionDiff = homePossession - awayPossession;
        homeScore += possessionDiff * 0.2;
        awayScore -= possessionDiff * 0.2;
    }

    // 7. Passes réussies (8%)
    if (homeStats.passes && awayStats.passes) {
        const homePassesAvg = parseFloat(homeStats.passes?.total?.home || 0);
        const awayPassesAvg = parseFloat(awayStats.passes?.total?.away || 0);
        const homePassesAccuracy = parseFloat(homeStats.passes?.accuracy?.home?.replace('%', '') || 0);
        const awayPassesAccuracy = parseFloat(awayStats.passes?.accuracy?.away?.replace('%', '') || 0);
        
        const homePassQuality = (homePassesAvg * homePassesAccuracy) / 100;
        const awayPassQuality = (awayPassesAvg * awayPassesAccuracy) / 100;
        
        homeScore += (homePassQuality - awayPassQuality) * 0.01;
        awayScore -= (homePassQuality - awayPassQuality) * 0.01;
    }

    // 8. Discipline - Cartons (6%)
    if (homeStats.cards && awayStats.cards) {
        const homeYellow = parseFloat(homeStats.cards?.yellow?.home || 0);
        const awayYellow = parseFloat(awayStats.cards?.yellow?.away || 0);
        const homeRed = parseFloat(homeStats.cards?.red?.home || 0);
        const awayRed = parseFloat(awayStats.cards?.red?.away || 0);
        
        const homeCards = homeYellow + (homeRed * 3);
        const awayCards = awayYellow + (awayRed * 3);
        
        homeScore -= homeCards * 0.3;
        awayScore -= awayCards * 0.3;
    }

    // 9. Fautes (5%)
    if (homeStats.fouls && awayStats.fouls) {
        const homeFoulsFor = parseFloat(homeStats.fouls?.for?.home || 0);
        const awayFoulsFor = parseFloat(awayStats.fouls?.for?.away || 0);
        
        homeScore -= homeFoulsFor * 0.1;
        awayScore -= awayFoulsFor * 0.1;
    }

    // 10. Classement (10%)
    if (standings && standings.length > 0 && standings[0].league && standings[0].league.standings) {
        const table = standings[0].league.standings[0];
        const homeRank = table.find(t => t.team.id === match.homeTeamId)?.rank || 10;
        const awayRank = table.find(t => t.team.id === match.awayTeamId)?.rank || 10;
        
        const rankDiff = awayRank - homeRank;
        homeScore += rankDiff * 1.2;
        awayScore -= rankDiff * 1.2;
    }

    // 11. H2H (3%)
    if (h2h.length > 0) {
        const homeWins = h2h.filter(m => 
            (m.teams.home.id === match.homeTeamId && m.teams.home.winner) ||
            (m.teams.away.id === match.homeTeamId && m.teams.away.winner)
        ).length;
        
        homeScore += homeWins * 1.0;
        awayScore += (h2h.length - homeWins) * 1.0;
    }

    homeScore = Math.max(10, homeScore);
    awayScore = Math.max(10, awayScore);

    const total = homeScore + awayScore;
    const homeRatio = homeScore / total;
    const awayRatio = awayScore / total;
    
    const balance = Math.abs(homeRatio - awayRatio);
    let drawProb = Math.round((1 - balance) * 40);
    drawProb = Math.max(5, drawProb);
    
    const remaining = 100 - drawProb;
    let homeWinProb = Math.round(homeRatio * remaining);
    let awayWinProb = remaining - homeWinProb;
    
    homeWinProb = Math.max(1, homeWinProb);
    awayWinProb = Math.max(1, awayWinProb);
    
    let currentTotal = homeWinProb + drawProb + awayWinProb;
    
    if (currentTotal > 100) {
        const factor = 100 / currentTotal;
        homeWinProb = Math.round(homeWinProb * factor);
        awayWinProb = Math.round(awayWinProb * factor);
        drawProb = Math.round(drawProb * factor);
    }
    
    currentTotal = homeWinProb + drawProb + awayWinProb;
    if (currentTotal !== 100) {
        const diff = 100 - currentTotal;
        if (homeWinProb >= awayWinProb && homeWinProb >= drawProb) {
            homeWinProb += diff;
        } else if (awayWinProb >= homeWinProb && awayWinProb >= drawProb) {
            awayWinProb += diff;
        } else {
            drawProb += diff;
        }
    }

    return {
        homeWin: homeWinProb,
        draw: drawProb,
        awayWin: awayWinProb,
        prediction: homeWinProb > awayWinProb && homeWinProb > drawProb ? 'home' : 
                   awayWinProb > homeWinProb && awayWinProb > drawProb ? 'away' : 'draw'
    };
}

// Fonction principale de test
async function testAccuracy() {
    console.log('🔍 Début du test d\'accuracy...\n');
    
    // Ligues à tester
    const leagues = [
        { id: 61, name: 'Ligue 1 🇫🇷' },
        { id: 39, name: 'Premier League 🏴' },
        { id: 140, name: 'La Liga 🇪🇸' }
    ];
    
    // 3 dernières saisons complètes
    const seasons = [2024, 2023, 2022];
    
    console.log('📊 Configuration du test:');
    console.log(`   Ligues: ${leagues.map(l => l.name).join(', ')}`);
    console.log(`   Saisons: ${seasons.join(', ')}\n`);
    
    const allMatches = [];
    
    // Récupérer les matchs de toutes les ligues et saisons
    for (const league of leagues) {
        for (const season of seasons) {
            console.log(`📥 Récupération ${league.name} - Saison ${season}...`);
            const matches = await getFinishedMatches(league.id, season);
            console.log(`   ✅ ${matches.length} matchs trouvés`);
            
            // Ajouter les infos de ligue et saison
            matches.forEach(m => {
                m.leagueInfo = league.name;
                m.seasonInfo = season;
            });
            
            allMatches.push(...matches);
            await delay(1000); // Délai entre chaque requête
        }
    }
    
    console.log(`\n🎯 TOTAL: ${allMatches.length} matchs à analyser\n`);
    console.log('⏳ Analyse en cours (cela peut prendre 15-30 minutes)...\n');
    
    let correct = 0;
    let total = 0;
    const results = [];
    
    // Limiter à 300 matchs maximum pour ne pas exploser le nombre de requêtes
    const matchesToTest = allMatches.slice(0, 300);
    
    for (let i = 0; i < matchesToTest.length; i++) {
        const match = matchesToTest[i];
        
        try {
            // Récupérer le classement de la bonne saison
            const standings = await getStandings(match.league.id, match.seasonInfo);
            
            const matchData = {
                homeTeamId: match.teams.home.id,
                awayTeamId: match.teams.away.id,
                leagueId: match.league.id
            };
            
            const [homeStats, awayStats, h2h] = await Promise.all([
                getTeamStats(matchData.homeTeamId, matchData.leagueId, match.seasonInfo),
                getTeamStats(matchData.awayTeamId, matchData.leagueId, match.seasonInfo),
                getH2H(matchData.homeTeamId, matchData.awayTeamId)
            ]);
            
            await delay(400);
            
            const prediction = calculatePrediction(homeStats, awayStats, h2h, standings, matchData);
            
            let actualResult;
            if (match.teams.home.winner) {
                actualResult = 'home';
            } else if (match.teams.away.winner) {
                actualResult = 'away';
            } else {
                actualResult = 'draw';
            }
            
            const isCorrect = prediction.prediction === actualResult;
            if (isCorrect) correct++;
            total++;
            
            results.push({
                league: match.leagueInfo,
                season: match.seasonInfo,
                match: `${match.teams.home.name} vs ${match.teams.away.name}`,
                predicted: prediction.prediction,
                actual: actualResult,
                correct: isCorrect,
                probabilities: `${prediction.homeWin}% / ${prediction.draw}% / ${prediction.awayWin}%`
            });
            
            if (total % 10 === 0) {
                console.log(`✓ Progression: ${total}/${matchesToTest.length} matchs analysés (${correct} corrects)`);
            }
            
        } catch (error) {
            console.error(`Erreur sur le match ${i + 1}:`, error.message);
        }
    }
    
    // Calcul des métriques
    const accuracy = (correct / total * 100).toFixed(2);
    
    // Statistiques par ligue
    const byLeague = {};
    results.forEach(r => {
        if (!byLeague[r.league]) {
            byLeague[r.league] = { correct: 0, total: 0 };
        }
        byLeague[r.league].total++;
        if (r.correct) byLeague[r.league].correct++;
    });
    
    console.log('\n' + '='.repeat(60));
    console.log('📊 RÉSULTATS FINAUX');
    console.log('='.repeat(60));
    console.log(`Total de matchs testés: ${total}`);
    console.log(`Prédictions correctes: ${correct}`);
    console.log(`Prédictions incorrectes: ${total - correct}`);
    console.log(`\n🎯 ACCURACY GLOBALE: ${accuracy}%`);
    console.log('='.repeat(60));
    
    // Détail par ligue
    console.log('\n📈 Accuracy par ligue:');
    Object.keys(byLeague).forEach(league => {
        const stats = byLeague[league];
        const leagueAccuracy = ((stats.correct / stats.total) * 100).toFixed(2);
        console.log(`   ${league}: ${leagueAccuracy}% (${stats.correct}/${stats.total})`);
    });
    
    // Baseline (hasard)
    const baselineAccuracy = 33.33;
    console.log(`\n📌 Comparaison:`);
    console.log(`   Hasard (baseline): 33.33%`);
    console.log(`   Votre modèle: ${accuracy}%`);
    console.log(`   Amélioration: +${(accuracy - baselineAccuracy).toFixed(2)}%`);
    
    if (parseFloat(accuracy) < 50) {
        console.log('\n⚠️  Accuracy < 50% → Il faut passer à XGBoost pour améliorer !');
    } else {
        console.log('\n✅ Accuracy >= 50% → Le modèle actuel est performant !');
    }
    
    return {
        accuracy: parseFloat(accuracy),
        correct,
        total,
        results,
        byLeague
    };
}

// Exécution
testAccuracy().catch(console.error);