# RwaMarket (rwamarket.rwazap.com) - Instructions Globales OpenCode

## Vision du Projet
Application web personnelle d'analyse financière Multi-Agents et de Paper Trading (Crypto 24/7 via Binance/CoinGecko + Actions US via Alpaca/Yahoo Finance), fusionnant :
1. `staskh/trading_skills` : Calculs techniques purs en Python (RSI, MACD, EMA 9/20/50/200, Bollinger, ATR, Matrice de corrélation, et Grecques Black-Scholes via chaînes d'options yfinance).
2. `Hainrixz/maia-skill` : Cascade de données gratuites sans clé obligatoire, adaptation au profil de risque (Conservative, Moderate, Aggressive) et contrat JSON strict.
3. `dragon1086/prism-insight` : Screener mathématique en amont (filtre volume/momentum avant d'appeler un LLM) + Mémoire auto-apprenante (`trading_memory` qui compresse les leçons des trades clôturés).
4. `Open-Finance-Lab/AgenticTrading` : Moteur de Paper Trading (portefeuille simulé, ordres, Stop-Loss/Take-Profit automatiques, trace complète des raisonnements, Win Rate, Drawdown, Sharpe).
5. `TauricResearch/Trading-R1` : Débat structuré Bull vs Bear tranché par un Risk Manager, Scheduler de rapports automatiques (Morning Brief) et alertes Telegram.

## Stack Technique & Règles Strictes
- **Langage & Framework** : Python 3.12, FastAPI, Uvicorn, `httpx`, `asyncio`, `yfinance`, `pandas`, `numpy`, `scipy`, `apscheduler`.
- **Base de données** : SQLite3 en mode WAL (`data/rwamarket.db`) pour éviter tout blocage en lecture/écriture concurrente.
- **Coût LLM minimal** :
  - ZÉRO appel LLM pour calculer des indicateurs techniques ou filtrer le marché. Tout calcul mathématique se fait dans `app/engine/quant_tools.py`.
  - Cache SQLite obligatoire par `(symbol, timeframe, risk_profile)` avec TTL de 30 minutes par défaut.
  - Architecture en entonnoir : 3 agents rapides en parallèle (`FAST_MODEL` = DeepSeek V4.1 Flash / Qwen 3.8 Flash) + 1 agent Juge/Stratège (`REASONING_MODEL` = DeepSeek 4 Pro / R1).
- **Zéro Latence Ressentie (Streaming SSE)** :
  - La route `/api/analyze/{symbol}/stream` utilise Server-Sent Events (SSE).
  - Étape 1 (< 0.5s) : Envoie `event: quant` avec OHLCV, indicateurs, supports/résistances.
  - Étape 2 (Parallèle `asyncio.gather`) : Envoie `event: agent_tech`, `event: agent_macro`, `event: agent_sentiment` dès que chaque agent Flash termine.
  - Étape 3 : Envoie `event: synthesis` avec la décision finale, le débat Bull/Bear et le plan de trade (Entrée, SL, TP, R:R).
- **Sécurité** : Toutes les routes sensibles et l'interface sont protégées par un mot de passe / cookie de session défini par `APP_PASSWORD` dans `.env`.
- **Qualité du code** : Aucun placeholder (`# TODO`, `pass`), gestion d'erreurs robuste (fallback automatique si une API externe échoue).
