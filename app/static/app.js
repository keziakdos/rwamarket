(() => {
  "use strict";

  const config = window.RWA_CONFIG || {};
  const state = {
    symbol: config.defaultSymbol || "BTCUSDT",
    timeframe: "1d",
    risk: config.defaultRisk || "moderate",
    mode: "auto",
    source: null,
    quant: null,
    agents: {},
    synthesis: null,
    chart: null,
    candleSeries: null,
    ema9Series: null,
    ema20Series: null,
    priceLines: [],
    latencyStart: 0,
    quantLatency: null,
    screenerRows: [],
    screenerSort: { key: "surge_score", dir: "desc" },
    streaming: false,
  };

  const $ = (id) => document.getElementById(id);

  // -------------------------------------------------------------------------
  // Utilitaires
  // -------------------------------------------------------------------------
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function toNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function num(value, digits = 2) {
    const number = toNumber(value);
    if (number === null) return "n/a";
    return number.toLocaleString("fr-BE", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtPrice(value) {
    const number = toNumber(value);
    if (number === null) return "n/a";
    const digits = Math.abs(number) >= 100 ? 2 : Math.abs(number) >= 1 ? 4 : 6;
    return num(number, digits);
  }

  function fmtMoney(value) {
    if (value == null) return "n/a";
    return num(value, 2) + " $";
  }

  function fmtSigned(value, digits = 2) {
    const number = toNumber(value);
    if (number === null) return "n/a";
    return (number >= 0 ? "+" : "") + num(number, digits);
  }

  function fmtPct(value, digits = 2) {
    const number = toNumber(value);
    if (number === null) return "n/a";
    return fmtSigned(number, digits) + " %";
  }

  function tone(value) {
    const number = toNumber(value);
    if (number === null) return "text-slate-300";
    if (number > 0) return "text-emerald-300";
    if (number < 0) return "text-rose-300";
    return "text-slate-300";
  }

  function biasClasses(bias) {
    if (bias === "bullish") return "border-emerald-500/35 bg-emerald-500/10 text-emerald-300";
    if (bias === "bearish") return "border-rose-500/35 bg-rose-500/10 text-rose-300";
    return "border-slate-600/40 bg-slate-600/10 text-slate-300";
  }

  function sideClasses(side) {
    return side === "short"
      ? "border-rose-500/35 bg-rose-500/10 text-rose-300"
      : "border-emerald-500/35 bg-emerald-500/10 text-emerald-300";
  }

  function setStatus(text, active = false) {
    const status = $("system-status");
    if (status) status.textContent = text;
    const dot = $("system-dot");
    if (dot) {
      dot.className = "h-2 w-2 rounded-full " + (active ? "bg-teal-400 pulse-dot" : "bg-emerald-400");
    }
  }

  function setLatency(text) {
    const element = $("latency-indicator");
    if (element) element.textContent = text;
  }

  function toast(message, error = false) {
    const element = $("toast");
    if (!element) return;
    element.textContent = message;
    element.className =
      "toast-enter fixed bottom-5 right-5 z-50 max-w-sm rounded-xl border px-4 py-3 text-sm shadow-2xl " +
      (error
        ? "border-rose-500/40 bg-rose-950/95 text-rose-100"
        : "border-teal-500/30 bg-slate-900/95 text-teal-100");
    element.classList.remove("hidden");
    clearTimeout(window.__rwaToastTimer);
    window.__rwaToastTimer = setTimeout(() => element.classList.add("hidden"), 3800);
  }

  async function api(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    if (response.status === 401) {
      window.location = "/login";
      throw new Error("Session expirée.");
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || data.message || "HTTP " + response.status);
    }
    return data;
  }

  function metricCard(label, value, valueTone = "text-slate-100", hint = "") {
    return (
      '<div class="soft rounded-xl px-3 py-2.5">' +
      '<div class="text-[9px] uppercase tracking-[.12em] text-slate-500">' + escapeHtml(label) + "</div>" +
      '<div class="metric-value mt-1 text-sm font-semibold ' + valueTone + '">' + escapeHtml(value) + "</div>" +
      (hint ? '<div class="mt-0.5 text-[9px] text-slate-500">' + escapeHtml(hint) + "</div>" : "") +
      "</div>"
    );
  }

  // -------------------------------------------------------------------------
  // Graphique TradingView Lightweight Charts
  // -------------------------------------------------------------------------
  function initChart() {
    const container = $("chart-container");
    if (!container || typeof window.LightweightCharts === "undefined") {
      const empty = $("chart-empty");
      if (empty) empty.textContent = "Graphique indisponible : CDN TradingView non chargé.";
      return;
    }

    const library = window.LightweightCharts;
    state.chart = library.createChart(container, {
      width: Math.max(container.clientWidth, 320),
      height: Math.max(container.clientHeight, 400),
      layout: {
        background: { type: "solid", color: "rgba(2,6,23,0)" },
        textColor: "#94a3b8",
        fontFamily: "Inter, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(148,163,184,.055)" },
        horzLines: { color: "rgba(148,163,184,.055)" },
      },
      rightPriceScale: { borderColor: "rgba(148,163,184,.16)" },
      timeScale: {
        borderColor: "rgba(148,163,184,.16)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 4,
        barSpacing: 7,
      },
      crosshair: {
        mode: library.CrosshairMode ? library.CrosshairMode.Normal : 0,
        vertLine: { color: "rgba(45,212,191,.35)", labelBackgroundColor: "#0f766e" },
        horzLine: { color: "rgba(45,212,191,.25)", labelBackgroundColor: "#0f766e" },
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });

    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#34d399",
      wickDownColor: "#f87171",
      priceLineVisible: false,
      lastValueVisible: true,
    });
    state.ema9Series = state.chart.addLineSeries({
      color: "#38bdf8",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: "EMA 9",
    });
    state.ema20Series = state.chart.addLineSeries({
      color: "#f59e0b",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: "EMA 20",
    });

    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver((entries) => {
        if (!state.chart || !entries.length) return;
        const width = Math.max(entries[0].contentRect.width, 320);
        state.chart.applyOptions({ width });
      });
      observer.observe(container);
    }
  }

  function toChartTime(timestamp) {
    const parsed = Date.parse(timestamp);
    if (Number.isFinite(parsed)) return Math.floor(parsed / 1000);
    const numeric = Number(timestamp);
    if (Number.isFinite(numeric)) {
      return numeric > 10_000_000_000 ? Math.floor(numeric / 1000) : Math.floor(numeric);
    }
    return null;
  }

  function computeEma(values, period) {
    if (!values.length) return [];
    const multiplier = 2 / (period + 1);
    let previous = values[0];
    return values.map((value, index) => {
      previous = index === 0 ? value : value * multiplier + previous * (1 - multiplier);
      return { value: previous, ready: index >= period - 1 };
    });
  }

  function clearChart() {
    state.priceLines = [];
    if (state.candleSeries) state.candleSeries.setData([]);
    if (state.ema9Series) state.ema9Series.setData([]);
    if (state.ema20Series) state.ema20Series.setData([]);
    const empty = $("chart-empty");
    if (empty) empty.classList.remove("hidden");
  }

  function drawChart(candles) {
    if (!state.candleSeries || !Array.isArray(candles) || !candles.length) {
      clearChart();
      return;
    }

    const byTime = new Map();
    candles.forEach((candle) => {
      const time = toChartTime(candle.timestamp);
      const close = toNumber(candle.close);
      if (time === null || close === null) return;
      byTime.set(time, {
        time,
        open: toNumber(candle.open),
        high: toNumber(candle.high),
        low: toNumber(candle.low),
        close,
      });
    });
    const rows = Array.from(byTime.values()).sort((a, b) => a.time - b.time);
    if (!rows.length) {
      clearChart();
      return;
    }

    const candleData = rows.filter(
      (row) =>
        row.open !== null &&
        row.high !== null &&
        row.low !== null &&
        row.close !== null &&
        row.high >= row.low
    );
    state.candleSeries.setData(candleData);

    const closeValues = rows.map((row) => row.close);
    const ema9 = computeEma(closeValues, 9);
    const ema20 = computeEma(closeValues, 20);
    state.ema9Series.setData(
      rows
        .map((row, index) => (ema9[index].ready ? { time: row.time, value: ema9[index].value } : null))
        .filter(Boolean)
    );
    state.ema20Series.setData(
      rows
        .map((row, index) => (ema20[index].ready ? { time: row.time, value: ema20[index].value } : null))
        .filter(Boolean)
    );

    state.priceLines = [];
    state.chart.timeScale().fitContent();
    const empty = $("chart-empty");
    if (empty) empty.classList.add("hidden");
  }

  function drawTradePlanLines(plan) {
    if (!state.candleSeries || !plan) return;
    state.priceLines.forEach((line) => {
      try {
        state.candleSeries.removePriceLine(line);
      } catch (error) {
        /* ligne déjà retirée */
      }
    });
    state.priceLines = [];

    const definitions = [
      { value: plan.entry_price, color: "#38bdf8", title: "Entrée", style: 0 },
      { value: plan.stop_loss, color: "#f43f5e", title: "Stop-Loss", style: 2 },
      { value: plan.take_profit_1, color: "#10b981", title: "TP1", style: 2 },
      { value: plan.take_profit_2, color: "#84cc16", title: "TP2", style: 2 },
    ];
    definitions.forEach((definition) => {
      const price = toNumber(definition.value);
      if (price === null || price <= 0) return;
      const line = state.candleSeries.createPriceLine({
        price,
        color: definition.color,
        lineWidth: 1,
        lineStyle: definition.style,
        axisLabelVisible: true,
        title: definition.title,
      });
      state.priceLines.push(line);
    });
  }

  // -------------------------------------------------------------------------
  // Métriques quantitatives
  // -------------------------------------------------------------------------
  function renderQuant(quant) {
    state.quant = quant;
    state.symbol = quant.symbol || state.symbol;
    state.quantLatency = Math.round(performance.now() - state.latencyStart);

    const title = $("symbol-title");
    if (title) title.textContent = quant.symbol || state.symbol;
    const market = $("symbol-market");
    if (market) {
      market.textContent = quant.market_type === "crypto" ? "Crypto" : "Actions US";
      market.className =
        "rounded-md border px-2 py-0.5 text-[10px] uppercase tracking-wide " +
        (quant.market_type === "crypto"
          ? "border-teal-500/30 bg-teal-500/10 text-teal-300"
          : "border-blue-500/30 bg-blue-500/10 text-blue-300");
    }
    const subtitle = $("symbol-subtitle");
    if (subtitle) {
      subtitle.textContent =
        (quant.timeframe || state.timeframe) +
        " · " +
        (quant.candles_count || 0) +
        " bougies · " +
        (quant.trend || "tendance n/a") +
        " · source " +
        (quant.cached ? "cache" : "live");
    }
    const priceElement = $("current-price");
    if (priceElement) priceElement.textContent = fmtPrice(quant.price);
    const changeElement = $("price-change");
    const change = toNumber((quant.price_snapshot || {}).change_pct_24h);
    if (changeElement) {
      changeElement.textContent = change === null ? "--" : fmtPct(change) + " sur 24h";
      changeElement.className = "text-xs font-medium " + tone(change);
    }

    const indicators = quant.indicators || {};
    const volume = indicators.volume || {};
    const macd = indicators.macd || {};
    const levels = quant.levels || {};
    const metrics = [
      ["Score Screener", num(quant.technical_score, 1) + " /100", tone((toNumber(quant.technical_score) || 0) - 50), quant.technical_grade || ""],
      ["RSI (14)", num(quant.rsi14, 1), rsiTone(quant.rsi14)],
      ["MACD", num(macd.hist, 4), tone(macd.hist), "histogramme"],
      ["ATR (14)", fmtPrice(quant.atr14), "text-slate-100", num(quant.atr_pct, 2) + " % du prix"],
      ["Volatilité annualisée", fmtPct(indicators.volatility_annualized_pct, 1), "text-slate-100"],
      ["Ratio de volume", num(volume.ratio, 2) + "x", tone((toNumber(volume.ratio) || 1) - 1), "vs moyenne 20"],
      ["Support clé", fmtPrice(levels.nearest_support), "text-emerald-300"],
      ["Résistance clé", fmtPrice(levels.nearest_resistance), "text-rose-300"],
    ];
    const grid = $("metrics-grid");
    if (grid) {
      grid.innerHTML = metrics.map((item) => metricCard(item[0], item[1], item[2], item[3])).join("");
    }

    const signals = (quant.surge_signals && quant.surge_signals.length ? quant.surge_signals : quant.signals) || [];
    const signalsRow = $("signals-row");
    if (signalsRow) {
      signalsRow.innerHTML = signals
        .slice(0, 10)
        .map(
          (signal) =>
            '<span class="rounded-lg border border-slate-700 bg-slate-950/40 px-2.5 py-1 text-[10px] text-slate-300">' +
            escapeHtml(signal) +
            "</span>"
        )
        .join("");
    }

    drawChart(quant.candles || []);
    resetAgents();
    const synthesis = $("synthesis-section");
    if (synthesis) synthesis.classList.add("hidden");
    state.synthesis = null;
    const gatesBadge = $("buy-gates-badge");
    if (gatesBadge) gatesBadge.textContent = "--";
    const gatesPanel = $("buy-gates-panel");
    if (gatesPanel) gatesPanel.textContent = "Le statut des gates déterministes apparaîtra après l'analyse.";

    const optionsSymbol = $("options-symbol");
    if (optionsSymbol && quant.market_type === "stock") optionsSymbol.value = quant.symbol || state.symbol;

    const latencyText = "Quant " + state.quantLatency + " ms";
    setLatency(latencyText);
    setStatus("Événement quant reçu", true);
  }

  function rsiTone(value) {
    const rsi = toNumber(value);
    if (rsi === null) return "text-slate-300";
    if (rsi >= 70) return "text-rose-300";
    if (rsi <= 30) return "text-emerald-300";
    return "text-slate-100";
  }

  // -------------------------------------------------------------------------
  // Agents
  // -------------------------------------------------------------------------
  function skeletonHtml() {
    return (
      '<div class="skeleton-line w-3/4"></div>' +
      '<div class="skeleton-line w-full"></div>' +
      '<div class="skeleton-line w-5/6"></div>' +
      '<div class="skeleton-line w-2/3"></div>'
    );
  }

  function resetAgents() {
    state.agents = {};
    ["tech", "macro", "sentiment"].forEach((key) => {
      const card = $("agent-" + key);
      const status = $("agent-" + key + "-status");
      const body = $("agent-" + key + "-body");
      if (card) {
        card.classList.add("is-loading");
        card.classList.remove("is-ready");
      }
      if (status) {
        status.textContent = "Analyse...";
        status.className =
          "rounded-md border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[9px] uppercase tracking-wide text-blue-300";
      }
      if (body) body.innerHTML = skeletonHtml();
    });
  }

  function markAgentsError(message) {
    ["tech", "macro", "sentiment"].forEach((key) => {
      const card = $("agent-" + key);
      const status = $("agent-" + key + "-status");
      const body = $("agent-" + key + "-body");
      if (card) card.classList.remove("is-loading");
      if (status) {
        status.textContent = "Erreur";
        status.className =
          "rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[9px] uppercase tracking-wide text-rose-300";
      }
      if (body) body.innerHTML = '<p class="text-[11px] text-rose-300/80">' + escapeHtml(message) + "</p>";
    });
  }

  function renderAgent(key, data) {
    state.agents[key] = data;
    const card = $("agent-" + key);
    const status = $("agent-" + key + "-status");
    const body = $("agent-" + key + "-body");
    if (!card || !status || !body) return;
    card.classList.remove("is-loading");
    card.classList.add("is-ready");

    const bias = data.bias || "neutral";
    status.textContent = data.cached ? "Cache" : "Terminé";
    status.className =
      "rounded-md border px-2 py-0.5 text-[9px] uppercase tracking-wide " + biasClasses(bias);

    const points = (data.key_points || [])
      .slice(0, 5)
      .map((point) => "<li>" + escapeHtml(point) + "</li>")
      .join("");
    const risks = (data.risks || [])
      .slice(0, 3)
      .map((risk) => "<li>" + escapeHtml(risk) + "</li>")
      .join("");

    body.innerHTML =
      '<div class="flex items-center justify-between gap-2">' +
      '<span class="rounded-md border px-2 py-0.5 text-[9px] uppercase tracking-wide ' + biasClasses(bias) + '">' + escapeHtml(bias) + "</span>" +
      '<span class="text-[10px] text-slate-400">Score ' + num(data.score, 0) + " · Conf. " + num(data.confidence, 0) + "%</span>" +
      "</div>" +
      '<p class="mt-2.5 text-[11px] leading-relaxed text-slate-300">' + escapeHtml(data.summary || "") + "</p>" +
      (points ? '<ul class="mt-2.5 list-disc space-y-1 pl-4 text-[10px] leading-relaxed text-slate-400">' + points + "</ul>" : "") +
      (risks
        ? '<div class="mt-2.5 text-[9px] uppercase tracking-wide text-rose-300/80">Risques</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[10px] leading-relaxed text-slate-500">' + risks + "</ul>"
        : "");
  }

  // -------------------------------------------------------------------------
  // Synthèse et ticket d'ordre
  // -------------------------------------------------------------------------
  function verdictClasses(verdict) {
    if (verdict === "BUY") return "border border-emerald-500/40 bg-emerald-500/20 text-emerald-200";
    if (verdict === "SELL") return "border border-rose-500/40 bg-rose-500/20 text-rose-200";
    if (verdict === "HOLD") return "border border-amber-500/40 bg-amber-500/20 text-amber-200";
    return "border border-slate-500/40 bg-slate-600/20 text-slate-200";
  }

  function renderBuyGates(synthesis, quant) {
    const badge = $("buy-gates-badge");
    const panel = $("buy-gates-panel");
    if (!panel) return;
    const gates = (synthesis && synthesis.buy_gates) || (quant && quant.buy_gates) || null;
    if (!gates) {
      if (badge) badge.textContent = "--";
      panel.innerHTML = '<p class="text-[11px] text-slate-500">Aucune évaluation de gates disponible.</p>';
      return;
    }
    const allowed = Boolean(gates.allowed);
    if (badge) {
      badge.textContent = allowed ? "BUY autorisé" : "BUY bloqué";
      badge.className = "rounded-md border px-2 py-0.5 text-[9px] uppercase tracking-wide " +
        (allowed ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" : "border-rose-500/40 bg-rose-500/10 text-rose-300");
    }
    const rows = (gates.gates || []).map((gate) => {
      const blocking = Boolean(gate.blocking);
      const passed = Boolean(gate.passed);
      const icon = passed ? "✓" : (blocking ? "✕" : "!");
      const cls = passed ? "text-emerald-300" : (blocking ? "text-rose-300" : "text-amber-300");
      return '<div class="flex items-start gap-2"><span class="' + cls + ' font-semibold">' + icon + "</span>" +
        '<div><span class="font-medium text-slate-300">' + escapeHtml(gate.name || "gate") + "</span>" +
        (blocking ? ' <span class="text-[9px] uppercase text-slate-500">bloquant</span>' : "") +
        '<div class="text-[10px] text-slate-500">' + escapeHtml(gate.detail || "") + "</div></div></div>";
    }).join("");
    const score = toNumber(gates.score);
    panel.innerHTML =
      '<div class="mb-1 flex items-center justify-between text-[10px] text-slate-500">' +
      "<span>Score gates : " + (score === null ? "n/a" : num(score, 0) + "/100") + "</span>" +
      "<span>" + (gates.blockers || []).length + " bloquant(s) · " + (gates.warnings || []).length + " avertissement(s)</span></div>" +
      (rows || '<p class="text-[11px] text-slate-500">Aucune gate évaluée.</p>');
  }

  function renderMemory(synthesis) {
    const container = $("memory-panel");
    const quant = state.quant || {};
    renderBuyGates(synthesis, quant);
    if (!container) return;
    const rules = (quant.memory_rules || []).concat(synthesis.memory_rules || []);
    const pitfalls = (quant.memory_pitfalls || []).concat(synthesis.memory_pitfalls || []);
    const summary = synthesis.memory_summary || quant.memory_summary || "";

    if (!summary && !rules.length && !pitfalls.length) {
      container.innerHTML = '<p class="text-[11px] text-slate-500">Aucune mémoire consolidée pour ce ticker. Le Risk Manager applique les règles générales de risque.</p>';
      return;
    }

    const uniqueRules = Array.from(new Set(rules)).slice(0, 3);
    const uniquePitfalls = Array.from(new Set(pitfalls)).slice(0, 3);
    container.innerHTML =
      (summary ? '<p class="text-[11px] leading-relaxed text-slate-300">' + escapeHtml(summary) + "</p>" : "") +
      (uniqueRules.length
        ? '<div class="mt-2 text-[9px] uppercase tracking-wide text-teal-300/80">Règles apprises</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[11px] text-slate-400">' +
          uniqueRules.map((rule) => "<li>" + escapeHtml(rule) + "</li>").join("") +
          "</ul>"
        : "") +
      (uniquePitfalls.length
        ? '<div class="mt-2 text-[9px] uppercase tracking-wide text-rose-300/80">Pièges à éviter</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[11px] text-slate-500">' +
          uniquePitfalls.map((pitfall) => "<li>" + escapeHtml(pitfall) + "</li>").join("") +
          "</ul>"
        : "");
  }

  function renderSynthesis(synthesis) {
    state.synthesis = synthesis;
    const section = $("synthesis-section");
    if (section) section.classList.remove("hidden");

    const verdict = synthesis.verdict || "WATCH";
    const badge = $("verdict-badge");
    if (badge) {
      badge.textContent = verdict;
      badge.className = "rounded-lg px-3 py-1.5 text-sm font-bold " + verdictClasses(verdict);
    }
    const confidence = Math.max(0, Math.min(100, toNumber(synthesis.confidence) || 0));
    const confidenceValue = $("confidence-value");
    if (confidenceValue) confidenceValue.textContent = "Confiance " + num(confidence, 0) + "%";
    const confidenceBar = $("confidence-bar");
    if (confidenceBar) confidenceBar.style.width = confidence + "%";
    const source = $("synthesis-source");
    if (source) source.textContent = "source " + (synthesis.source || "n/a");

    const consensus = synthesis.agent_consensus || {};
    const consensusElement = $("agent-consensus");
    if (consensusElement) {
      consensusElement.textContent =
        "Tech " + num(consensus.technical_score, 0) +
        " · Macro " + num(consensus.macro_score, 0) +
        " · News " + num(consensus.sentiment_score, 0) +
        " · Pondéré " + num(consensus.weighted_score, 1);
    }

    const bullList = $("bull-list");
    if (bullList) {
      const items = (synthesis.bull_case || []).slice(0, 6);
      bullList.innerHTML = (items.length ? items : ["Aucun argument haussier majeur identifié."])
        .map((item) => "<li>" + escapeHtml(item) + "</li>")
        .join("");
    }
    const bearList = $("bear-list");
    if (bearList) {
      const items = (synthesis.bear_case || []).slice(0, 6);
      bearList.innerHTML = (items.length ? items : ["Aucun argument baissier majeur identifié."])
        .map((item) => "<li>" + escapeHtml(item) + "</li>")
        .join("");
    }

    const plan = synthesis.trade_plan || {};
    const ticket = {
      "ticket-entry": fmtPrice(plan.entry_price),
      "ticket-stop": fmtPrice(plan.stop_loss),
      "ticket-tp1": fmtPrice(plan.take_profit_1),
      "ticket-tp2": fmtPrice(plan.take_profit_2),
      "ticket-size": num(plan.position_size_pct, 2) + " %",
      "ticket-rr": num(plan.risk_reward_ratio, 2),
    };
    Object.entries(ticket).forEach(([id, value]) => {
      const element = $(id);
      if (element) element.textContent = value;
    });

    const rationale = $("synthesis-rationale");
    if (rationale) rationale.textContent = synthesis.rationale || "";

    renderMemory(synthesis);
    drawTradePlanLines(plan);

    const actionable = verdict === "BUY" || verdict === "SELL";
    const button = $("open-trade-btn");
    if (button) {
      button.disabled = !actionable;
      button.textContent = actionable
        ? "Ouvrir " + (verdict === "BUY" ? "Long" : "Short") + " en Paper Trading"
        : "Verdict non actionnable";
    }
    const hint = $("ticket-hint");
    if (hint) {
      hint.textContent = actionable
        ? "Le ticket est pré-rempli par le Risk Manager (ATR + profil de risque)."
        : "Attendez un verdict BUY ou SELL pour ouvrir le ticket.";
    }

    if (synthesis.elapsed_ms) {
      setLatency("Quant " + (state.quantLatency || 0) + " ms · Total " + num(synthesis.elapsed_ms, 0) + " ms");
    }
  }

  async function openTrade() {
    const synthesis = state.synthesis;
    if (!synthesis) return;
    if (!["BUY", "SELL"].includes(synthesis.verdict)) {
      toast("Le verdict actuel n'est pas actionnable.", true);
      return;
    }
    const button = $("open-trade-btn");
    if (button) {
      button.disabled = true;
      button.textContent = "Ouverture en cours...";
    }
    try {
      const result = await api("/api/portfolio/order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: synthesis.symbol || state.symbol,
          market_type: synthesis.market_type || (state.quant || {}).market_type,
          risk_profile: synthesis.risk_profile || state.risk,
          reasoning_summary: synthesis.rationale || "",
          trade_plan: synthesis.trade_plan || {},
          verdict: synthesis.verdict,
        }),
      });
      toast((result.message || "Position ouverte.") + (result.quantity_adjusted ? " Quantité ajustée au cash." : ""));
      await loadPortfolio();
      activateTab("portfolio");
    } catch (error) {
      toast("Ouverture impossible : " + error.message, true);
    } finally {
      if (button && state.synthesis) {
        button.disabled = !["BUY", "SELL"].includes(state.synthesis.verdict);
        button.textContent =
          state.synthesis.verdict === "BUY"
            ? "Ouvrir Long en Paper Trading"
            : state.synthesis.verdict === "SELL"
              ? "Ouvrir Short en Paper Trading"
              : "Verdict non actionnable";
      }
    }
  }

  // -------------------------------------------------------------------------
  // Flux SSE
  // -------------------------------------------------------------------------
  function closeStream() {
    if (state.source) {
      state.source.close();
      state.source = null;
    }
  }

  function setStreaming(streaming) {
    state.streaming = streaming;
    const analyze = $("analyze-btn");
    const force = $("force-refresh-btn");
    if (analyze) {
      analyze.disabled = streaming;
      analyze.textContent = streaming ? "Analyse en cours..." : "Analyser (Live Stream)";
      analyze.classList.toggle("opacity-60", streaming);
    }
    if (force) {
      force.disabled = streaming;
      force.classList.toggle("opacity-60", streaming);
    }
  }

  function resetAnalysisUi() {
    state.quant = null;
    state.agents = {};
    state.synthesis = null;
    state.quantLatency = null;
    const title = $("symbol-title");
    if (title) title.textContent = state.symbol;
    const subtitle = $("symbol-subtitle");
    if (subtitle) subtitle.textContent = "Connexion au flux SSE...";
    const price = $("current-price");
    if (price) price.textContent = "--";
    const change = $("price-change");
    if (change) {
      change.textContent = "--";
      change.className = "text-xs font-medium text-slate-500";
    }
    const metrics = $("metrics-grid");
    if (metrics) metrics.innerHTML = "";
    const signals = $("signals-row");
    if (signals) signals.innerHTML = "";
    const synthesis = $("synthesis-section");
    if (synthesis) synthesis.classList.add("hidden");
    clearChart();
    resetAgents();
    setLatency("Latence -- ms");
    setStatus("Analyse en cours", true);
  }

  function runAnalysis(force = false) {
    const input = $("ticker-input");
    const symbol = (input ? input.value : "").trim().toUpperCase();
    if (!symbol) {
      toast("Entrez un symbole à analyser.", true);
      return;
    }
    state.symbol = symbol;
    state.timeframe = $("timeframe-select") ? $("timeframe-select").value : "1d";
    state.risk = $("risk-select") ? $("risk-select").value : "moderate";
    state.mode = $("mode-select") ? $("mode-select").value : "auto";
    closeStream();
    resetAnalysisUi();
    setStreaming(true);
    state.latencyStart = performance.now();

    const params = new URLSearchParams({
      timeframe: state.timeframe,
      risk_profile: state.risk,
      mode: state.mode,
      force_refresh: force ? "true" : "false",
    });
    const source = new EventSource(
      "/api/analyze/" + encodeURIComponent(symbol) + "/stream?" + params.toString()
    );
    state.source = source;

    const handlers = {
      quant: (payload) => renderQuant(payload),
      agent_tech: (payload) => renderAgent("tech", payload),
      agent_macro: (payload) => renderAgent("macro", payload),
      agent_sentiment: (payload) => renderAgent("sentiment", payload),
      synthesis: (payload) => {
        renderSynthesis(payload);
        setStreaming(false);
        setStatus("Analyse terminée", false);
      },
    };

    Object.entries(handlers).forEach(([eventName, handler]) => {
      source.addEventListener(eventName, (event) => {
        let payload = {};
        try {
          payload = JSON.parse(event.data || "{}");
        } catch (error) {
          toast("Réponse SSE invalide.", true);
          return;
        }
        handler(payload);
        if (eventName === "synthesis") closeStream();
      });
    });

    source.addEventListener("error", (event) => {
      if (event && event.data) {
        let payload = {};
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          payload = { message: "Erreur de streaming." };
        }
        toast(payload.message || "Erreur d'analyse.", true);
        markAgentsError(payload.message || "Analyse indisponible.");
        setStatus("Erreur d'analyse", false);
        setStreaming(false);
        closeStream();
      }
    });

    source.onerror = (event) => {
      if (event && event.data) return;
      if (state.source !== source) return;
      if (source.readyState === EventSource.CLOSED) {
        setStreaming(false);
        setStatus("Flux interrompu", false);
      } else {
        setStatus("Reconnexion SSE...", true);
      }
    };
  }

  // -------------------------------------------------------------------------
  // Portefeuille
  // -------------------------------------------------------------------------
  function renderPortfolio(data) {
    const summary = data.summary || {};
    const cards = [
      ["Cash disponible", fmtMoney(summary.cash_available_usd), "text-slate-100"],
      ["Equity totale", fmtMoney(summary.equity_usd), "text-white"],
      ["Win Rate", num(summary.win_rate_pct, 1) + " %", "text-slate-100"],
      ["PnL réalisé", fmtMoney(summary.realized_pnl_usd), tone(summary.realized_pnl_usd)],
      ["PnL latent", fmtMoney(summary.unrealized_pnl_usd), tone(summary.unrealized_pnl_usd)],
      ["Profit Factor", summary.profit_factor == null ? "n/a" : num(summary.profit_factor, 2), "text-slate-100"],
      ["Max Drawdown", num(summary.max_drawdown_pct, 2) + " %", "text-rose-300"],
    ];
    const metrics = $("portfolio-metrics");
    if (metrics) {
      metrics.innerHTML = cards.map((item) => metricCard(item[0], item[1], item[2])).join("");
    }

    const positions = data.positions || [];
    const positionsBody = $("positions-body");
    const positionsCount = $("positions-count");
    if (positionsCount) positionsCount.textContent = String(positions.length);
    if (positionsBody) {
      if (!positions.length) {
        positionsBody.innerHTML =
          '<tr><td colspan="8" class="px-3 py-6 text-center text-slate-500">Aucune position ouverte.</td></tr>';
      } else {
        positionsBody.innerHTML = positions
          .map((position) => {
            const pnlTone = tone(position.unrealized_pnl_usd);
            return (
              '<tr class="border-t border-slate-800/60">' +
              '<td class="px-3 py-2.5"><div class="font-semibold text-slate-100">' + escapeHtml(position.symbol) + "</div>" +
              '<div class="text-[9px] text-slate-500">' + escapeHtml(position.market_type || "") + (position.price_stale ? " · prix indicatif" : "") + "</div></td>" +
              '<td class="px-2 py-2.5"><span class="rounded border px-1.5 py-0.5 text-[9px] uppercase ' + sideClasses(position.side) + '">' + escapeHtml(position.side) + "</span></td>" +
              '<td class="px-2 py-2.5 metric-value">' + num(position.quantity, 6) + "</td>" +
              '<td class="px-2 py-2.5 metric-value">' + fmtPrice(position.entry_price) + "</td>" +
              '<td class="px-2 py-2.5 metric-value">' + fmtPrice(position.current_price) + "</td>" +
              '<td class="px-2 py-2.5 metric-value ' + pnlTone + '">' + fmtSigned(position.unrealized_pnl_usd, 2) + " $<br><span class=\"text-[9px]\">" + fmtSigned(position.unrealized_pnl_pct, 2) + " %</span></td>" +
              '<td class="px-2 py-2.5 text-[10px] text-slate-400">SL ' + fmtPrice(position.stop_loss) + '<br>TP ' + fmtPrice(position.take_profit) + "</td>" +
              '<td class="px-3 py-2.5 text-right"><button type="button" data-close-id="' + Number(position.id) + '" class="rounded-lg border border-slate-700 px-2.5 py-1.5 text-[10px] text-slate-400 transition hover:border-rose-500/60 hover:text-rose-300">Fermer</button></td>' +
              "</tr>"
            );
          })
          .join("");
      }
    }

    const history = data.history || [];
    const historyCount = $("history-count");
    if (historyCount) historyCount.textContent = String(history.length);
    const historyBody = $("history-body");
    if (historyBody) {
      if (!history.length) {
        historyBody.innerHTML =
          '<tr><td colspan="8" class="px-3 py-6 text-center text-slate-500">Aucun trade clôturé.</td></tr>';
      } else {
        historyBody.innerHTML = history
          .slice(0, 40)
          .map((trade) => {
            const reasoning = trade.reasoning_summary || trade.lesson_learned || "—";
            return (
              '<tr class="border-t border-slate-800/60">' +
              '<td class="whitespace-nowrap px-3 py-2.5 text-slate-500">' + escapeHtml((trade.closed_at || "").slice(0, 16).replace("T", " ")) + "</td>" +
              '<td class="px-2 py-2.5 font-medium text-slate-200">' + escapeHtml(trade.symbol) + "</td>" +
              '<td class="px-2 py-2.5"><span class="rounded border px-1.5 py-0.5 text-[9px] uppercase ' + sideClasses(trade.side) + '">' + escapeHtml(trade.side) + "</span></td>" +
              '<td class="px-2 py-2.5 metric-value">' + fmtPrice(trade.entry_price) + "</td>" +
              '<td class="px-2 py-2.5 metric-value">' + fmtPrice(trade.exit_price) + "</td>" +
              '<td class="px-2 py-2.5 metric-value ' + tone(trade.pnl_usd) + '">' + fmtSigned(trade.pnl_usd, 2) + " $<br><span class=\"text-[9px]\">" + fmtSigned(trade.pnl_pct, 2) + " %</span></td>" +
              '<td class="px-2 py-2.5 text-[10px] text-slate-400">' + escapeHtml(trade.exit_reason || "—") + "</td>" +
              '<td class="max-w-[220px] px-3 py-2.5 text-[10px] leading-relaxed text-slate-400"><span title="' + escapeHtml(reasoning) + '">' + escapeHtml(String(reasoning).slice(0, 120)) + (String(reasoning).length > 120 ? "…" : "") + "</span></td>" +
              "</tr>"
            );
          })
          .join("");
      }
    }

    const memory = data.memory || [];
    const memoryContainer = $("portfolio-memory");
    if (memoryContainer) {
      if (!memory.length) {
        memoryContainer.innerHTML = '<div class="text-xs text-slate-500">Aucune mémoire consolidée.</div>';
      } else {
        memoryContainer.innerHTML = memory
          .slice(0, 12)
          .map(
            (item) =>
              '<div class="soft rounded-xl p-3"><div class="flex items-center justify-between gap-2">' +
              '<span class="text-xs font-semibold text-slate-100">' + escapeHtml(item.symbol) + "</span>" +
              '<span class="text-[9px] uppercase tracking-wide text-slate-500">' + escapeHtml(item.market_type) + " · WR " + num(item.win_rate, 1) + "% · " + escapeHtml(item.total_trades) + " trades</span></div>" +
              '<pre class="mt-2 whitespace-pre-wrap text-[10px] leading-relaxed text-slate-400">' + escapeHtml(item.compressed_rules || "") + "</pre></div>"
          )
          .join("");
      }
    }

    const meta = $("tab-meta");
    if (meta && data.updated_at) {
      meta.textContent = "Portefeuille actualisé à " + data.updated_at.slice(11, 19) + " UTC";
    }
  }

  async function loadPortfolio() {
    try {
      const data = await api("/api/portfolio");
      renderPortfolio(data);
    } catch (error) {
      const body = $("positions-body");
      if (body) {
        body.innerHTML =
          '<tr><td colspan="8" class="px-3 py-6 text-center text-rose-300">' + escapeHtml(error.message) + "</td></tr>";
      }
    }
  }

  async function closePosition(positionId) {
    if (!window.confirm("Clôturer cette position au prix du marché ?")) return;
    try {
      await api("/api/portfolio/close/" + positionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exit_reason: "manual" }),
      });
      toast("Position clôturée.");
      await loadPortfolio();
    } catch (error) {
      toast("Clôture impossible : " + error.message, true);
    }
  }

  // -------------------------------------------------------------------------
  // Screener
  // -------------------------------------------------------------------------
  function renderScreener() {
    const rows = state.screenerRows.slice();
    const { key, dir } = state.screenerSort;
    rows.sort((a, b) => {
      const left = a[key];
      const right = b[key];
      if (typeof left === "number" || typeof right === "number") {
        return ((toNumber(left) || 0) - (toNumber(right) || 0)) * (dir === "asc" ? 1 : -1);
      }
      return String(left || "").localeCompare(String(right || ""), "fr") * (dir === "asc" ? 1 : -1);
    });

    const body = $("screener-body");
    if (!body) return;
    if (!rows.length) {
      body.innerHTML =
        '<tr><td colspan="9" class="px-3 py-8 text-center text-slate-500">Aucune opportunité détectée.</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((row) => {
        const score = toNumber(row.surge_score) || 0;
        const scoreTone = score >= 70 ? "text-emerald-300" : score >= 50 ? "text-amber-200" : "text-slate-400";
        return (
          '<tr class="row-click border-t border-slate-800/60" data-symbol="' + escapeHtml(row.symbol) + '">' +
          '<td class="px-3 py-2.5 font-semibold text-slate-100">' + escapeHtml(row.symbol) + '<div class="text-[9px] font-normal text-slate-500">' + escapeHtml(row.grade || "") + "</div></td>" +
          '<td class="px-2 py-2.5 text-[10px] uppercase tracking-wide text-slate-400">' + escapeHtml(row.market_type) + "</td>" +
          '<td class="px-2 py-2.5 metric-value">' + fmtPrice(row.price) + "</td>" +
          '<td class="px-2 py-2.5"><div class="flex items-center gap-2"><span class="metric-value w-10 ' + scoreTone + '">' + num(score, 1) + '</span><span class="h-1 w-14 overflow-hidden rounded-full bg-slate-800"><span class="block h-full rounded-full bg-gradient-to-r from-teal-500 to-blue-500" style="width:' + Math.max(2, Math.min(100, score)) + '%"></span></span></div></td>' +
          '<td class="px-2 py-2.5"><span class="rounded border border-slate-700 px-1.5 py-0.5 text-[9px] text-slate-300">' + escapeHtml(row.grade || "-") + "</span></td>" +
          '<td class="px-2 py-2.5 metric-value ' + rsiTone(row.rsi14) + '">' + num(row.rsi14, 1) + "</td>" +
          '<td class="px-2 py-2.5 metric-value">' + num(row.volume_ratio, 2) + "x</td>" +
          '<td class="px-2 py-2.5 metric-value ' + tone(row.momentum_5d_pct) + '">' + fmtPct(row.momentum_5d_pct, 2) + "</td>" +
          '<td class="px-3 py-2.5 text-[10px] capitalize text-slate-300">' + escapeHtml(row.trend || "n/a") + "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  async function loadScreener() {
    const button = $("run-screener-btn");
    const body = $("screener-body");
    if (button) {
      button.disabled = true;
      button.textContent = "Scan en cours...";
      button.classList.add("opacity-60");
    }
    if (body) {
      body.innerHTML =
        '<tr><td colspan="9" class="px-3 py-8 text-center text-slate-500">Scan mathématique Crypto + US en cours...</td></tr>';
    }
    try {
      const data = await api("/api/screener");
      state.screenerRows = data.results || [];
      const updated = $("screener-updated");
      if (updated) {
        updated.textContent = "Actualisé à " + new Date().toLocaleTimeString("fr-BE");
      }
      renderScreener();
    } catch (error) {
      if (body) {
        body.innerHTML =
          '<tr><td colspan="9" class="px-3 py-8 text-center text-rose-300">' + escapeHtml(error.message) + "</td></tr>";
      }
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Lancer le screener";
        button.classList.remove("opacity-60");
      }
    }
  }

  // -------------------------------------------------------------------------
  // Laboratoire options
  // -------------------------------------------------------------------------
  function grekCard(label, value, hint = "") {
    return metricCard(label, value, "text-slate-100", hint);
  }

  function renderOptionsResult(data) {
    const meta = $("options-meta");
    if (meta) {
      meta.textContent =
        "Sous-jacent " + fmtPrice(data.spot) +
        " · expiration " + (data.expiration || "n/a") +
        " · " + (data.calls_count || 0) + " calls / " + (data.puts_count || 0) + " puts";
    }

    const setup = data.setup || {};
    const summaryBody = $("options-summary-body");
    if (summaryBody) {
      if (setup.error) {
        summaryBody.innerHTML = '<p class="text-rose-300">' + escapeHtml(setup.error) + "</p>";
      } else {
        summaryBody.innerHTML =
          '<div class="mb-3 flex flex-wrap items-center gap-2">' +
          '<span class="rounded-md border border-teal-500/30 bg-teal-500/10 px-2 py-1 text-[10px] uppercase tracking-wide text-teal-200">' + escapeHtml(setup.option_type || "call") + " " + escapeHtml(setup.strike) + "</span>" +
          '<span class="text-[10px] text-slate-400">' + escapeHtml(setup.contract_symbol || "") + "</span>" +
          "</div>" +
          '<div class="grid grid-cols-2 gap-2.5 sm:grid-cols-3">' +
          metricCard("Prime", fmtPrice(setup.premium)) +
          metricCard("Coût / contrat", fmtMoney(setup.cost_usd)) +
          metricCard("Break-even", fmtPrice(setup.breakeven)) +
          metricCard("IV", num(setup.iv_pct, 2) + " %") +
          metricCard("Perte max", fmtMoney(setup.max_loss_usd)) +
          metricCard("Profit cible", fmtMoney(setup.target_profit_usd), "text-emerald-300") +
          metricCard("Delta", num(setup.delta, 3)) +
          metricCard("R:R", num(setup.risk_reward, 2)) +
          metricCard("Cible", fmtPrice(setup.target_price)) +
          "</div>" +
          '<p class="mt-3 text-[11px] leading-relaxed text-slate-400">' + escapeHtml(setup.rationale || "") + "</p>";
      }
    }

    const greeksBody = $("options-greeks-body");
    const custom = data.custom_greeks;
    if (greeksBody) {
      if (!custom) {
        greeksBody.innerHTML = '<p class="text-[11px] text-slate-500">Renseignez un strike et un DTE pour calculer les Grecques Black-Scholes personnalisées.</p>';
      } else if (custom.error) {
        greeksBody.innerHTML = '<p class="text-[11px] text-rose-300">' + escapeHtml(custom.error) + "</p>";
      } else {
        const greeks = custom.greeks || {};
        greeksBody.innerHTML =
          '<div class="grid grid-cols-2 gap-2.5 sm:grid-cols-3">' +
          grekCard("Delta", num(greeks.delta, 4)) +
          grekCard("Gamma", num(greeks.gamma, 6)) +
          grekCard("Theta (/jour)", num(greeks.theta, 4)) +
          grekCard("Vega (1%)", num(greeks.vega, 4)) +
          grekCard("Rho", num(greeks.rho, 4)) +
          grekCard("Prix théorique", fmtPrice(custom.theoretical_price), "text-slate-100") +
          grekCard("Valeur intrinsèque", fmtPrice(custom.intrinsic_value)) +
          grekCard("Valeur temps", fmtPrice(custom.time_value)) +
          grekCard("IV implicite", num(custom.iv_pct, 2) + " %") +
          "</div>";
      }
    }
  }

  async function calculateOptions() {
    const symbol = (($("options-symbol") || {}).value || "").trim().toUpperCase();
    if (!symbol) {
      toast("Entrez un ticker US.", true);
      return;
    }
    const button = $("options-calc-btn");
    const summaryBody = $("options-summary-body");
    if (button) {
      button.disabled = true;
      button.textContent = "Calcul en cours...";
    }
    if (summaryBody) {
      summaryBody.innerHTML = '<p class="text-[11px] text-slate-500">Récupération de la chaîne d\'options et calcul...</p>';
    }
    try {
      const params = new URLSearchParams({
        outlook: ($("options-outlook") || {}).value || "bullish",
        risk_profile: ($("options-risk") || {}).value || "moderate",
        option_type: ($("options-type") || {}).value || "call",
      });
      const strike = toNumber(($("options-strike") || {}).value);
      const dte = toNumber(($("options-dte") || {}).value);
      const iv = toNumber(($("options-iv") || {}).value);
      const target = toNumber(($("options-target") || {}).value);
      if (strike !== null && strike > 0) params.set("strike", String(strike));
      if (dte !== null && dte > 0) params.set("dte", String(dte));
      if (iv !== null && iv > 0) params.set("volatility", String(iv / 100));
      if (target !== null && target > 0) params.set("target_price", String(target));

      const data = await api("/api/options/preview/" + encodeURIComponent(symbol) + "?" + params.toString());
      renderOptionsResult(data);
    } catch (error) {
      if (summaryBody) {
        summaryBody.innerHTML = '<p class="text-[11px] text-rose-300">' + escapeHtml(error.message) + "</p>";
      }
      toast("Options indisponibles : " + error.message, true);
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Calculer les Grecques";
      }
    }
  }

  // -------------------------------------------------------------------------
  // Onglets
  // -------------------------------------------------------------------------
  function activateTab(name) {
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.setAttribute("aria-selected", button.dataset.tab === name ? "true" : "false");
    });
    ["portfolio", "screener", "options", "memory"].forEach((tab) => {
      const panel = $("tab-" + tab);
      if (panel) panel.classList.toggle("hidden", tab !== name);
    });
    if (name === "portfolio") loadPortfolio();
    if (name === "screener" && !state.screenerRows.length) loadScreener();
    if (name === "memory") loadMemory();
  }

  // -------------------------------------------------------------------------
  // Initialisation et événements
  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------
  // Mémoire auto-apprenante (onglet dédié)
  // -------------------------------------------------------------------------
  function renderMemoryMetrics(items) {
    const container = $("memory-tab-metrics");
    if (!container) return;
    const total = items.length;
    const trades = items.reduce((acc, item) => acc + (toNumber(item.total_trades) || 0), 0);
    const weightedWin = items.reduce(
      (acc, item) => acc + (toNumber(item.win_rate) || 0) * (toNumber(item.total_trades) || 0),
      0
    );
    const avgWin = trades > 0 ? weightedWin / trades : null;
    const cards = [
      ["Symboles suivis", String(total)],
      ["Trades mémorisés", String(trades)],
      ["Win rate pondéré", avgWin === null ? "n/a" : num(avgWin, 1) + " %"],
      ["Dernière mise à jour", items.length ? String(items[0].updated_at || "n/a").slice(0, 16).replace("T", " ") : "n/a"],
    ];
    container.innerHTML = cards.map(([label, value]) =>
      '<div class="soft rounded-xl border border-slate-800 p-3"><div class="text-[9px] uppercase tracking-wide text-slate-500">' +
      escapeHtml(label) + '</div><div class="metric-value mt-1 text-sm font-semibold text-slate-100">' +
      escapeHtml(value) + "</div></div>"
    ).join("");
  }

  function renderMemoryRows(items) {
    const body = $("memory-tab-body");
    if (!body) return;
    if (!items.length) {
      body.innerHTML = '<tr><td colspan="6" class="px-3 py-6 text-center text-slate-500">Aucune mémoire consolidée pour le moment.</td></tr>';
      return;
    }
    body.innerHTML = items.map((item) => {
      const win = toNumber(item.win_rate);
      return "<tr class=\"border-t border-slate-800/60 hover:bg-slate-900/30\">" +
        '<td class="px-3 py-2.5 font-semibold text-slate-100">' + escapeHtml(item.symbol) + "</td>" +
        '<td class="px-2 py-2.5 text-slate-400">' + escapeHtml((item.market_type || "").toUpperCase()) + "</td>" +
        '<td class="px-2 py-2.5 ' + (win !== null && win >= 50 ? "text-emerald-300" : "text-rose-300") + '">' + (win === null ? "n/a" : num(win, 1) + " %") + "</td>" +
        '<td class="px-2 py-2.5 text-slate-300">' + escapeHtml(item.total_trades ?? 0) + "</td>" +
        '<td class="px-2 py-2.5 text-slate-500">' + escapeHtml(String(item.updated_at || "n/a").slice(0, 16).replace("T", " ")) + "</td>" +
        '<td class="px-3 py-2.5 text-slate-400"><div class="line-clamp-3 whitespace-pre-line">' + escapeHtml(item.compressed_rules || "—") + "</div></td>" +
        "</tr>";
    }).join("");
  }

  async function loadMemory() {
    try {
      const response = await fetch("/api/memory", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const data = await response.json();
      const items = Array.isArray(data.items) ? data.items : [];
      renderMemoryMetrics(items);
      renderMemoryRows(items);
    } catch (error) {
      const body = $("memory-tab-body");
      if (body) body.innerHTML = '<tr><td colspan="6" class="px-3 py-6 text-center text-rose-300">' + escapeHtml(error.message) + "</td></tr>";
      toast("Mémoire indisponible : " + error.message, true);
    }
  }

  async function loadMemorySymbol() {
    const input = $("memory-symbol-input");
    const container = $("memory-tab-current");
    if (!container) return;
    const symbol = (input && input.value.trim()) || state.symbol || config.defaultSymbol;
    if (!symbol) return;
    container.innerHTML = '<span class="text-slate-500">Chargement de ' + escapeHtml(symbol) + "...</span>";
    try {
      const params = new URLSearchParams({ symbol: symbol, risk_profile: state.risk || "moderate" });
      const response = await fetch("/api/memory?" + params.toString(), { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const data = await response.json();
      const context = data.context || {};
      const performance = context.performance || {};
      const rules = context.rules || [];
      const pitfalls = context.pitfalls || [];
      const lessons = context.recent_lessons || [];
      container.innerHTML =
        '<div class="mb-2 flex flex-wrap items-center gap-2"><span class="font-semibold text-slate-100">' + escapeHtml((context.symbol || symbol).toUpperCase()) + "</span>" +
        '<span class="rounded-md border border-slate-700 px-2 py-0.5 text-[9px] uppercase text-slate-400">' + escapeHtml(context.market_type || "") + "</span>" +
        '<span class="text-[10px] text-slate-500">Win rate ' + num(performance.win_rate_pct, 1) + " % · " + escapeHtml(performance.total_trades ?? 0) + " trades</span></div>" +
        '<p class="text-[11px] leading-relaxed text-slate-300">' + escapeHtml(context.summary || "Aucun résumé.") + "</p>" +
        (rules.length ? '<div class="mt-2 text-[9px] uppercase tracking-wide text-teal-300/80">Règles apprises</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[11px] text-slate-400">' + rules.map((rule) => "<li>" + escapeHtml(rule) + "</li>").join("") + "</ul>" : "") +
        (lessons.length ? '<div class="mt-2 text-[9px] uppercase tracking-wide text-amber-300/80">Dernières leçons</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[11px] text-slate-400">' + lessons.map((lesson) => "<li>" + escapeHtml(lesson) + "</li>").join("") + "</ul>" : "") +
        (pitfalls.length ? '<div class="mt-2 text-[9px] uppercase tracking-wide text-rose-300/80">Pièges à éviter</div><ul class="mt-1 list-disc space-y-1 pl-4 text-[11px] text-slate-500">' + pitfalls.map((pitfall) => "<li>" + escapeHtml(pitfall) + "</li>").join("") + "</ul>" : "");
    } catch (error) {
      container.innerHTML = '<span class="text-rose-300">' + escapeHtml(error.message) + "</span>";
    }
  }

  function renderQuickTickers() {
    const container = $("quick-tickers");
    if (!container) return;
    const symbols = Array.from(new Set(config.watchlist || [])).slice(0, 10);
    container.innerHTML = symbols
      .map(
        (symbol) =>
          '<button type="button" data-quick-ticker="' + escapeHtml(symbol) + '" class="rounded-lg border border-slate-700/80 bg-slate-900/30 px-2.5 py-1 text-[10px] font-medium text-slate-400 transition hover:border-teal-500/60 hover:text-teal-200">' + escapeHtml(symbol) + "</button>"
      )
      .join("");
    container.querySelectorAll("[data-quick-ticker]").forEach((button) => {
      button.addEventListener("click", () => {
        const ticker = button.dataset.quickTicker;
        const input = $("ticker-input");
        if (input) input.value = ticker;
        runAnalysis(false);
      });
    });
  }

  function bindEvents() {
    const analyze = $("analyze-btn");
    if (analyze) analyze.addEventListener("click", () => runAnalysis(false));
    const force = $("force-refresh-btn");
    if (force) force.addEventListener("click", () => runAnalysis(true));
    const ticker = $("ticker-input");
    if (ticker) {
      ticker.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          runAnalysis(false);
        }
      });
    }

    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => activateTab(button.dataset.tab));
    });

    const refresh = $("refresh-portfolio-btn");
    if (refresh) refresh.addEventListener("click", loadPortfolio);
    const refreshMemory = $("refresh-memory-btn");
    if (refreshMemory) refreshMemory.addEventListener("click", loadMemory);
    const loadMemoryBtn = $("memory-symbol-load-btn");
    if (loadMemoryBtn) loadMemoryBtn.addEventListener("click", loadMemorySymbol);
    const memoryInput = $("memory-symbol-input");
    if (memoryInput) {
      memoryInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          loadMemorySymbol();
        }
      });
    }
    const runScreener = $("run-screener-btn");
    if (runScreener) runScreener.addEventListener("click", loadScreener);
    const calc = $("options-calc-btn");
    if (calc) calc.addEventListener("click", calculateOptions);
    const openTradeButton = $("open-trade-btn");
    if (openTradeButton) openTradeButton.addEventListener("click", openTrade);

    const positionsBody = $("positions-body");
    if (positionsBody) {
      positionsBody.addEventListener("click", (event) => {
        const button = event.target.closest("[data-close-id]");
        if (button) closePosition(button.dataset.closeId);
      });
    }

    const screenerBody = $("screener-body");
    if (screenerBody) {
      screenerBody.addEventListener("click", (event) => {
        const row = event.target.closest("[data-symbol]");
        if (!row) return;
        const input = $("ticker-input");
        if (input) input.value = row.dataset.symbol;
        runAnalysis(false);
      });
    }

    document.querySelectorAll(".sortable[data-sort]").forEach((header) => {
      header.addEventListener("click", () => {
        const key = header.dataset.sort;
        if (state.screenerSort.key === key) {
          state.screenerSort.dir = state.screenerSort.dir === "asc" ? "desc" : "asc";
        } else {
          state.screenerSort = { key, dir: "desc" };
        }
        renderScreener();
      });
    });
  }

  function init() {
    const riskSelect = $("risk-select");
    if (riskSelect) riskSelect.value = state.risk;
    const timeframeSelect = $("timeframe-select");
    if (timeframeSelect) timeframeSelect.value = state.timeframe;
    const modeSelect = $("mode-select");
    if (modeSelect) modeSelect.value = state.mode;
    const memoryInput = $("memory-symbol-input");
    if (memoryInput) memoryInput.value = state.symbol;
    const optionsSymbol = $("options-symbol");
    if (optionsSymbol && (!optionsSymbol.value || optionsSymbol.value === config.defaultSymbol)) {
      const stocks = config.stocks || [];
      const defaultStock = stocks.includes(optionsSymbol.value) ? optionsSymbol.value : stocks[0];
      if (defaultStock) optionsSymbol.value = defaultStock;
    }
    renderQuickTickers();
    bindEvents();
    initChart();
    activateTab("portfolio");
    setInterval(loadPortfolio, 30_000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
