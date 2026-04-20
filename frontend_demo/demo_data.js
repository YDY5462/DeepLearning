/* Shared state and selectors for real model output visualization */
(function () {
  const STORE_KEY = "metro_real_demo_state_v1";
  const STATE_VERSION = 1;
  const GRID_ROWS = 12;
  const GRID_COLS = 23;
  const bundle = window.ModelBundle;
  const STATION_NAME_MAP = window.StationNameMap || {};

  if (!bundle) {
    console.error("model_bundle.js is missing. Run: python frontend_demo/build_model_bundle.py");
    return;
  }

  const STATION_COUNT = bundle.stationCount;
  const SAMPLE_COUNT = bundle.sampleCount;

  function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
  }

  function stationName(index) {
    return `S${String(index + 1).padStart(3, "0")}`;
  }

  function lineName(row) {
    return `L${String(row + 1).padStart(2, "0")}`;
  }

  function shortHash(value, n = 12) {
    return String(value || "").slice(0, n);
  }

  function getMappedMeta(stationId, fallbackLine, index) {
    const byId = STATION_NAME_MAP[stationId];
    const byIndex = STATION_NAME_MAP[index + 1] || STATION_NAME_MAP[String(index + 1)];
    const raw = byId || byIndex || null;

    if (!raw) {
      return {
        name: `Station ${stationId}`,
        line: fallbackLine,
      };
    }

    if (typeof raw === "string") {
      return {
        name: raw.trim() || `Station ${stationId}`,
        line: fallbackLine,
      };
    }

    const mappedName = String(raw.name || raw.station_name || "").trim();
    const mappedLine = String(raw.line || raw.line_name || "").trim();
    return {
      name: mappedName || `Station ${stationId}`,
      line: mappedLine || fallbackLine,
    };
  }

  function makeStations() {
    const stations = [];
    for (let index = 0; index < STATION_COUNT; index++) {
      const row = Math.floor(index / GRID_COLS);
      const col = index % GRID_COLS;
      const id = stationName(index);
      const fallbackLine = lineName(row);
      const mapped = getMappedMeta(id, fallbackLine, index);
      const x = col * 10 + (row % 2 === 0 ? 1.4 : -1.4) + Math.sin((col + row) / 2.5) * 0.8;
      const y = row * 8.4 + Math.cos((col - row) / 3.2) * 0.7;
      stations.push({
        index,
        id,
        name: mapped.name,
        line: mapped.line,
        row,
        col,
        x,
        y,
      });
    }
    return stations;
  }

  function makeLinks() {
    const links = [];
    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS - 1; col++) {
        const a = row * GRID_COLS + col;
        const b = a + 1;
        if (a < STATION_COUNT && b < STATION_COUNT) {
          links.push({ source: stationName(a), target: stationName(b), type: "line" });
        }
      }
    }
    const transferCols = [3, 7, 11, 15, 19];
    for (let row = 0; row < GRID_ROWS - 1; row++) {
      transferCols.forEach((col) => {
        if ((row + col) % 2 === 0) {
          const a = row * GRID_COLS + col;
          const b = (row + 1) * GRID_COLS + col;
          if (a < STATION_COUNT && b < STATION_COUNT) {
            links.push({ source: stationName(a), target: stationName(b), type: "transfer" });
          }
        }
      });
    }
    return links;
  }

  const stations = makeStations();
  const links = makeLinks();

  function parseState(raw) {
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || parsed.version !== STATE_VERSION) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  }

  function createInitialState() {
    return {
      version: STATE_VERSION,
      sampleIndex: 0,
      selectedIndex: 0,
      mode: "truth",
      autoplay: true,
      updatedAt: new Date().toISOString(),
    };
  }

  function loadState() {
    const parsed = parseState(window.sessionStorage.getItem(STORE_KEY));
    if (parsed) return parsed;
    const fresh = createInitialState();
    saveState(fresh);
    return fresh;
  }

  function saveState(state) {
    window.sessionStorage.setItem(STORE_KEY, JSON.stringify(state));
  }

  function setSampleIndex(state, sampleIndex) {
    const next = state;
    next.sampleIndex = clamp(Math.round(Number(sampleIndex) || 0), 0, SAMPLE_COUNT - 1);
    next.updatedAt = new Date().toISOString();
    return next;
  }

  function nextSample(state) {
    const next = state;
    next.sampleIndex = (state.sampleIndex + 1) % SAMPLE_COUNT;
    next.updatedAt = new Date().toISOString();
    return next;
  }

  function riskLevelByError(absError) {
    if (absError >= 120) return "high";
    if (absError >= 70) return "mid";
    return "low";
  }

  function riskLabel(level) {
    if (level === "high") return "High";
    if (level === "mid") return "Mid";
    return "Low";
  }

  function riskColor(level) {
    if (level === "high") return "#b42318";
    if (level === "mid") return "#b45309";
    return "#0f766e";
  }

  function getSnapshot(state) {
    const t = clamp(state.sampleIndex, 0, SAMPLE_COUNT - 1);
    const truth = bundle.trueMatrix[t];
    const pred = bundle.predMatrix[t];
    const error = new Array(STATION_COUNT);
    const absError = new Array(STATION_COUNT);
    const ape = new Array(STATION_COUNT);

    let sumTruth = 0;
    let sumPred = 0;
    let sumAbs = 0;
    let sumSq = 0;
    let highErrorStations = 0;

    for (let i = 0; i < STATION_COUNT; i++) {
      const e = pred[i] - truth[i];
      const abs = Math.abs(e);
      const denom = Math.max(truth[i], 1);
      const thisApe = abs / denom;

      error[i] = e;
      absError[i] = abs;
      ape[i] = thisApe;

      sumTruth += truth[i];
      sumPred += pred[i];
      sumAbs += abs;
      sumSq += e * e;
      if (abs >= 120) highErrorStations += 1;
    }

    const sampleMAE = sumAbs / STATION_COUNT;
    const sampleRMSE = Math.sqrt(sumSq / STATION_COUNT);
    const sampleWMAPE = sumAbs / Math.max(sumTruth, 1);

    return {
      sampleIndex: t,
      sampleLabel: `Test Sample #${String(t + 1).padStart(3, "0")}`,
      sampleHint: "Displayed in test-set sequence order",
      truth,
      pred,
      error,
      absError,
      ape,
      sumTruth,
      sumPred,
      sampleMAE,
      sampleRMSE,
      sampleWMAPE,
      highErrorStations,
      stations,
      links,
      officialMetrics: bundle.officialMergeMetrics,
      recomputedMetrics: bundle.recomputedMetrics,
      latestRunMetrics: bundle.latestRunMetrics,
      sourceFiles: bundle.sourceFiles,
      provenance: bundle.provenance,
      dataset: bundle.dataset,
      generatedAtUtc: bundle.generatedAtUtc,
      stationSummary: bundle.stationSummary,
    };
  }

  function getStationPayload(snapshot, stationIndex) {
    const i = clamp(stationIndex, 0, STATION_COUNT - 1);
    const station = stations[i];
    const truth = snapshot.truth[i];
    const pred = snapshot.pred[i];
    const err = snapshot.error[i];
    const absErr = snapshot.absError[i];
    const ape = snapshot.ape[i];
    const meanMAE = snapshot.stationSummary.meanMAE[i];
    const meanMAPE = snapshot.stationSummary.meanMAPE[i];
    const risk = riskLevelByError(absErr);
    return {
      station,
      truth,
      pred,
      err,
      absErr,
      ape,
      meanMAE,
      meanMAPE,
      risk,
    };
  }

  function getStationSeries(stationIndex, endSample, lookback) {
    const i = clamp(stationIndex, 0, STATION_COUNT - 1);
    const end = clamp(endSample, 0, SAMPLE_COUNT - 1);
    const len = Math.max(1, lookback || 24);
    const start = Math.max(0, end - len + 1);

    const labels = [];
    const truth = [];
    const pred = [];
    const absError = [];
    const ape = [];

    for (let t = start; t <= end; t++) {
      const y = bundle.trueMatrix[t][i];
      const p = bundle.predMatrix[t][i];
      const ae = Math.abs(p - y);
      labels.push(`#${String(t + 1).padStart(3, "0")}`);
      truth.push(y);
      pred.push(p);
      absError.push(ae);
      ape.push(ae / Math.max(y, 1));
    }

    return { labels, truth, pred, absError, ape, startSample: start, endSample: end };
  }

  function getTopErrorStations(snapshot, limit = 10) {
    const rows = stations.map((station, i) => ({
      index: i,
      station,
      truth: snapshot.truth[i],
      pred: snapshot.pred[i],
      err: snapshot.error[i],
      absErr: snapshot.absError[i],
      ape: snapshot.ape[i],
      risk: riskLevelByError(snapshot.absError[i]),
    }));
    rows.sort((a, b) => b.absErr - a.absErr);
    return rows.slice(0, limit);
  }

  function getStationById(stationId) {
    const match = stations.find((s) => s.id === stationId);
    if (match) return match;
    return stations[0];
  }

  function getMetaSummary() {
    const mappedStationCount = stations.filter((s) => !String(s.name).startsWith("Station S")).length;
    return {
      stationCount: STATION_COUNT,
      sampleCount: SAMPLE_COUNT,
      mappedStationCount,
      timeGranularityMin: bundle.timeGranularityMin,
      dataset: bundle.dataset,
      generatedAtUtc: bundle.generatedAtUtc,
      sourceFiles: bundle.sourceFiles,
      provenance: bundle.provenance.map((row) => ({
        file: row.file,
        metricValue: row.metricValue,
        gitBlobId: shortHash(row.gitBlobId, 10),
        sha256: shortHash(row.sha256, 14),
      })),
    };
  }

  window.DemoData = {
    STATION_COUNT,
    SAMPLE_COUNT,
    stations,
    links,
    stationName,
    lineName,
    loadState,
    saveState,
    setSampleIndex,
    nextSample,
    getSnapshot,
    getStationPayload,
    getStationSeries,
    getTopErrorStations,
    getStationById,
    riskLevelByError,
    riskLabel,
    riskColor,
    getMetaSummary,
  };
})();
