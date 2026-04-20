const AUTO_STEP_MS = 3000;

if (typeof echarts === "undefined") {
  document.body.innerHTML =
    '<main style="padding:20px;font-family:sans-serif;color:#b42318;">' +
    "ECharts load failed. Please check frontend_demo/vendor/echarts.min.js" +
    "</main>";
  throw new Error("ECharts is not loaded");
}

let state = DemoData.loadState();
let selectedIndex = state.selectedIndex || 0;
let mode = state.mode || "truth";
if (!["truth", "pred", "error"].includes(mode)) {
  mode = "truth";
  state.mode = mode;
  DemoData.saveState(state);
}

const chartMap = echarts.init(document.getElementById("networkMap"));
const chartMini = echarts.init(document.getElementById("miniTrend"));

const playBtn = document.getElementById("playBtn");
const slider = document.getElementById("sampleSlider");
const sampleLabelEl = document.getElementById("sampleLabel");
const sampleHintEl = document.getElementById("sampleHint");
const datasetTagEl = document.getElementById("datasetTag");

function fmt(num) {
  return Number(num).toLocaleString("zh-CN");
}

function pct(v, digits = 2) {
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

function valueByMode(snapshot, i) {
  if (mode === "truth") return snapshot.truth[i];
  if (mode === "pred") return snapshot.pred[i];
  return snapshot.absError[i];
}

function modeLabel() {
  if (mode === "truth") return "Truth";
  if (mode === "pred") return "Prediction";
  return "Absolute Error";
}

function modeRamp() {
  if (mode === "truth") return ["#dbeafe", "#1d4ed8"];
  if (mode === "pred") return ["#dcfce7", "#0f766e"];
  return ["#fff7ed", "#b42318"];
}

function modeRange(snapshot) {
  const values = snapshot.stations.map((_, i) => valueByMode(snapshot, i));
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) max = min + 1;
  return { min, max };
}

function nodeSize(snapshot, i) {
  const value = valueByMode(snapshot, i);
  if (mode === "error") return 7 + Math.min(18, value / 12);
  return 7 + Math.min(18, value / 220);
}

function setPlayStyle(isAutoplay) {
  playBtn.textContent = isAutoplay ? "Pause Autoplay" : "Resume Autoplay";
  playBtn.classList.toggle("paused", !isAutoplay);
}

function renderSampleHeader(snapshot) {
  slider.max = String(DemoData.SAMPLE_COUNT - 1);
  slider.value = String(snapshot.sampleIndex);
  sampleLabelEl.textContent = snapshot.sampleLabel;
  sampleHintEl.textContent = snapshot.sampleHint;
  datasetTagEl.textContent = `Dataset: ${snapshot.dataset}`;
  document.getElementById("updateTime").textContent =
    `Updated: ${new Date(state.updatedAt).toLocaleString("zh-CN", { hour12: false })}`;
}

function renderStationCard(snapshot, stationIndex) {
  const payload = DemoData.getStationPayload(snapshot, stationIndex);
  const st = payload.station;
  selectedIndex = st.index;
  state.selectedIndex = selectedIndex;
  DemoData.saveState(state);

  document.getElementById("stationTitle").textContent = `${st.id} / ${st.name}`;
  document.getElementById("stationLine").textContent = `Line: ${st.line}`;
  document.getElementById("curTruth").textContent = `${fmt(payload.truth)} pax/15min`;
  document.getElementById("curPred").textContent = `${fmt(payload.pred)} pax/15min`;
  document.getElementById("curAbsError").textContent = fmt(payload.absErr);
  document.getElementById("curAPE").textContent = pct(payload.ape);
  document.getElementById("stationMAE").textContent = payload.meanMAE.toFixed(2);
  document.getElementById("errorRisk").textContent = DemoData.riskLabel(payload.risk);
  document.getElementById("detailLink").href = `./station.html?sid=${st.id}&t=${snapshot.sampleIndex}`;

  renderMiniTrend(snapshot.sampleIndex, selectedIndex);
}

function renderMiniTrend(sampleIndex, stationIndex) {
  const series = DemoData.getStationSeries(stationIndex, sampleIndex, 24);
  chartMini.setOption({
    animationDuration: 250,
    tooltip: { trigger: "axis" },
    legend: {
      top: 4,
      data: ["Truth", "Prediction", "Abs Error"],
      textStyle: { color: "#526070", fontSize: 12 },
    },
    grid: { left: 36, right: 36, top: 30, bottom: 22 },
    xAxis: {
      type: "category",
      data: series.labels,
      axisLabel: { color: "#60717e", fontSize: 11 },
      axisLine: { lineStyle: { color: "#ccd5d0" } },
    },
    yAxis: [
      {
        type: "value",
        axisLabel: { color: "#60717e", fontSize: 11 },
        splitLine: { lineStyle: { color: "rgba(190,202,197,.45)" } },
      },
      {
        type: "value",
        axisLabel: { color: "#8a5a2b", fontSize: 11 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "Truth",
        type: "line",
        data: series.truth,
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2.3, color: "#1d4ed8" },
      },
      {
        name: "Prediction",
        type: "line",
        data: series.pred,
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2.3, color: "#0f766e" },
      },
      {
        name: "Abs Error",
        type: "bar",
        yAxisIndex: 1,
        data: series.absError,
        itemStyle: { color: "rgba(180,83,9,.45)" },
        barMaxWidth: 10,
      },
    ],
  });
}

function buildMapData(snapshot) {
  const stationIndexById = new Map(snapshot.stations.map((s, i) => [s.id, i]));
  const nodeData = snapshot.stations.map((s, i) => ({
    name: s.id,
    value: [s.x, s.y, valueByMode(snapshot, i), i],
    symbolSize: nodeSize(snapshot, i),
    label: { show: false },
  }));

  const lineData = snapshot.links.map((link) => {
    const a = snapshot.stations[stationIndexById.get(link.source)];
    const b = snapshot.stations[stationIndexById.get(link.target)];
    return {
      coords: [
        [a.x, a.y],
        [b.x, b.y],
      ],
      lineStyle: {
        color: link.type === "transfer" ? "rgba(29,78,216,0.34)" : "rgba(34,52,69,0.22)",
        width: link.type === "transfer" ? 1.5 : 1,
      },
    };
  });

  return { nodeData, lineData };
}

function renderMap(snapshot) {
  const { nodeData, lineData } = buildMapData(snapshot);
  const maxX = Math.max(...snapshot.stations.map((s) => s.x)) + 2;
  const maxY = Math.max(...snapshot.stations.map((s) => s.y)) + 2;
  const range = modeRange(snapshot);

  chartMap.setOption({
    animationDuration: 350,
    tooltip: {
      trigger: "item",
      formatter: (params) => {
        if (params.seriesName !== "Station") return "";
        const idx = params.data.value[3];
        const payload = DemoData.getStationPayload(snapshot, idx);
        return [
          `<strong>${payload.station.id}</strong> (${payload.station.line})`,
          `${modeLabel()}: ${fmt(valueByMode(snapshot, idx))}`,
          `Truth: ${fmt(payload.truth)} pax/15min`,
          `Prediction: ${fmt(payload.pred)} pax/15min`,
          `Abs Error: ${fmt(payload.absErr)}`,
          `APE: ${pct(payload.ape)}`,
        ].join("<br/>");
      },
    },
    xAxis: { type: "value", min: -2, max: maxX, show: false },
    yAxis: { type: "value", min: -2, max: maxY, inverse: true, show: false },
    visualMap: {
      show: false,
      min: range.min,
      max: range.max,
      dimension: 2,
      inRange: { color: modeRamp() },
    },
    series: [
      {
        name: "Network",
        type: "lines",
        coordinateSystem: "cartesian2d",
        data: lineData,
        silent: true,
        z: 1,
      },
      {
        name: "Station",
        type: "scatter",
        coordinateSystem: "cartesian2d",
        data: nodeData,
        emphasis: {
          scale: 1.25,
          itemStyle: { borderWidth: 2, borderColor: "#0f172a" },
          label: {
            show: true,
            formatter: "{b}",
            position: "right",
            color: "#0f172a",
            fontSize: 11,
            backgroundColor: "rgba(255,255,255,0.92)",
            padding: [2, 5],
            borderRadius: 4,
          },
        },
        z: 2,
      },
    ],
  });
}

function renderKpis(snapshot) {
  const official = snapshot.officialMetrics;
  const cards = [
    { label: "Total Truth", value: fmt(snapshot.sumTruth), sub: "pax/15min" },
    { label: "Total Prediction", value: fmt(snapshot.sumPred), sub: "pax/15min" },
    { label: "Sample MAE", value: snapshot.sampleMAE.toFixed(2), sub: "mean abs error by station" },
    { label: "Sample RMSE", value: snapshot.sampleRMSE.toFixed(2), sub: "root mean square error" },
    { label: "Sample WMAPE", value: pct(snapshot.sampleWMAPE), sub: "sample level" },
    { label: "Official RMSE", value: official.RMSE.toFixed(2), sub: "final result/15" },
    { label: "Official WMAPE", value: pct(official.WMAPE), sub: "final result/15" },
    { label: "High Error Stations", value: `${snapshot.highErrorStations}`, sub: "threshold: abs error >= 120" },
  ];

  document.getElementById("kpiRow").innerHTML = cards
    .map(
      (k) => `
      <div class="kpi-item">
        <label>${k.label}</label>
        <strong>${k.value}</strong>
        <small>${k.sub}</small>
      </div>
    `
    )
    .join("");
}

function renderTopError(snapshot) {
  const rows = DemoData.getTopErrorStations(snapshot, 10);
  document.getElementById("topError").innerHTML = rows
    .map(
      (row) =>
        `<li><a class="detail-link" href="./station.html?sid=${row.station.id}&t=${snapshot.sampleIndex}">${row.station.id}</a> ` +
        `truth ${fmt(row.truth)} / pred ${fmt(row.pred)} / abs error ${fmt(row.absErr)} / APE ${pct(row.ape)}</li>`
    )
    .join("");
}

function renderProvenance(snapshot) {
  const lines = [];
  lines.push(`<div class="meta-line"><code>prediction</code>: ${snapshot.sourceFiles.prediction}</div>`);
  lines.push(`<div class="meta-line"><code>ground truth</code>: ${snapshot.sourceFiles.groundTruth}</div>`);
  lines.push(`<div class="meta-line"><code>bundle generated</code>: ${snapshot.generatedAtUtc}</div>`);

  snapshot.provenance.forEach((item) => {
    lines.push(
      `<div class="meta-line"><code>${item.file}</code> = ${Number(item.metricValue).toFixed(6)} ` +
      `(blob ${String(item.gitBlobId).slice(0, 10)}, sha256 ${String(item.sha256).slice(0, 12)}...)</div>`
    );
  });

  lines.push(
    `<div class="meta-line">R2 scope mismatch: official merge = ${snapshot.officialMetrics.R2.toFixed(6)}, ` +
      `frontend matrix recompute = ${snapshot.recomputedMetrics.R2.toFixed(6)}.</div>`
  );

  document.getElementById("provenanceBox").innerHTML = lines.join("");
}

function renderAll() {
  const snapshot = DemoData.getSnapshot(state);
  renderSampleHeader(snapshot);
  renderMap(snapshot);
  renderStationCard(snapshot, selectedIndex);
  renderKpis(snapshot);
  renderTopError(snapshot);
  renderProvenance(snapshot);
}

function bindMapEvents() {
  chartMap.on("mouseover", (params) => {
    if (params.seriesName !== "Station") return;
    const idx = params.data.value[3];
    const snapshot = DemoData.getSnapshot(state);
    renderStationCard(snapshot, idx);
  });

  chartMap.on("click", (params) => {
    if (params.seriesName !== "Station") return;
    const idx = params.data.value[3];
    const sid = DemoData.stations[idx].id;
    window.location.href = `./station.html?sid=${sid}&t=${state.sampleIndex}`;
  });
}

function bindModeSwitch() {
  const buttons = document.querySelectorAll(".mode-btn");
  buttons.forEach((x) => x.classList.toggle("active", x.dataset.mode === mode));
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      mode = btn.dataset.mode;
      state.mode = mode;
      DemoData.saveState(state);
      renderAll();
    });
  });
}

function bindControls() {
  playBtn.addEventListener("click", () => {
    state.autoplay = !state.autoplay;
    DemoData.saveState(state);
    setPlayStyle(state.autoplay);
  });

  slider.addEventListener("input", () => {
    state = DemoData.setSampleIndex(state, Number(slider.value));
    DemoData.saveState(state);
    renderAll();
  });

  document.getElementById("randomStationBtn").addEventListener("click", () => {
    selectedIndex = Math.floor(Math.random() * DemoData.STATION_COUNT);
    const snapshot = DemoData.getSnapshot(state);
    renderStationCard(snapshot, selectedIndex);
  });
}

function init() {
  setPlayStyle(state.autoplay);
  bindMapEvents();
  bindModeSwitch();
  bindControls();
  renderAll();

  setInterval(() => {
    if (!state.autoplay) return;
    state = DemoData.nextSample(state);
    DemoData.saveState(state);
    renderAll();
  }, AUTO_STEP_MS);

  window.addEventListener("resize", () => {
    chartMap.resize();
    chartMini.resize();
  });
}

init();
