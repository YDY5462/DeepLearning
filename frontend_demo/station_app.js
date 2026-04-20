const AUTO_STEP_MS = 3000;

if (typeof echarts === "undefined") {
  document.body.innerHTML =
    '<main style="padding:20px;font-family:sans-serif;color:#b42318;">' +
    "ECharts load failed. Please check frontend_demo/vendor/echarts.min.js" +
    "</main>";
  throw new Error("ECharts is not loaded");
}

function q(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function fmt(num) {
  return Number(num).toLocaleString("zh-CN");
}

function pct(v, digits = 2) {
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

function riskBadge(level) {
  if (level === "high") return '<span class="badge high">高</span>';
  if (level === "mid") return '<span class="badge mid">中</span>';
  return '<span class="badge low">低</span>';
}

let state = DemoData.loadState();
const sid = q("sid") || "S001";
const querySample = Number(q("t"));
if (Number.isFinite(querySample)) {
  state = DemoData.setSampleIndex(state, querySample);
  DemoData.saveState(state);
}

const station = DemoData.getStationById(sid);
const stationIndex = station.index;
document.getElementById("stationHeaderTitle").textContent = `${station.id} / ${station.name} (${station.line})`;

const playBtn = document.getElementById("playBtn");
const slider = document.getElementById("sampleSlider");
const sampleLabelEl = document.getElementById("sampleLabel");
const sampleHintEl = document.getElementById("sampleHint");

const trendChart = echarts.init(document.getElementById("trendChart"));
const errorPieChart = echarts.init(document.getElementById("errorPieChart"));

function setPlayStyle(isAutoplay) {
  playBtn.textContent = isAutoplay ? "暂停自动播放" : "继续自动播放";
  playBtn.classList.toggle("paused", !isAutoplay);
}

function renderSampleHeader(snapshot) {
  slider.max = String(DemoData.SAMPLE_COUNT - 1);
  slider.value = String(snapshot.sampleIndex);
  sampleLabelEl.textContent = snapshot.sampleLabel;
  sampleHintEl.textContent = snapshot.sampleHint;
}

function renderCards(snapshot) {
  const payload = DemoData.getStationPayload(snapshot, stationIndex);
  const cards = [
    { label: "当前样本真值", value: `${fmt(payload.truth)} 人/15min` },
    { label: "当前样本预测", value: `${fmt(payload.pred)} 人/15min` },
    { label: "当前绝对误差", value: `${fmt(payload.absErr)}` },
    { label: "当前 APE", value: pct(payload.ape) },
    { label: "站点平均 MAE", value: payload.meanMAE.toFixed(2) },
    { label: "站点平均 MAPE", value: pct(payload.meanMAPE) },
    { label: "官方 RMSE (merge)", value: snapshot.officialMetrics.RMSE.toFixed(2) },
    { label: "官方 WMAPE (merge)", value: pct(snapshot.officialMetrics.WMAPE) },
  ];

  document.getElementById("detailCards").innerHTML = cards
    .map(
      (c) => `
      <div class="detail-card">
        <label>${c.label}</label>
        <strong>${c.value}</strong>
      </div>
    `
    )
    .join("");
}

function renderTrend(snapshot) {
  const s = DemoData.getStationSeries(stationIndex, snapshot.sampleIndex, 36);

  trendChart.setOption({
    animationDuration: 260,
    tooltip: { trigger: "axis" },
    legend: {
      top: 4,
      data: ["真值", "预测值", "绝对误差"],
      textStyle: { color: "#5d6c79" },
    },
    grid: { left: 40, right: 40, top: 34, bottom: 26 },
    xAxis: {
      type: "category",
      data: s.labels,
      axisLabel: { color: "#5d6c79", fontSize: 11 },
      axisLine: { lineStyle: { color: "#ced8d2" } },
    },
    yAxis: [
      {
        type: "value",
        axisLabel: { color: "#5d6c79", fontSize: 11 },
        splitLine: { lineStyle: { color: "rgba(197,206,201,.45)" } },
      },
      {
        type: "value",
        axisLabel: { color: "#8c5f2b", fontSize: 11 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "真值",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2.8, color: "#1d4ed8" },
        areaStyle: { color: "rgba(29,78,216,.10)" },
        data: s.truth,
      },
      {
        name: "预测值",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2.8, color: "#0f766e" },
        areaStyle: { color: "rgba(15,118,110,.10)" },
        data: s.pred,
      },
      {
        name: "绝对误差",
        type: "bar",
        yAxisIndex: 1,
        barMaxWidth: 12,
        itemStyle: { color: "rgba(180,83,9,.45)" },
        data: s.absError,
      },
    ],
  });
}

function renderErrorDistribution(snapshot) {
  const s = DemoData.getStationSeries(stationIndex, snapshot.sampleIndex, snapshot.sampleIndex + 1);
  let low = 0;
  let mid = 0;
  let high = 0;
  s.absError.forEach((v) => {
    const level = DemoData.riskLevelByError(v);
    if (level === "high") high += 1;
    else if (level === "mid") mid += 1;
    else low += 1;
  });

  errorPieChart.setOption({
    tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
    legend: {
      bottom: 0,
      textStyle: { color: "#5d6c79" },
    },
    series: [
      {
        name: "误差等级分布",
        type: "pie",
        radius: ["42%", "72%"],
        center: ["50%", "45%"],
        label: { formatter: "{b}\n{d}%" },
        data: [
          { value: low, name: "低误差 (|e|<70)", itemStyle: { color: "#0f766e" } },
          { value: mid, name: "中误差 (70-119)", itemStyle: { color: "#b45309" } },
          { value: high, name: "高误差 (>=120)", itemStyle: { color: "#b42318" } },
        ],
      },
    ],
  });
}

function renderTable(snapshot) {
  const s = DemoData.getStationSeries(stationIndex, snapshot.sampleIndex, 8);
  const rows = [];
  for (let i = s.labels.length - 1; i >= 0; i--) {
    const absErr = s.absError[i];
    const level = DemoData.riskLevelByError(absErr);
    rows.push({
      label: s.labels[i],
      truth: s.truth[i],
      pred: s.pred[i],
      absErr,
      ape: s.ape[i],
      risk: level,
    });
  }

  document.getElementById("snapshotTable").innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${r.label}</td>
        <td>${fmt(r.truth)}</td>
        <td>${fmt(r.pred)}</td>
        <td>${fmt(r.absErr)}</td>
        <td>${pct(r.ape)}</td>
        <td>${riskBadge(r.risk)}</td>
      </tr>
    `
    )
    .join("");
}

function renderProvenance(snapshot) {
  const lines = [];
  lines.push(`<div class="meta-line"><code>prediction</code>: ${snapshot.sourceFiles.prediction}</div>`);
  lines.push(`<div class="meta-line"><code>ground truth</code>: ${snapshot.sourceFiles.groundTruth}</div>`);
  lines.push(
    `<div class="meta-line">官方 merge 指标: RMSE=${snapshot.officialMetrics.RMSE.toFixed(4)}, ` +
      `R2=${snapshot.officialMetrics.R2.toFixed(6)}, MAE=${snapshot.officialMetrics.MAE.toFixed(4)}, ` +
      `WMAPE=${pct(snapshot.officialMetrics.WMAPE, 4)}。</div>`
  );
  lines.push(
    `<div class="meta-line">整矩阵复算指标: RMSE=${snapshot.recomputedMetrics.RMSE.toFixed(4)}, ` +
      `R2=${snapshot.recomputedMetrics.R2.toFixed(6)}, MAE=${snapshot.recomputedMetrics.MAE.toFixed(4)}, ` +
      `WMAPE=${pct(snapshot.recomputedMetrics.WMAPE, 4)}。</div>`
  );
  lines.push(
    `<div class="meta-line">说明: 官方 merge 与整矩阵复算 R2 不同，属于统计口径差异，答辩时建议直接声明。</div>`
  );

  document.getElementById("provenanceBox").innerHTML = lines.join("");
}

function renderAll() {
  const snapshot = DemoData.getSnapshot(state);
  renderSampleHeader(snapshot);
  renderCards(snapshot);
  renderTrend(snapshot);
  renderErrorDistribution(snapshot);
  renderTable(snapshot);
  renderProvenance(snapshot);
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
}

function init() {
  setPlayStyle(state.autoplay);
  bindControls();
  renderAll();

  setInterval(() => {
    if (!state.autoplay) return;
    state = DemoData.nextSample(state);
    DemoData.saveState(state);
    renderAll();
  }, AUTO_STEP_MS);

  window.addEventListener("resize", () => {
    trendChart.resize();
    errorPieChart.resize();
  });
}

init();
