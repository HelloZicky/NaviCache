const SVG_NS = "http://www.w3.org/2000/svg";
const X_MAX = 0.43;
const Y_MAX = 0.82;
const PLOT = { left: 42, top: 14, width: 302, height: 214 };

const scenes = [...document.querySelectorAll(".scene")];
const navButtons = [...document.querySelectorAll("[data-goto]")];
const chapterIndex = document.querySelector("#chapter-index");
const chapterName = document.querySelector("#chapter-name");
const dataStatus = document.querySelector("#data-status");
const progressFill = document.querySelector("#global-progress-fill");
const motionToggle = document.querySelector("#motion-toggle");
const motionLabel = motionToggle.querySelector("span");
const videos = [...document.querySelectorAll(".comparison-video")];
const gaugeFill = document.querySelector("#gauge-fill");
const errorValue = document.querySelector("#error-value");
const decisionState = document.querySelector("#decision-state");
const stepValue = document.querySelector("#step-value");
const rail = document.querySelector("#decision-rail");
const token = document.querySelector(".signal-token");
const decisionBoard = document.querySelector(".decision-board");

const chapters = {
  trajectory: ["01", "TRACK THE MANIFOLD"],
  decision: ["02", "BOUND THE ERROR"],
  results: ["03", "PRESERVE THE MOTION"]
};

let master = null;
let decisions = [];
let railCells = [];
let manualPaused = false;
let activeScene = "trajectory";

const fallbackRaw = [
  [0.3973089159,0.7465339303],[0.1038883999,0.3085092306],[0.0794156492,0.1904211193],
  [0.0598831773,0.1270166337],[0.0632802024,0.1106812060],[0.0560389422,0.1138660237],
  [0.0584373213,0.0838896856],[0.0472819135,0.0777225047],[0.0856295079,0.1210419387],
  [0.0449128449,0.0796626955],[0.0507972650,0.0524236113],[0.0478680767,0.0688784495],
  [0.0539037175,0.0474564992],[0.0472859144,0.0627444685],[0.0528944582,0.0492547005],
  [0.0506104119,0.0639290810],[0.0600932762,0.0367296822],[0.0652858540,0.0410635881],
  [0.0580828749,0.0561583340],[0.0711809322,0.0494445935],[0.0671812892,0.0419748351],
  [0.0722742006,0.0632077456],[0.0995402709,0.0462118872],[0.1025526375,0.0472448990],
  [0.1245614886,0.0546590798],[0.1456945390,0.0614619367],[0.1988047510,0.0997735858],
  [0.2465351224,0.1208989173],[0.3554533422,0.2544613183]
].map(([x,y], index) => ({ step: index + 1, x, y }));

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((key, index) => [key, values[index]]));
  });
}

async function loadTrajectoryData() {
  try {
    const response = await fetch("assets/merged_stats_diff.csv");
    if (!response.ok) throw new Error(`CSV ${response.status}`);
    const rows = parseCsv(await response.text());
    const cleaned = rows.filter((row) => Number(row.step) !== 0);
    const representative = cleaned
      .filter((row) => row.experiment_id === "subject_consistency/prompt_00")
      .map((row) => ({ step: Number(row.step), x: Number(row.input_rel_l1), y: Number(row.output_rel_l1) }))
      .sort((a, b) => a.step - b.step);
    if (representative.length !== 29) throw new Error("Representative trajectory is incomplete");
    dataStatus.textContent = `CSV LIVE · STEP 0 REMOVED · ${representative.length} POINTS`;
    return { raw: representative, all: cleaned.map((row) => ({ x: Number(row.input_rel_l1), y: Number(row.output_rel_l1) })) };
  } catch (error) {
    console.warn("Using embedded trajectory fallback:", error);
    dataStatus.textContent = "EMBEDDED TRACE · STEP 0 REMOVED · 29 POINTS";
    return { raw: fallbackRaw, all: fallbackRaw };
  }
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const augmented = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
    }
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
    const divisor = augmented[col][col] || 1e-9;
    for (let j = col; j <= n; j += 1) augmented[col][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = augmented[row][col];
      for (let j = col; j <= n; j += 1) augmented[row][j] -= factor * augmented[col][j];
    }
  }
  return augmented.map((row) => row[n]);
}

function polynomialFit(points, degree = 4) {
  const sums = Array.from({ length: degree * 2 + 1 }, (_, power) => points.reduce((total, point) => total + point.x ** power, 0));
  const matrix = Array.from({ length: degree + 1 }, (_, row) => Array.from({ length: degree + 1 }, (_, col) => sums[row + col]));
  const vector = Array.from({ length: degree + 1 }, (_, power) => points.reduce((total, point) => total + point.y * point.x ** power, 0));
  return solveLinearSystem(matrix, vector);
}

function evaluatePolynomial(coefficients, x) {
  return coefficients.reduce((sum, coefficient, power) => sum + coefficient * x ** power, 0);
}

function buildPredictions(raw, all) {
  const coefficients = polynomialFit(all, 4);
  const tea = raw.map((point) => ({ ...point, y: Math.max(0.025, Math.min(0.78, evaluatePolynomial(coefficients, point.x))) }));

  const easy = raw.map((point, index) => {
    if (index === 0) return { ...point, y: point.y * 0.9 };
    const previous = raw[index - 1];
    const heldRatio = previous.y / Math.max(previous.x, 1e-5);
    return { ...point, y: Math.max(0.025, Math.min(0.78, heldRatio * point.x)) };
  });

  let ratio = raw[0].y / raw[0].x;
  let velocity = 0;
  let uncertainty = 1;
  const processNoise = 0.05;
  const measurementNoise = 0.05;
  const navi = raw.map((point, index) => {
    const observation = point.y / Math.max(point.x, 1e-5);
    if (index < 5) {
      const previousRatio = ratio;
      ratio = observation;
      velocity = ratio - previousRatio;
      uncertainty = Math.max(0.08, uncertainty * 0.55);
      return { ...point, y: point.y };
    }
    const prior = ratio + velocity * 0.42;
    uncertainty += processNoise;
    const gain = uncertainty / (uncertainty + measurementNoise);
    const posterior = prior + gain * (observation - prior);
    velocity = velocity * 0.72 + (posterior - ratio) * 0.28;
    ratio = posterior;
    uncertainty = (1 - gain) * uncertainty;
    return { ...point, y: Math.max(0.02, Math.min(0.8, ratio * point.x)) };
  });
  return { raw, tea, easy, navi };
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function toSvgPoint(point) {
  return {
    x: PLOT.left + (point.x / X_MAX) * PLOT.width,
    y: PLOT.top + PLOT.height - (point.y / Y_MAX) * PLOT.height
  };
}

function pathFrom(points) {
  return points.map((point, index) => {
    const mapped = toSvgPoint(point);
    return `${index ? "L" : "M"}${mapped.x.toFixed(2)},${mapped.y.toFixed(2)}`;
  }).join(" ");
}

function deviationPath(raw, predicted) {
  return `${pathFrom(raw)} ${pathFrom([...predicted].reverse()).replace(/^M/, "L")} Z`;
}

function addChartScaffold(svg) {
  [0.2, 0.4, 0.6, 0.8].forEach((value) => {
    const y = toSvgPoint({ x: 0, y: value }).y;
    svg.appendChild(svgElement("line", { x1: PLOT.left, y1: y, x2: PLOT.left + PLOT.width, y2: y, class: "chart-grid" }));
    const label = svgElement("text", { x: PLOT.left - 8, y: y + 3, class: "chart-label", "text-anchor": "end" });
    label.textContent = value.toFixed(1);
    svg.appendChild(label);
  });
  [0.1, 0.2, 0.3, 0.4].forEach((value) => {
    const x = toSvgPoint({ x: value, y: 0 }).x;
    svg.appendChild(svgElement("line", { x1: x, y1: PLOT.top, x2: x, y2: PLOT.top + PLOT.height, class: "chart-grid" }));
    const label = svgElement("text", { x, y: PLOT.top + PLOT.height + 17, class: "chart-label", "text-anchor": "middle" });
    label.textContent = value.toFixed(1);
    svg.appendChild(label);
  });
  svg.appendChild(svgElement("line", { x1: PLOT.left, y1: PLOT.top + PLOT.height, x2: PLOT.left + PLOT.width, y2: PLOT.top + PLOT.height, class: "chart-axis" }));
  svg.appendChild(svgElement("line", { x1: PLOT.left, y1: PLOT.top, x2: PLOT.left, y2: PLOT.top + PLOT.height, class: "chart-axis" }));
  const xLabel = svgElement("text", { x: PLOT.left + PLOT.width / 2, y: 260, class: "chart-label", "text-anchor": "middle" });
  xLabel.textContent = "INPUT DIFFERENCE ΔI";
  svg.appendChild(xLabel);
}

function renderCharts(series) {
  document.querySelectorAll(".chart-card").forEach((card) => {
    const method = card.dataset.method;
    const svg = card.querySelector("svg");
    addChartScaffold(svg);

    if (method !== "raw") {
      svg.appendChild(svgElement("path", { d: deviationPath(series.raw, series[method]), class: "deviation-area" }));
      svg.appendChild(svgElement("path", { d: pathFrom(series.raw), class: "raw-line" }));
    }

    const lineClass = method === "raw" ? "raw-line" : "estimate-line";
    const line = svgElement("path", { d: pathFrom(series[method]), class: lineClass });
    svg.appendChild(line);
    series[method].forEach((point) => {
      const mapped = toSvgPoint(point);
      svg.appendChild(svgElement("circle", { cx: mapped.x, cy: mapped.y, r: method === "navi" ? 3 : 2.3, class: "data-dot" }));
    });
  });

  document.querySelectorAll(".raw-line, .estimate-line").forEach((path) => {
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;
  });
}

function buildDecisions(raw, navi) {
  let accumulated = 0;
  return raw.map((point, index) => {
    const localMotion = index ? Math.abs(point.y - raw[index - 1].y) : point.y;
    const predictionError = Math.abs(point.y - navi[index].y) + localMotion * 0.16;
    if (index < 5) return { type: "compute", error: 0.012 + index * 0.006 };
    accumulated += predictionError;
    if (accumulated >= 0.07) {
      const error = accumulated;
      accumulated = 0;
      return { type: "update", error };
    }
    return { type: "skip", error: accumulated };
  });
}

function renderDecisionRail() {
  decisions.forEach((decision, index) => {
    const cell = document.createElement("i");
    cell.className = `rail-cell ${decision.type}`;
    cell.style.setProperty("--h", `${28 + ((index * 43) % 68)}%`);
    cell.setAttribute("aria-hidden", "true");
    rail.appendChild(cell);
    railCells.push(cell);
  });
}

function boardPoint(selector, xRatio = 0.5, yRatio = 0.5) {
  const boardRect = decisionBoard.getBoundingClientRect();
  const rect = document.querySelector(selector).getBoundingClientRect();
  return {
    x: rect.left - boardRect.left + rect.width * xRatio,
    y: rect.top - boardRect.top + rect.height * yRatio
  };
}

function orthogonalPath(start, end, direction = "horizontal") {
  if (direction === "vertical") {
    const middleY = (start.y + end.y) / 2;
    return `M ${start.x} ${start.y} V ${middleY} H ${end.x} V ${end.y}`;
  }
  const middleX = (start.x + end.x) / 2;
  return `M ${start.x} ${start.y} H ${middleX} V ${end.y} H ${end.x}`;
}

function updateConnectorPaths() {
  const boardRect = decisionBoard.getBoundingClientRect();
  const connectorLayer = document.querySelector(".connector-layer");
  connectorLayer.setAttribute("viewBox", `0 0 ${boardRect.width} ${boardRect.height}`);
  connectorLayer.setAttribute("preserveAspectRatio", "none");
  const links = {
    "align-predict": [boardPoint(".align-node", 1, 0.5), boardPoint(".predict-node", 0, 0.5), "horizontal"],
    "predict-gate": [boardPoint(".predict-node", 1, 0.5), boardPoint(".gate-node", 0, 0.5), "horizontal"],
    "gate-skip": [boardPoint(".gate-node", 0.32, 1), boardPoint(".skip-node", 0.5, 0), "vertical"],
    "gate-update": [boardPoint(".gate-node", 0.72, 1), boardPoint(".update-node", 0.5, 0), "vertical"]
  };

  document.querySelectorAll(".flow-path").forEach((path) => {
    const [start, end, direction] = links[path.dataset.link];
    path.setAttribute("d", orthogonalPath(start, end, direction));
    path.style.strokeDasharray = "none";
    path.style.strokeDashoffset = "0";
  });
}

function playComparisonVideos() {
  const sharedTime = videos[0]?.currentTime || 0;
  videos.forEach((item, index) => {
    if (index) item.currentTime = sharedTime;
    item.play().catch(() => {});
  });
}

function pauseComparisonVideos() {
  videos.forEach((item) => item.pause());
}

function setDecision(index) {
  const decision = decisions[index];
  railCells.forEach((cell, cellIndex) => {
    cell.classList.toggle("done", cellIndex <= index);
    cell.classList.toggle("current", cellIndex === index);
  });
  stepValue.textContent = String(index + 1).padStart(2, "0");
  decisionState.textContent = decision.type === "compute" ? "INITIAL ALIGNMENT" : decision.type === "update" ? "UPDATE / RE-ANCHOR" : "SAFE TO SKIP";
  decisionState.style.color = decision.type === "update" ? "var(--ember)" : "var(--acid)";
  errorValue.textContent = decision.error.toFixed(3);
  gsap.set(gaugeFill, { scaleX: Math.min(decision.error / 0.1, 1), backgroundColor: decision.type === "update" ? "var(--ember)" : "var(--acid)" });
  document.querySelector(".skip-node").classList.toggle("lit", decision.type === "skip");
  document.querySelector(".update-node").classList.toggle("lit", decision.type === "update");
}

function setChapter(name) {
  activeScene = name;
  const [index, label] = chapters[name];
  chapterIndex.textContent = index;
  chapterName.textContent = label;
  scenes.forEach((scene) => scene.classList.toggle("is-active", scene.dataset.scene === name));
  navButtons.forEach((button) => button.classList.toggle("active", button.dataset.goto === name));
  if (name === "results" && !manualPaused) playComparisonVideos();
  else pauseComparisonVideos();
}

function buildDecisionTimeline() {
  const timeline = gsap.timeline({ defaults: { ease: "power2.inOut" } });
  const stepDuration = 0.18;

  decisions.forEach((decision, index) => {
    const at = index * stepDuration;
    timeline.call(() => setDecision(index), null, at);
  });

  const coordinate = (selector, xRatio = 0.5, yRatio = 0.5, axis = "x") => {
    const point = boardPoint(selector, xRatio, yRatio);
    return point[axis] - 4.5;
  };

  timeline.set(token, {
    x: () => coordinate(".align-node", 1, .5, "x"),
    y: () => coordinate(".align-node", 1, .5, "y"),
    autoAlpha: 1,
    scale: 1,
    backgroundColor: "var(--acid)"
  }, 0)
    .to(token, {
      x: () => coordinate(".predict-node", .5, .5, "x"),
      y: () => coordinate(".predict-node", .5, .5, "y"),
      duration: 1.15,
      ease: "sine.inOut"
    }, .55)
    .to(token, {
      x: () => coordinate(".gate-node", .5, .5, "x"),
      y: () => coordinate(".gate-node", .5, .5, "y"),
      duration: 1.2,
      ease: "sine.inOut"
    }, 1.85)
    .to(token, {
      x: () => coordinate(".skip-node", .5, .5, "x"),
      y: () => coordinate(".skip-node", .5, .5, "y"),
      duration: 1.35,
      ease: "sine.inOut"
    }, 3.2);
  return timeline;
}

function animateCounter(timeline, position) {
  const target = document.querySelector("[data-count]");
  const proxy = { value: 0 };
  timeline.to(proxy, {
    value: Number(target.dataset.count),
    duration: 1.35,
    ease: "power3.out",
    onUpdate: () => { target.firstChild.nodeValue = proxy.value.toFixed(2); }
  }, position);
}

function buildMasterTimeline() {
  const quickProgress = gsap.quickSetter(progressFill, "scaleX");
  const timeline = gsap.timeline({
    repeat: -1,
    repeatDelay: .35,
    defaults: { duration: .72, ease: "power3.out" },
    onUpdate: () => quickProgress(timeline.progress())
  });

  gsap.set(scenes, { autoAlpha: 0 });
  gsap.set(".deviation-area", { autoAlpha: 0 });
  gsap.set(".data-dot", { scale: 0, transformOrigin: "center" });
  gsap.set(".flow-path", { strokeDashoffset: 0, autoAlpha: 1 });
  gsap.set(token, { autoAlpha: 0 });

  timeline.addLabel("trajectory", 0)
    .call(() => setChapter("trajectory"), null, "trajectory")
    .set(".scene-trajectory", { autoAlpha: 1 }, "trajectory")
    .from(".scene-trajectory .scene-title > *", { y: 22, autoAlpha: 0, stagger: .08 }, "trajectory+=.1")
    .from(".trajectory-caption > *", { y: 18, autoAlpha: 0, stagger: .1 }, "trajectory+=.28")
    .from(".chart-card", { y: 34, autoAlpha: 0, stagger: .12 }, "trajectory+=1.05")
    .to(".raw-line", { strokeDashoffset: 0, duration: 2.2, ease: "power1.inOut", stagger: .08 }, "trajectory+=1.5")
    .to(".estimate-line", { strokeDashoffset: 0, duration: 2.3, ease: "power1.inOut", stagger: .13 }, "trajectory+=1.95")
    .to(".deviation-area", { autoAlpha: 1, duration: .8, stagger: .1 }, "trajectory+=3.45")
    .to(".data-dot", { scale: 1, duration: .26, stagger: { amount: 1.1, from: "start" } }, "trajectory+=2.5")
    .to(".chart-card.navi", { boxShadow: "inset 0 0 60px rgba(184,248,61,.09)", duration: .8, yoyo: true, repeat: 1 }, "trajectory+=4.8")
    .to(".scene-trajectory", { xPercent: -2, autoAlpha: 0, duration: .65, ease: "power2.in" }, "trajectory+=7.4")

    .addLabel("decision", "trajectory+=8")
    .call(() => setChapter("decision"), null, "decision")
    .set(".scene-decision", { xPercent: 0, autoAlpha: 1 }, "decision")
    .from(".scene-decision .scene-title > *", { y: 22, autoAlpha: 0, stagger: .08 }, "decision+=.1")
    .from(".decision-node", { autoAlpha: 0, stagger: .1 }, "decision+=.45")
    .from(".decision-rail-wrap", { y: 20, autoAlpha: 0 }, "decision+=.9")
    .add(buildDecisionTimeline(), "decision+=1.45")
    .to(".scene-decision", { xPercent: -2, autoAlpha: 0, duration: .65, ease: "power2.in" }, "decision+=7.25")

    .addLabel("results", "decision+=7.85")
    .call(() => setChapter("results"), null, "results")
    .set(".scene-results", { xPercent: 0, autoAlpha: 1 }, "results")
    .from(".result-copy > *", { y: 22, autoAlpha: 0, stagger: .1 }, "results+=.1")
    .from(".video-shell", { y: 34, autoAlpha: 0, scale: .985 }, "results+=.45")
    .from(".results-panel", { x: 28, autoAlpha: 0 }, "results+=.55")
    .from(".proof-grid > div", { y: 16, autoAlpha: 0, stagger: .09 }, "results+=.9")
    .to("[data-bar]", { scaleX: (index, element) => Number(element.dataset.bar) / 100, stagger: .12, duration: 1.1 }, "results+=1.05");

  animateCounter(timeline, "results+=.7");
  timeline.to(".video-shell", { boxShadow: "0 30px 95px rgba(184,248,61,.09)", duration: 1, yoyo: true, repeat: 1 }, "results+=3.6")
    .to(".scene-results", { autoAlpha: 0, duration: .7, ease: "power2.in" }, "results+=7.4");

  return timeline;
}

function showSceneInstant(name) {
  setChapter(name);
  gsap.set(scenes, { autoAlpha: 0 });
  gsap.set(`.scene-${name}`, { autoAlpha: 1, xPercent: 0 });
  gsap.set(".raw-line, .estimate-line", { strokeDashoffset: 0 });
  gsap.set(".deviation-area, .data-dot", { autoAlpha: 1, scale: 1 });
  gsap.set("[data-bar]", { scaleX: (index, element) => Number(element.dataset.bar) / 100 });
  gsap.set(".flow-path", { strokeDashoffset: 0 });
  if (name === "decision") setDecision(decisions.length - 1);
}

function wireControls(reduceMotion) {
  navButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.goto;
      if (reduceMotion || !master) showSceneInstant(target);
      else master.seek(target).play();
    });
  });

  motionToggle.addEventListener("click", () => {
    manualPaused = !manualPaused;
    motionToggle.setAttribute("aria-pressed", String(manualPaused));
    motionLabel.textContent = manualPaused ? "PLAY" : "PAUSE";
    if (master) master.paused(manualPaused);
    if (manualPaused) pauseComparisonVideos();
    else if (activeScene === "results") playComparisonVideos();
  });

  document.addEventListener("visibilitychange", () => {
    if (!master) return;
    if (document.hidden) {
      master.pause();
      pauseComparisonVideos();
    } else if (!manualPaused) {
      master.play();
      if (activeScene === "results") playComparisonVideos();
    }
  });
}

async function init() {
  if (!window.gsap) {
    dataStatus.textContent = "GSAP FAILED TO LOAD";
    return;
  }

  const data = await loadTrajectoryData();
  const series = buildPredictions(data.raw, data.all);
  renderCharts(series);
  decisions = buildDecisions(series.raw, series.navi);
  renderDecisionRail();
  updateConnectorPaths();
  window.addEventListener("resize", updateConnectorPaths, { passive: true });

  const media = gsap.matchMedia();
  media.add({
    reduceMotion: "(prefers-reduced-motion: reduce)",
    standardMotion: "(prefers-reduced-motion: no-preference)"
  }, (context) => {
    if (context.conditions.reduceMotion) {
      showSceneInstant("trajectory");
      wireControls(true);
      return () => pauseComparisonVideos();
    }
    master = buildMasterTimeline();
    wireControls(false);
    gsap.to(".ambient", { xPercent: -8, yPercent: 6, duration: 6, repeat: -1, yoyo: true, ease: "sine.inOut" });
    gsap.to(".grain", { xPercent: 2, yPercent: -2, duration: .22, repeat: -1, yoyo: true, ease: "steps(2)" });
    return () => {
      master?.kill();
      pauseComparisonVideos();
    };
  });
}

init();
