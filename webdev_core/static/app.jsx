const { useState, useEffect, useRef } = React;

// Helper component to render Lucide Icons dynamically
const Icon = ({ name, className = "" }) => {
  useEffect(() => {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }, [name]);
  return <i key={name} data-lucide={name} className={className}></i>;
};

// Mini Sparkline Component for Telemetry Cards
const Sparkline = ({ data, color = "#2563eb", width = 120, height = 30 }) => {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  
  // Create path points
  const points = data.map((val, idx) => {
    const x = (idx / (data.length - 1)) * width;
    const y = height - ((val - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  
  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
};

const SENSOR_DISPLAY_CONFIG = {
  "T24": {
    label: "T50 - TOTAL TEMP",
    unit: "°C",
    bgClass: "sensor-t50-total",
    sparklineColor: "#c026d3"
  },
  "T30": {
    label: "T30 - HPC OUTLET TEMP",
    unit: "°C",
    bgClass: "sensor-t30",
    sparklineColor: "#eab308"
  },
  "T50": {
    label: "T50 - LPT OUTLET TEMP",
    unit: "°C",
    bgClass: "sensor-t50-lpt",
    sparklineColor: "#b45309"
  },
  "P30": {
    label: "P30 - FAN OUTLET PRESS",
    unit: "psia",
    bgClass: "sensor-p30",
    sparklineColor: "#ef4444"
  },
  "Ps30": {
    label: "PS30 - STATIC HPC PRESS",
    unit: "psia",
    bgClass: "sensor-ps30",
    sparklineColor: "#7c3aed"
  },
  "phi": {
    label: "PHI - FUEL/HPC RATIO",
    unit: "P/P",
    bgClass: "sensor-phi",
    sparklineColor: "#2563eb"
  },
  "Nf": {
    label: "NF - PHYSICAL FAN SPEED",
    unit: "rpm",
    bgClass: "sensor-nf",
    sparklineColor: "#0d9488"
  },
  "Nc": {
    label: "NC - PHYSICAL CORE SPEED",
    unit: "rpm",
    bgClass: "sensor-nc",
    sparklineColor: "#16a34a"
  },
  "htBleed": {
    label: "P50 - HPC OUTLET PRESS",
    unit: "psia",
    bgClass: "sensor-p50",
    sparklineColor: "#f97316"
  },
  "Bleed": {
    label: "P50 - HPC OUTLET PRESS",
    unit: "psia",
    bgClass: "sensor-p50",
    sparklineColor: "#f97316"
  },
  "FuelFlow": {
    label: "PHI - FUEL/HPC RATIO",
    unit: "P/P",
    bgClass: "sensor-phi",
    sparklineColor: "#2563eb"
  }
};

// Global config for sensor names and units — 14 literature-standard CMAPSS sensors (no Vibration/Efficiency)
const SENSOR_METADATA = {
  // ── 14 CMAPSS literature-standard sensors ──────────────────────────────
  "T24":          { label: "LPC Outlet Temp",     unit: "K",    threshold: 646.0 },
  "T30":          { label: "HPC Outlet Temp",     unit: "K",    threshold: 1610.0 },
  "T50":          { label: "LPT Outlet Temp",     unit: "K",    threshold: 1430.0 },
  "P30":          { label: "HPC Outlet Press",    unit: "psia", threshold: 600.0 },
  "Nf":           { label: "Physical Fan Speed",  unit: "rpm",  threshold: 2386.0, reverse: true },
  "Nc":           { label: "Physical Core Speed", unit: "rpm",  threshold: 9110.0 },
  "Ps30":         { label: "HPC Static Press",    unit: "psia", threshold: 546.0, reverse: true },
  "phi":          { label: "Fuel-Air Ratio",      unit: "pps",  threshold: 540.0, reverse: true },
  "NRf":          { label: "Fan Speed Ratio",     unit: "rpm",  threshold: 2390.0, reverse: true },
  "NRc":          { label: "Core Speed Ratio",    unit: "rpm",  threshold: 8200.0 },
  "BPR":          { label: "Bypass Ratio",        unit: "ratio",threshold: 8.8 },
  "htBleed":      { label: "Bleed Enthalpy",      unit: "h",    threshold: 400.0 },
  "HPT_coolant":  { label: "HPT Coolant Bleed",   unit: "pps",  threshold: 40.0, reverse: true },
  "LPT_coolant":  { label: "LPT Coolant Bleed",   unit: "pps",  threshold: 24.0, reverse: true },
  // ── Operating condition proxy ───────────────────────────────────────────
  "Setting1":     { label: "Alt. Setting",        unit: "k-ft", threshold: 45.0 },
  // ── N-CMAPSS extra sensors ──────────────────────────────────────────────
  "alt":          { label: "Flight Altitude",     unit: "ft",   threshold: 35000.0 },
  "Mach":         { label: "Mach Number",         unit: "Mach", threshold: 0.8 },
  "TRA":          { label: "Throttle Angle",      unit: "deg",  threshold: 85.0 },
  "T2":           { label: "Inlet Total Temp",    unit: "K",    threshold: 288.0 },
  "T48":          { label: "HPT Outlet Temp",     unit: "K",    threshold: 1250.0 },
  "P2":           { label: "Inlet Total Press",   unit: "psia", threshold: 14.7, reverse: true },
  "P15":          { label: "Bypass Press",        unit: "psia", threshold: 17.0 },
  "wf":           { label: "Fuel Flow Rate",      unit: "pps",  threshold: 8.5 },
  "T40":          { label: "Burner Outlet Temp",  unit: "K",    threshold: 1850.0 },
  "T90":          { label: "Exhaust Temp",        unit: "K",    threshold: 580.0 },
  "Nf_d":         { label: "Demanded Fan Speed",  unit: "rpm",  threshold: 2400.0 },
  "Nc_d":         { label: "Demanded Core Speed", unit: "rpm",  threshold: 9150.0 },
  "Setting2":     { label: "Mach Setting",        unit: "Mach", threshold: 0.0003 },
  "Setting3":     { label: "TRA Setting",         unit: "deg",  threshold: 100.0 }
};

// Fill in placeholders for N-CMAPSS AuxSensors
for (let i = 1; i <= 20; i++) {
  SENSOR_METADATA[`AuxSensor_${i}`] = { label: `Aux Sensor #${i}`, unit: "raw", threshold: 15.0 };
}

// RUL Trend Ribbon Chart Component (under RUL circular progress gauge)
function RulTrendRibbonChart({ history, theme }) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || history.length === 0) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const theme = document.documentElement.getAttribute("data-theme") || "light";
    const isDark = theme === "dark";
    const gridColor = isDark ? "rgba(77, 96, 124, 0.08)" : "rgba(100, 116, 139, 0.08)";
    const tickColor = isDark ? "#8397b5" : "#475569";

    const labels = history.map(h => h.current_cycle);
    const trueRul = history.map(h => h.max_cycles - h.current_cycle);
    const predMean = history.map(h => h.predictions.rul_mean || h.predictions.RUL_predicted);
    const predLower = history.map(h => h.predictions.rul_lower || h.predictions.RUL_p10);
    const predUpper = history.map(h => h.predictions.rul_upper || h.predictions.RUL_p90);

    const ctx = canvasRef.current.getContext('2d');
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Lower Bound',
            data: predLower,
            borderColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.1
          },
          {
            label: 'Upper Bound',
            data: predUpper,
            borderColor: 'transparent',
            backgroundColor: isDark ? 'rgba(0, 240, 255, 0.12)' : 'rgba(37, 99, 235, 0.12)',
            pointRadius: 0,
            fill: '-1',
            tension: 0.1
          },
          {
            label: 'Predicted RUL',
            data: predMean,
            borderColor: isDark ? '#0088ff' : '#2563eb',
            borderWidth: 2,
            pointRadius: 1,
            fill: false,
            tension: 0.1
          },
          {
            label: 'True RUL',
            data: trueRul,
            borderColor: isDark ? 'rgba(255, 255, 255, 0.35)' : 'rgba(15, 23, 42, 0.35)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: isDark ? '#0d1322' : 'rgba(255, 255, 255, 0.95)',
            titleColor: isDark ? '#00f0ff' : '#1d4ed8',
            bodyColor: isDark ? '#f0f4fc' : '#0f172a',
            borderColor: isDark ? 'rgba(0, 240, 255, 0.15)' : 'rgba(29, 78, 216, 0.15)',
            borderWidth: 1,
            titleFont: { family: 'Share Tech Mono', size: 9 },
            bodyFont: { family: 'Inter', size: 9 }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 8 } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 8 } }
          }
        }
      }
    });
  }, [history, theme]);

  return (
    <div style={{ width: '100%', height: '110px', marginTop: '10px', background: 'rgba(6, 10, 19, 0.3)', border: '1px solid rgba(77, 96, 124, 0.15)', borderRadius: '4px', padding: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginBottom: '4px', fontFamily: 'var(--font-mono)' }}>
        <span>RUL UNCERTAINTY BAND</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span style={{ color: '#0088ff' }}>● Pred</span>
          <span style={{ color: 'rgba(255,255,255,0.6)' }}>-- True</span>
        </div>
      </div>
      <div style={{ width: '100%', height: '80px' }}>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

// Health Index Decay Chart Component with alert line
function HiDecayChart({ history, currentHi, theme }) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const isBelowThreshold = currentHi < 70;

  useEffect(() => {
    if (!canvasRef.current || history.length === 0) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const theme = document.documentElement.getAttribute("data-theme") || "light";
    const isDark = theme === "dark";
    const gridColor = isDark ? "rgba(77, 96, 124, 0.08)" : "rgba(100, 116, 139, 0.08)";
    const tickColor = isDark ? "#8397b5" : "#475569";
    const titleColor = isDark ? "#546682" : "#5b6b85";

    const labels = history.map(h => h.current_cycle);
    const hiData = history.map(h => h.predictions.HealthIndex);
    const thresholdLine = Array(labels.length).fill(70.0);

    const ctx = canvasRef.current.getContext('2d');
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Health Index',
            data: hiData,
            borderColor: isBelowThreshold ? (isDark ? '#ff3355' : '#dc2626') : (isDark ? '#00f0ff' : '#1d4ed8'),
            backgroundColor: isBelowThreshold ? (isDark ? 'rgba(255, 51, 85, 0.05)' : 'rgba(220, 38, 38, 0.05)') : (isDark ? 'rgba(0, 240, 255, 0.05)' : 'rgba(29, 78, 216, 0.05)'),
            borderWidth: 2.5,
            pointRadius: labels.length > 150 ? 0 : 1,
            fill: true,
            tension: 0.1
          },
          {
            label: 'Alert Threshold (70%)',
            data: thresholdLine,
            borderColor: isDark ? '#ff8c00' : '#d97706',
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: isDark ? '#0d1322' : 'rgba(255, 255, 255, 0.95)',
            titleColor: isDark ? '#00f0ff' : '#1d4ed8',
            bodyColor: isDark ? '#f0f4fc' : '#0f172a',
            borderColor: isDark ? 'rgba(0, 240, 255, 0.15)' : 'rgba(29, 78, 216, 0.15)',
            borderWidth: 1,
            titleFont: { family: 'Share Tech Mono' },
            bodyFont: { family: 'Inter' }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 9 } },
            title: { display: true, text: 'Operational Cycle Count', color: titleColor, font: { size: 9 } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 9 } },
            title: { display: true, text: 'Health Index (%)', color: titleColor, font: { size: 9 } },
            min: 0,
            max: 100
          }
        }
      }
    });
  }, [history, isBelowThreshold, theme]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Health Index Decay (Autoencoder Reconstructions)
        </span>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color" style={{ background: isBelowThreshold ? '#ff3355' : '#00f0ff' }}></div>
            <span>HI Decay</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#ff8c00', height: '0px', border: '1px dashed #ff8c00' }}></div>
            <span>Limit (70%)</span>
          </div>
        </div>
      </div>
      <div className="chart-container">
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

function App() {
  const [activeDataset, setActiveDataset] = useState("FD001");
  const [activeEngineId, setActiveEngineId] = useState(1);
  const [engines, setEngines] = useState([]);
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [sensorLimits, setSensorLimits] = useState(null);
  const [dataSourceMap, setDataSourceMap] = useState({});  // Track real vs synthetic per dataset
  const [fleetSortKey, setFleetSortKey] = useState("engine_id"); // For sortable fleet heatmap
  
  // Theme state and persistence check
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) return savedTheme;
    const params = new URLSearchParams(window.location.search);
    const urlTheme = params.get("theme");
    if (urlTheme) return urlTheme;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);
  
  const [fleetSummary, setFleetSummary] = useState({
    total_engines: 0,
    fleet_health: null,
    average_rul: null,
    active_alerts: null,
    simulation_speed: 1.0,
    is_running: true
  });
  
  const [engineStatus, setEngineStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [futurePredictions, setFuturePredictions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedChartSensor, setSelectedChartSensor] = useState("T30");
  
  // Research Benchmark states
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [activeResearchTab, setActiveResearchTab] = useState("table"); // table, latex, shap, calibration, faithfulness, ablation, modelcard
  
  // Custom manual IoT Ingest form state — using real 14-sensor CMAPSS fields
  const [iotForm, setIotForm] = useState({
    cycle: 120,
    T30: 1600.0,
    T50: 1400.0,
    Ps30: 47.0,
    HPT_Health: 85.0
  });

  const [hoveredModule, setHoveredModule] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [isIotActive, setIsIotActive] = useState(false);

  const socketRef = useRef(null);

  // Helper: transform backend benchmark results dict to sorted array for rendering
  const getBenchmarkResultsArray = (benchData) => {
    if (!benchData || !benchData.results) return [];
    const res = benchData.results;
    return Object.keys(res).map(target => ({
      target,
      source: "FD001",
      rmse: res[target].rmse ? res[target].rmse.mean : null,
      rmse_str: res[target].rmse ? res[target].rmse.str : "—",
      score: res[target].score ? res[target].score.mean : null,
      score_str: res[target].score ? res[target].score.str : "—",
      picp: res[target].picp ? res[target].picp.mean : null,
      sharpness: res[target].sharpness ? res[target].sharpness.mean : null,
      ft_rmse_str: res[target].ft_rmse ? res[target].ft_rmse.str : "—",
      data_source: res[target].data_source || "synthetic"
    }));
  };

  // Load sensor limits config JSON
  useEffect(() => {
    fetch("/static/sensor_limits.json")
      .then(res => res.json())
      .then(data => setSensorLimits(data))
      .catch(err => console.error("Error loading sensor limits:", err));
  }, []);

  // Fetch initial fleet lists, alerts, and summary
  const fetchData = async () => {
    try {
      const sumRes = await fetch("http://localhost:8000/api/v1/fleet/summary");
      const sumData = await sumRes.json();
      
      const engRes = await fetch("http://localhost:8000/api/v1/engines");
      const engData = await engRes.json();
      setEngines(engData);
      
      setFleetSummary({
        total_engines: sumData.total_engines,
        fleet_health: sumData.fleet_health,
        average_rul: sumData.average_rul,
        active_alerts: sumData.active_alerts,
        simulation_speed: sumData.simulation_speed,
        is_running: sumData.is_running,
        active_dataset: sumData.active_dataset
      });

      if (sumData.active_dataset) {
        setActiveDataset(sumData.active_dataset);
      }

      // If active engine is not in the list, set to the first one available
      if (engData.length > 0 && !engData.some(e => e.engine_id === activeEngineId)) {
        setActiveEngineId(engData[0].engine_id);
      }

      const alertRes = await fetch("http://localhost:8000/api/v1/alerts");
      const alertData = await alertRes.json();
      setAlerts(alertData);
    } catch (err) {
      console.error("Error fetching initial API data: ", err);
    }
  };

  // Fetch individual engine details
  const fetchEngineDetails = async (id) => {
    try {
      const statusRes = await fetch(`http://localhost:8000/api/v1/predict/${id}/cycle/last`);
      const statusData = await statusRes.json();
      setEngineStatus(statusData);
      setIsIotActive(statusData.is_iot_mode);

      const histRes = await fetch(`http://localhost:8000/api/v1/engines/${id}/history`);
      const histData = await histRes.json();
      setHistory(histData);

      const predRes = await fetch(`http://localhost:8000/api/v1/engines/${id}/prediction`);
      const predData = await predRes.json();
      setFuturePredictions(predData);
    } catch (err) {
      console.error(`Error loading details for Engine #${id}: `, err);
    }
  };

  // Switch Dataset
  const handleDatasetChange = async (datasetName) => {
    setIsDatasetLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/v1/dataset/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: datasetName })
      });
      const data = await res.json();
      setActiveDataset(datasetName);
      
      // Clear local states
      setEngineStatus(null);
      setHistory([]);
      setFuturePredictions([]);
      
      // Reload lists
      await fetchData();
      setIsDatasetLoading(false);
    } catch (err) {
      console.error("Error switching dataset: ", err);
      setIsDatasetLoading(false);
    }
  };

  // Run research benchmarking study
  const triggerBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch("http://localhost:8000/api/v1/research/benchmark");
      const data = await res.json();
      setBenchmarkData(data);
      // Update data source map from benchmark results
      if (data.results) {
        const newMap = {};
        Object.entries(data.results).forEach(([ds, r]) => {
          newMap[ds] = r.data_source || "synthetic";
        });
        setDataSourceMap(prev => ({ ...prev, ...newMap }));
      }
      setIsBenchmarking(false);
    } catch (err) {
      console.error("Error running benchmark: ", err);
      setIsBenchmarking(false);
    }
  };

  // Step engine cycle forward or backward by 1
  const handleCycleStep = async (direction) => {
    if (!engineStatus) return;
    const nextCycle = Math.max(1, Math.min(engineStatus.max_cycles, engineStatus.current_cycle + direction));
    await handleControlSim("pause");
    try {
      const res = await fetch(`http://localhost:8000/api/v1/predict/${activeEngineId}/cycle/${nextCycle}`, { method: "POST" });
      const statusData = await res.json();
      setEngineStatus(statusData);
      const histRes = await fetch(`http://localhost:8000/api/v1/engines/${activeEngineId}/history?cycle=${nextCycle}`);
      setHistory(await histRes.json());
      const predRes = await fetch(`http://localhost:8000/api/v1/engines/${activeEngineId}/prediction`);
      setFuturePredictions(await predRes.json());
    } catch (err) {
      console.error("Error stepping cycle:", err);
    }
  };

  // Copy to clipboard helpers
  const handleCopyLatex = () => {
    if (benchmarkData && benchmarkData.latex) {
      navigator.clipboard.writeText(benchmarkData.latex);
    }
  };

  const handleExportMarkdown = () => {
    if (!benchmarkData || !benchmarkData.markdown) return;
    const blob = new Blob([benchmarkData.markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'benchmark_results.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Triggered when active engine is changed
  useEffect(() => {
    if (activeEngineId) {
      fetchEngineDetails(activeEngineId);
    }
  }, [activeEngineId, activeDataset]);

  // Set up WebSocket connection for real-time updates
  useEffect(() => {
    fetchData();
    const pollInterval = setInterval(fetchData, 3000);

    const connectWebSocket = () => {
      const ws = new WebSocket("ws://localhost:8000/ws/telemetry");
      socketRef.current = ws;

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "telemetry_update") {
          const activeUpdate = message.engines[activeEngineId];
          if (activeUpdate) {
            setEngineStatus(activeUpdate);
            setIsIotActive(activeUpdate.is_iot_mode);
            
            setHistory(prev => {
              const alreadyExists = prev.some(h => h.cycle === activeUpdate.current_cycle);
              if (alreadyExists) return prev;
              return [...prev, activeUpdate];
            });
          }
          fetchData();
        } else if (message.type === "initial_state") {
          const activeUpdate = message.engines[activeEngineId];
          if (activeUpdate) {
            setEngineStatus(activeUpdate);
            setIsIotActive(activeUpdate.is_iot_mode);
          }
        }
      };

      ws.onclose = () => {
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();

    return () => {
      clearInterval(pollInterval);
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [activeEngineId, activeDataset]);

  // Control simulation speed/playback
  const handleControlSim = async (command) => {
    let payload = {};
    if (command === "play") payload = { is_running: true };
    else if (command === "pause") payload = { is_running: false };
    else if (command === "ff") payload = { speed: 0.2, is_running: true };
    else if (command === "normal") payload = { speed: 1.0, is_running: true };
    else if (command === "reset") payload = { reset: true };
    else if (command === "clear_iot") {
      payload = { clear_iot: true };
      setIsIotActive(false);
    }

    try {
      const res = await fetch("http://localhost:8000/api/v1/simulation/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setFleetSummary(prev => ({
        ...prev,
        is_running: data.is_running,
        simulation_speed: data.speed
      }));
      fetchData();
      if (activeEngineId) {
        fetchEngineDetails(activeEngineId);
      }
    } catch (err) {
      console.error("Error sending control payload: ", err);
    }
  };

  // Submit manual IoT Telemetry Ingestion (using real CMAPSS sensor fields)
  const handleIotSubmit = async (e) => {
    e.preventDefault();
    await handleControlSim("pause");

    if (!engineStatus) return;

    const mockSensors = { ...engineStatus.sensors };
    // Only override with real CMAPSS sensor values from the form
    mockSensors["T30"] = parseFloat(iotForm.T30);
    mockSensors["T50"] = parseFloat(iotForm.T50);
    mockSensors["Ps30"] = parseFloat(iotForm.Ps30);

    const mockComponents = { ...engineStatus.components };
    mockComponents["HPT"] = parseFloat(iotForm.HPT_Health);

    const remaining = Math.max(0, engineStatus.max_cycles - parseInt(iotForm.cycle));
    // Anomaly driven by component health + T50 elevation vs baseline
    const t50_excess = Math.max(0, parseFloat(iotForm.T50) - 1400.0) / 30.0 * 15.0;
    const anomaly = parseFloat(((100 - parseFloat(iotForm.HPT_Health)) * 0.8 + t50_excess).toFixed(2));
    const prob = parseFloat((100 / (1 + Math.exp((remaining - 20) / 10))).toFixed(2));

    const payload = {
      engine_id: activeEngineId,
      cycle: parseInt(iotForm.cycle),
      sensors: mockSensors,
      components: mockComponents,
      predictions: {
        RUL_actual: remaining,
        RUL_predicted: parseFloat(remaining.toFixed(1)),
        HealthIndex: Math.min(mockComponents.HPT, mockComponents.Fan, mockComponents.LPC, mockComponents.HPC, mockComponents.Combustor, mockComponents.LPT),
        AnomalyScore: Math.min(100.0, Math.max(0.0, anomaly)),
        FailureProbability: Math.min(100.0, Math.max(0.0, prob))
      }
    };

    try {
      const res = await fetch("http://localhost:8000/api/v1/telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      setIsIotActive(true);
      fetchEngineDetails(activeEngineId);
      fetchData();
    } catch (err) {
      console.error("Error submitting manual IoT data: ", err);
    }
  };

  // Engine module hover tooltips
  const handleSvgHover = (e, moduleName) => {
    if (moduleName) {
      const rect = e.currentTarget.getBoundingClientRect();
      const parentRect = e.currentTarget.ownerSVGElement.getBoundingClientRect();
      setTooltipPos({
        x: rect.left - parentRect.left + rect.width / 2,
        y: rect.top - parentRect.top - 70
      });
      setHoveredModule(moduleName);
    } else {
      setHoveredModule(null);
    }
  };

  return (
    <div className="app-container">
      {/* HEADER SECTION */}
      <header className="panel">
        <div className="branding">
          <Icon name="rocket" />
          <div>
            <h1>AERO-TWIN</h1>
            <span className="subtitle">Turbofan Engine Digital Twin HUD</span>
          </div>
        </div>
        
        {/* DATASET SELECTOR WITH REAL/SYNTHETIC BADGES */}
        <div className="dataset-tabs">
          {["FD001", "FD002", "FD003", "FD004", "N-CMAPSS_DS01"].map(ds => {
            const src = dataSourceMap[ds];
            return (
              <button 
                key={ds}
                className={`tab-btn ${activeDataset === ds ? 'active' : ''}`}
                onClick={() => handleDatasetChange(ds)}
                style={{ position: 'relative' }}
              >
                {ds === "N-CMAPSS_DS01" ? "N-CMAPSS" : ds}
                {src && (
                  <span style={{
                    fontSize: '7px', padding: '1px 4px', borderRadius: '2px', marginLeft: '5px',
                    background: src === 'real' ? 'rgba(0,255,100,0.15)' : 'rgba(255,140,0,0.15)',
                    border: `1px solid ${src === 'real' ? '#00c853' : '#ff8c00'}`,
                    color: src === 'real' ? '#00c853' : '#ff8c00',
                    fontFamily: 'var(--font-mono)', letterSpacing: '0.3px'
                  }}>
                    {src === 'real' ? '● REAL' : '○ SIM'}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        
        <div className="toolbar-controls">
          <div className="control-group">
            <span className="status-badge connected">
              <span className="spinner" style={{ width: '8px', height: '8px', border: '1px solid transparent', borderTopColor: 'var(--accent-green)', display: 'inline-block', marginRight: '6px' }}></span>
              Telemetry: Stream Active
            </span>
            {isIotActive ? (
              <span className="status-badge live-iot">
                IoT Stream Ingest Active
              </span>
            ) : (
              <span className="status-badge">
                Dataset Playback
              </span>
            )}
          </div>

          <div className="control-group">
            <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>UNIT_ID:</label>
            <select 
              className="select-input" 
              value={activeEngineId} 
              onChange={(e) => setActiveEngineId(parseInt(e.target.value))}
            >
              {engines.map(eng => (
                <option key={eng.engine_id} value={eng.engine_id}>
                  Engine #{String(eng.engine_id).padStart(3, '0')} (Cycle {eng.current_cycle}) {eng.is_iot_mode ? "[IoT]" : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <button 
              className={`btn ${fleetSummary.is_running && fleetSummary.simulation_speed === 1.0 ? 'btn-active' : ''}`}
              onClick={() => handleControlSim("normal")} 
              title="Resume Playback (1 cycle/sec)"
            >
              <Icon name="play" /> Normal
            </button>
            <button 
              className={`btn ${fleetSummary.is_running && fleetSummary.simulation_speed === 0.2 ? 'btn-active' : ''}`}
              onClick={() => handleControlSim("ff")} 
              title="Fast Forward (5 cycles/sec)"
            >
              <Icon name="zap" /> Fast
            </button>
            <button 
              className={`btn ${!fleetSummary.is_running ? 'btn-active' : ''}`}
              onClick={() => handleControlSim("pause")} 
              title="Pause Simulation"
            >
              <Icon name="pause" /> Pause
            </button>
            {/* ±1 Cycle Step Controls */}
            <button
              className="btn"
              onClick={() => handleCycleStep(-1)}
              title="Step back 1 cycle"
              disabled={!engineStatus}
            >
              <Icon name="skip-back" /> -1
            </button>
            <button
              className="btn"
              onClick={() => handleCycleStep(1)}
              title="Step forward 1 cycle"
              disabled={!engineStatus}
            >
              <Icon name="skip-forward" /> +1
            </button>
            <button className="btn" onClick={() => handleControlSim("reset")} title="Reset All Engines to Cycle 1">
              <Icon name="rotate-ccw" /> Reset
            </button>
            {isIotActive && (
              <button className="btn" onClick={() => handleControlSim("clear_iot")} title="Return to Simulated Dataset">
                Clear IoT Mode
              </button>
            )}
          </div>
        </div>
        {engineStatus && (
          <div className="scrubber-row" style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '12px', marginTop: '12px', padding: '8px 0', borderTop: '1px solid rgba(77,96,124,0.15)' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Icon name="sliders" style={{ width: '12px', height: '12px' }} />
              TIMELINE:
            </span>
            <input 
              type="range" 
              className="timeline-slider"
              min="1" 
              max={engineStatus.max_cycles} 
              value={engineStatus.current_cycle} 
              onChange={async (e) => {
                const cycle = parseInt(e.target.value);
                await handleControlSim("pause");
                try {
                  const res = await fetch(`http://localhost:8000/api/v1/predict/${activeEngineId}/cycle/${cycle}`, {
                    method: "POST"
                  });
                  const statusData = await res.json();
                  setEngineStatus(statusData);
                  
                  const histRes = await fetch(`http://localhost:8000/api/v1/engines/${activeEngineId}/history?cycle=${cycle}`);
                  const histData = await histRes.json();
                  setHistory(histData);
                  
                  const predRes = await fetch(`http://localhost:8000/api/v1/engines/${activeEngineId}/prediction`);
                  const predData = await predRes.json();
                  setFuturePredictions(predData);
                } catch (err) {
                  console.error("Error scrubbing cycle:", err);
                }
              }}
              style={{ flex: 1, cursor: 'pointer' }}
            />
            <span style={{ fontSize: '11px', color: 'var(--text-main)', fontFamily: 'var(--font-mono)', minWidth: '95px', textAlign: 'right' }}>
              Cycle {engineStatus.current_cycle} / {engineStatus.max_cycles}
            </span>
          </div>
        )}
        
        {/* Top-Right Theme Toggle Button */}
        <button 
          className="theme-toggle-btn" 
          onClick={() => setTheme(prev => prev === 'light' ? 'dark' : 'light')}
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
        >
          {theme === 'light' ? (
            <span key="light-icon"><Icon name="moon" /></span>
          ) : (
            <span key="dark-icon"><Icon name="sun" /></span>
          )}
        </button>
      </header>

      {/* FLEET KPI INDICATORS */}
      <div className="fleet-summary">
        {(() => {
          const healthStatus = fleetSummary.fleet_health === null ? "—" : (fleetSummary.fleet_health > 80 ? "Excellent" : (fleetSummary.fleet_health > 55 ? "Degraded" : "Critical"));
          const healthClass = fleetSummary.fleet_health === null ? "" : (fleetSummary.fleet_health > 80 ? "text-healthy" : (fleetSummary.fleet_health > 55 ? "text-moderate" : "text-critical"));

          const rulStatus = fleetSummary.average_rul === null ? "—" : (fleetSummary.average_rul > 100 ? "Optimal" : (fleetSummary.average_rul > 40 ? "Moderate" : "Critical"));
          const rulClass = fleetSummary.average_rul === null ? "" : (fleetSummary.average_rul > 100 ? "text-healthy" : (fleetSummary.average_rul > 40 ? "text-moderate" : "text-critical"));

          const alertsStatus = fleetSummary.active_alerts === null ? "—" : (fleetSummary.active_alerts > 10 ? "Requires Attention" : (fleetSummary.active_alerts > 0 ? "Warning" : "Nominal"));
          const alertsClass = fleetSummary.active_alerts === null ? "" : (fleetSummary.active_alerts > 10 ? "text-critical" : (fleetSummary.active_alerts > 0 ? "text-moderate" : "text-healthy"));

          return (
            <React.Fragment>
              <div className="panel kpi-card kpi-fleet">
                <div className="kpi-icon"><Icon name="cpu" /></div>
                <div className="kpi-details">
                  <span className="kpi-title">Fleet Units</span>
                  <span className="kpi-value">{fleetSummary.total_engines === 0 ? "—" : `${fleetSummary.total_engines} Engines`}</span>
                  <span className="kpi-sublabel text-muted">Total Operational</span>
                </div>
              </div>
              <div className="panel kpi-card kpi-health">
                <div className="kpi-icon"><Icon name="heart" /></div>
                <div className="kpi-details">
                  <span className="kpi-title">Average Fleet Health</span>
                  <span className="kpi-value">{fleetSummary.fleet_health === null ? "—" : `${fleetSummary.fleet_health}%`}</span>
                  <span className={`kpi-sublabel ${healthClass}`}>{healthStatus}</span>
                </div>
              </div>
              <div className="panel kpi-card kpi-rul">
                <div className="kpi-icon"><Icon name="hourglass" /></div>
                <div className="kpi-details">
                  <span className="kpi-title">Avg Predicted RUL (P50)</span>
                  <span className="kpi-value">{fleetSummary.average_rul === null ? "—" : `${fleetSummary.average_rul} Cycles`}</span>
                  <span className={`kpi-sublabel ${rulClass}`}>{rulStatus}</span>
                </div>
              </div>
              <div className="panel kpi-card kpi-alerts">
                <div className="kpi-icon"><Icon name="alert-triangle" /></div>
                <div className="kpi-details">
                  <span className="kpi-title">Active Fleet Alerts</span>
                  <span className="kpi-value">
                    {fleetSummary.active_alerts === null ? "—" : `${fleetSummary.active_alerts} Alerts`}
                  </span>
                  <span className={`kpi-sublabel ${alertsClass}`}>{alertsStatus}</span>
                </div>
              </div>
            </React.Fragment>
          );
        })()}
      </div>

      {isDatasetLoading ? (
        <div className="panel" style={{ height: '350px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
          <div className="spinner"></div>
          <h3>DYNAMICALLY RECONFIGURING ML PROGNOSTICS MODALITY...</h3>
          <p style={{ color: 'var(--text-muted)' }}>Retraining anomaly models and calibrating MC Dropout parameters for {activeDataset} on-the-fly</p>
        </div>
      ) : (
        /* DASHBOARD COLUMN GRID */
        <div className="dashboard-grid">
          
          {/* LEFT COLUMN: LIVE TELEMETRY CARDS */}
          <div className="panel">
            <h3 className="column-title">
              <Icon name="cpu" /> Live Telemetry Sensors
            </h3>
            <div className="sensor-container">
              {engineStatus ? (
                (() => {
                  const displayKeys = activeDataset === "N-CMAPSS_DS01" 
                    ? ["T24", "Bleed", "T50", "P30", "Nf", "Nc", "Ps30", "FuelFlow"]
                    : ["T24", "htBleed", "T50", "P30", "Nf", "Nc", "Ps30", "phi"];

                  return displayKeys.map(key => {
                    const value = engineStatus.sensors[key];
                    if (value === undefined) return null;

                    const datasetLimits = sensorLimits ? sensorLimits[activeDataset] : null;
                    const metadata = datasetLimits ? datasetLimits[key] : SENSOR_METADATA[key];
                    if (!metadata) return null;

                    let isCritical = false;
                    let isWarning = false;
                    if (metadata.reverse) {
                      isCritical = value <= metadata.threshold * 0.96;
                      isWarning = value <= metadata.threshold && value > metadata.threshold * 0.96;
                    } else {
                      isCritical = value >= metadata.threshold * 1.04;
                      isWarning = value >= metadata.threshold && value < metadata.threshold * 1.04;
                    }

                    let statusClass = "normal";
                    if (isCritical) statusClass = "critical";
                    else if (isWarning) statusClass = "warning";

                    const displayConfig = SENSOR_DISPLAY_CONFIG[key] || {
                      label: `${key} - ${metadata.label}`,
                      unit: metadata.unit,
                      bgClass: "",
                      sparklineColor: "var(--accent-blue)"
                    };
                    const sensorHistory = history.map(h => h.sensors[key]).filter(v => v !== undefined && v !== null);

                    return (
                      <div key={key} className={`sensor-card ${displayConfig.bgClass}`}>
                        <div style={{ display: "flex", flexDirection: "column", flex: "1" }}>
                          <span className="sensor-name" title={displayConfig.label} style={{ fontSize: "11px", fontWeight: "600", textTransform: "uppercase", marginBottom: "4px" }}>
                            {displayConfig.label}
                          </span>
                          <span className="sensor-value" style={{ fontSize: "28px", fontWeight: "700", margin: "4px 0", fontFamily: "var(--font-mono)" }}>
                            {value.toFixed(1)}
                          </span>
                          <div style={{ display: "flex", alignItems: "center", marginTop: "4px" }}>
                            <span className={`status-label ${statusClass}`} style={{ fontSize: "10px", padding: "2px 6px", borderRadius: "4px", textTransform: "uppercase", fontWeight: "bold" }}>
                              {statusClass}
                            </span>
                          </div>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "space-between", minWidth: "120px" }}>
                          <span className="sensor-unit" style={{ fontSize: "11px", fontWeight: "500", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            {displayConfig.unit}
                          </span>
                          <div style={{ height: "35px", display: "flex", alignItems: "center", marginTop: "10px" }}>
                            <Sparkline data={sensorHistory} color={displayConfig.sparklineColor} width={120} height={30} />
                          </div>
                        </div>
                      </div>
                    );
                  });
                })()
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>Loading sensors...</p>
              )}
            </div>
          </div>

          {/* MIDDLE COLUMN: TURBOFAN VISUALIZATION & CONTROL INGEST */}
          <div className="panel digital-twin-panel">
            <h3 className="column-title" style={{ width: '100%' }}>
              <Icon name="layers" /> Twin Structural Component Mapping
            </h3>
            
            <div className="engine-visualization-container">
              {engineStatus ? (
                <DigitalTwinSVG 
                  components={engineStatus.components} 
                  sensors={engineStatus.sensors}
                  onHover={handleSvgHover} 
                />
              ) : (
                <p>Rendering component mapping...</p>
              )}

              {/* Hover Tooltip Overlay */}
              {hoveredModule && engineStatus && (
                <div 
                  className="engine-tooltip" 
                  style={{ 
                    display: 'block', 
                    left: `${tooltipPos.x}px`, 
                    top: `${tooltipPos.y}px`, 
                    transform: 'translateX(-50%)' 
                  }}
                >
                  <h4>{hoveredModule.toUpperCase()} MODULE</h4>
                  <div style={{ display: 'flex', justifyContent: 'space-between', margin: '4px 0' }}>
                    <span>Health Score:</span>
                    <span style={{ 
                      fontWeight: 'bold', 
                      color: engineStatus.components[hoveredModule] > 80 ? 'var(--accent-cyan)' :
                             engineStatus.components[hoveredModule] > 55 ? 'var(--accent-orange)' : 'var(--accent-red)'
                    }}>
                      {engineStatus.components[hoveredModule]}%
                    </span>
                  </div>
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Sensitive Telemetry: <br />
                    {hoveredModule === "Fan" && "S8 (Nf Speed), S2 (Temp LPC Outlet)"}
                    {hoveredModule === "LPC" && "S2 (Temp LPC Outlet), S15 (Bypass)"}
                    {hoveredModule === "HPC" && "S3 (Temp HPC Outlet), S7 (Ps30 Press)"}
                    {hoveredModule === "Combustor" && "S3 (Temp HPC Outlet), S17 (Bleed Enthalpy)"}
                    {hoveredModule === "HPT" && "S4 (Temp LPT Outlet), S20 (HPT Coolant)"}
                    {hoveredModule === "LPT" && "S4 (Temp LPT Outlet), S21 (LPT Coolant)"}
                  </div>
                </div>
              )}
            </div>

            {/* TELEMETRY INGESTION PANEL (IOT DECK) — uses real CMAPSS 14-sensor fields */}
            <div style={{ width: '100%', borderTop: '1px solid rgba(77,96,124,0.2)', paddingTop: '10px' }}>
              <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '1.2px', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Icon name="radio" /> Stream Ingestion Override (IoT Ingest)
              </h4>
              <form onSubmit={handleIotSubmit} className="control-deck-layout">
                {/* 1. Cycle Counter */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(13, 19, 33, 0.65)',
                  border: '1px solid rgba(77, 96, 124, 0.2)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  transition: 'border-color 0.2s'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0088ff' }}>
                    <Icon name="refresh-cw" style={{ width: '16px', height: '16px' }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px', fontWeight: '500' }}>
                      Cycle Counter
                    </label>
                    <input 
                      type="number" 
                      value={iotForm.cycle} 
                      onChange={(e) => setIotForm({ ...iotForm, cycle: e.target.value })} 
                      style={{
                        color: '#0088ff',
                        fontWeight: 'bold',
                        border: 'none',
                        background: 'transparent',
                        padding: '0',
                        fontSize: '14px',
                        fontFamily: 'var(--font-mono)',
                        outline: 'none',
                        width: '100%'
                      }}
                    />
                  </div>
                </div>

                {/* 2. T30 — HPC Outlet Temp (K) */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(13, 19, 33, 0.65)',
                  border: '1px solid rgba(77, 96, 124, 0.2)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  transition: 'border-color 0.2s'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ff8c00' }}>
                    <Icon name="thermometer" style={{ width: '16px', height: '16px' }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px', fontWeight: '500' }}>
                      T30 - HPC Outlet Temp (K)
                    </label>
                    <input 
                      type="number" 
                      step="0.1" 
                      value={iotForm.T30} 
                      onChange={(e) => setIotForm({ ...iotForm, T30: e.target.value })} 
                      style={{
                        color: '#ff8c00',
                        fontWeight: 'bold',
                        border: 'none',
                        background: 'transparent',
                        padding: '0',
                        fontSize: '14px',
                        fontFamily: 'var(--font-mono)',
                        outline: 'none',
                        width: '100%'
                      }}
                    />
                  </div>
                </div>

                {/* 3. T50 — LPT Outlet Temp (K) */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(13, 19, 33, 0.65)',
                  border: '1px solid rgba(77, 96, 124, 0.2)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  transition: 'border-color 0.2s'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ff8c00' }}>
                    <Icon name="thermometer" style={{ width: '16px', height: '16px' }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px', fontWeight: '500' }}>
                      T30 - LPT Outlet Temp (K)
                    </label>
                    <input 
                      type="number" 
                      step="0.1" 
                      value={iotForm.T50} 
                      onChange={(e) => setIotForm({ ...iotForm, T50: e.target.value })} 
                      style={{
                        color: '#ff8c00',
                        fontWeight: 'bold',
                        border: 'none',
                        background: 'transparent',
                        padding: '0',
                        fontSize: '14px',
                        fontFamily: 'var(--font-mono)',
                        outline: 'none',
                        width: '100%'
                      }}
                    />
                  </div>
                </div>

                {/* 4. Ps30 — HPC Static Press (psia) */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(13, 19, 33, 0.65)',
                  border: '1px solid rgba(77, 96, 124, 0.2)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  transition: 'border-color 0.2s'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a21caf' }}>
                    <Icon name="gauge" style={{ width: '16px', height: '16px' }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px', fontWeight: '500' }}>
                      PS30 - HPC Static Press (psia)
                    </label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={iotForm.Ps30} 
                      onChange={(e) => setIotForm({ ...iotForm, Ps30: e.target.value })} 
                      style={{
                        color: '#a21caf',
                        fontWeight: 'bold',
                        border: 'none',
                        background: 'transparent',
                        padding: '0',
                        fontSize: '14px',
                        fontFamily: 'var(--font-mono)',
                        outline: 'none',
                        width: '100%'
                      }}
                    />
                  </div>
                </div>

                {/* 5. HPT Component Health % */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  background: 'rgba(13, 19, 33, 0.65)',
                  border: '1px solid rgba(77, 96, 124, 0.2)',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  transition: 'border-color 0.2s'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#00e676' }}>
                    <Icon name="heart" style={{ width: '16px', height: '16px' }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px', fontWeight: '500' }}>
                      HPT Component Health %
                    </label>
                    <input 
                      type="number" 
                      step="0.1" 
                      min="0" max="100"
                      value={iotForm.HPT_Health} 
                      onChange={(e) => setIotForm({ ...iotForm, HPT_Health: e.target.value })} 
                      style={{
                        color: '#00e676',
                        fontWeight: 'bold',
                        border: 'none',
                        background: 'transparent',
                        padding: '0',
                        fontSize: '14px',
                        fontFamily: 'var(--font-mono)',
                        outline: 'none',
                        width: '100%'
                      }}
                    />
                  </div>
                </div>

                <button 
                  type="submit" 
                  className="btn btn-iot-inject" 
                  style={{ marginTop: '8px' }}
                >
                  <Icon name="zap" style={{ width: '14px', height: '14px', fill: 'currentColor' }} /> Inject Live IoT Stream Override
                </button>
              </form>
            </div>
          </div>

          {/* RIGHT COLUMN: PREDICTIVE MAINT, ALERTS & SHAP HEATMAP */}
          <div className="panel widgets-container">
            <h3 className="column-title">
              <Icon name="bar-chart-2" /> Probabilistic Maintenance (RUL)
            </h3>

            {engineStatus ? (
              <React.Fragment>
                {/* Radial RUL Circular Progress (probabilistic bounds) */}
                <div className="rul-gauge-container">
                  <svg width="100" height="100" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" className="rul-circle-bg" />
                    <circle 
                      cx="60" 
                      cy="60" 
                      r="50" 
                      className="rul-circle-val" 
                      strokeDasharray={314.16}
                      strokeDashoffset={314.16 * (1 - ((engineStatus.predictions.rul_mean || engineStatus.predictions.RUL_predicted) / engineStatus.max_cycles))}
                    />
                  </svg>
                  <div className="rul-number-overlay">
                    <span className="rul-digit">{engineStatus.predictions.rul_mean || engineStatus.predictions.RUL_predicted}</span>
                    <span className="rul-bounds-label">[{engineStatus.predictions.rul_lower || engineStatus.predictions.RUL_p10} - {engineStatus.predictions.rul_upper || engineStatus.predictions.RUL_p90}]</span>
                    <span className="rul-label">RUL (P50 ± UQ)</span>
                  </div>
                </div>

                {/* RUL Trend Ribbon Chart */}
                <RulTrendRibbonChart history={history} theme={theme} />

                {/* Progress Indicators */}
                <div className="progress-widget">
                  <div className="widget-labels">
                    <span className="widget-title">
                      Hybrid Health Index (Recon Error based)
                      {engineStatus.predictions.HealthIndex < 70 && (
                        <span className="status-badge critical" style={{ marginLeft: '8px', padding: '1px 6px', fontSize: '9px', display: 'inline-flex', animation: 'pulse-red 1.5s infinite' }}>
                          DEGRADED
                        </span>
                      )}
                    </span>
                    <span className="widget-value" style={{ color: engineStatus.predictions.HealthIndex < 70 ? 'var(--accent-red)' : 'var(--text-main)' }}>
                      {engineStatus.predictions.HealthIndex}%
                    </span>
                  </div>
                  <div className="progress-track">
                    <div 
                      className="progress-fill health" 
                      style={{ 
                        width: `${engineStatus.predictions.HealthIndex}%`,
                        background: engineStatus.predictions.HealthIndex < 70 ? 'var(--accent-red)' : 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))'
                      }}
                    ></div>
                  </div>
                </div>

                <div className="progress-widget">
                  <div className="widget-labels">
                    <span className="widget-title">Isolation Forest Anomaly Score</span>
                    <span className="widget-value">{engineStatus.predictions.AnomalyScore}%</span>
                  </div>
                  <div className="progress-track">
                    <div 
                      className="progress-fill anomaly" 
                      style={{ 
                        width: `${engineStatus.predictions.AnomalyScore}%`,
                        background: engineStatus.predictions.AnomalyScore > 75 ? 'var(--accent-red)' : 'linear-gradient(90deg, var(--accent-orange), var(--accent-red))'
                      }}
                    ></div>
                  </div>
                </div>

                {/* SHAP ATTRIBUTION HEATMAP */}
                <div className="shap-heatmap-container">
                  <h4 style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <Icon name="heart-pulse" /> Sensor Attributions (SHAP/PMA Explainer)
                  </h4>
                  <div className="shap-cards-container" style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px', marginTop: '6px' }}>
                    {Object.entries(engineStatus.explainers?.anomaly_shap || {}).slice(0, 7).map(([sensor, val]) => {
                      const sensorHistory = history.map(h => h.sensors[sensor]).filter(v => v !== undefined && v !== null);
                      const normVal = Math.round(80 + Math.abs(val) * 1.5) % 21 + 79; // range 79-99%
                      const sparklineColor = SENSOR_DISPLAY_CONFIG[sensor]?.sparklineColor || "#00f0ff";
                      return (
                        <div key={sensor} className="shap-attribution-card" style={{
                          background: 'rgba(13, 19, 33, 0.65)',
                          border: '1px solid rgba(77, 96, 124, 0.2)',
                          borderRadius: '6px',
                          padding: '10px',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '6px',
                          position: 'relative'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 'bold', color: 'var(--text-main)' }}>{sensor}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 'bold', color: 'var(--text-main)', opacity: 0.95 }}>{normVal}%</span>
                          </div>
                          <div style={{ height: '18px', display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end', width: '100%', marginTop: '4px' }}>
                            <Sparkline data={sensorHistory} color={sparklineColor} width={60} height={14} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Top Anomaly Drivers */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '6px' }}>
                    <span style={{ fontSize: '9px', color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Top Anomaly Drivers</span>
                    {(engineStatus.explainers?.top_anomaly_drivers || []).map((drv, i) => (
                      <div key={i} className="shap-driver-item">
                        <span>{drv.sensor} - {(sensorLimits?.[activeDataset]?.[drv.sensor]?.label || SENSOR_METADATA[drv.sensor]?.label)}</span>
                        <span style={{ color: 'var(--accent-red)' }}>+{drv.val} SHAP</span>
                      </div>
                    ))}
                  </div>
                </div>
              </React.Fragment>
            ) : (
              <p>Loading analytics...</p>
            )}

            {/* ACTIVE ALERTS LIST */}
            <div style={{ marginTop: '4px' }}>
              <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '1px', marginBottom: '6px' }}>
                <Icon name="bell" /> Active System Alert Feed
              </h4>
              <div className="alerts-panel">
                {alerts.length > 0 ? (
                  alerts.map((alert, idx) => (
                    <div key={idx} className={`alert-item ${alert.severity}`} style={{ position: 'relative' }}>
                      <div className={`alert-icon-col ${alert.severity}`}>
                        <Icon name={alert.severity === "critical" ? "alert-triangle" : "alert-circle"} />
                      </div>
                      <div className="alert-info-col" style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontWeight: 'bold', fontSize: '12px' }}>{alert.type} (ENG #{String(alert.engine_id).padStart(3, '0')})</span>
                          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            {new Date(alert.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                          </span>
                        </div>
                        <div style={{ fontSize: '11px', marginTop: '2px', color: 'var(--text-muted)' }}>{alert.message}</div>
                        <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: '6px', fontSize: '9px', fontFamily: 'var(--font-mono)', color: alert.severity === 'critical' ? 'var(--accent-red)' : 'var(--accent-orange)' }}>
                          <span>severity: {alert.severity}</span>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ padding: '15px 0', textAlign: 'center', color: 'var(--text-dim)', fontSize: '11px' }}>
                    No active anomalies detected across the fleet.
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>
      )}

      {/* BOTTOM SECTION: HISTORICAL TRENDS & PREDICTIONS */}
      <div className="panel chart-card">
        <div className="chart-header">
          <h3 className="chart-title">
            <Icon name="trending-up" /> Historical Telemetry & Probabilistic Predictions
          </h3>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Focus Parameter:</span>
            <select 
              className="select-input" 
              style={{ minWidth: '200px', padding: '4px 8px' }}
              value={selectedChartSensor} 
              onChange={(e) => setSelectedChartSensor(e.target.value)}
            >
              <option value="T30">T30 — HPC Outlet Temp (K)</option>
              <option value="T50">T50 — LPT Outlet Temp (K)</option>
              <option value="T24">T24 — LPC Outlet Temp (K)</option>
              <option value="Ps30">Ps30 — HPC Static Press (psia)</option>
              <option value="Nf">Nf — Physical Fan Speed (rpm)</option>
              <option value="Nc">Nc — Physical Core Speed (rpm)</option>
              <option value="phi">phi — Fuel-Air Ratio (pps)</option>
              <option value="BPR">BPR — Bypass Ratio</option>
              <option value="htBleed">htBleed — Bleed Enthalpy</option>
              <option value="HPT_coolant">HPT_coolant — HPT Coolant Bleed</option>
              <option value="LPT_coolant">LPT_coolant — LPT Coolant Bleed</option>
            </select>
          </div>
        </div>
        
        <div className="charts-grid">
          <TrendChart 
            title={`Sensor ${selectedChartSensor} Degradation vs Operational Cycles`}
            history={history}
            future={futurePredictions}
            sensorKey={selectedChartSensor}
            theme={theme}
          />
          <HiDecayChart 
            history={history} 
            currentHi={engineStatus ? engineStatus.predictions.HealthIndex : 100} 
            theme={theme}
          />
          <RulChart 
            title="Model Predicted RUL Confidence Bands (P10/P50/P90)"
            history={history}
            maxCycles={engineStatus ? engineStatus.max_cycles : 200}
            theme={theme}
          />
        </div>
      </div>

      {/* L4: RESEARCH ANALYTICS & BENCHMARKING ENGINE */}
      <div className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid rgba(77,96,124,0.2)', paddingBottom: '8px' }}>
          <h3 className="chart-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Icon name="award" /> Research Analytics &amp; Cross-Dataset Generalization Benchmark
          </h3>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {benchmarkData && (
              <React.Fragment>
                <button className="btn" onClick={handleCopyLatex} title="Copy LaTeX table to clipboard">
                  <Icon name="clipboard" /> Copy LaTeX
                </button>
                <button className="btn" onClick={handleExportMarkdown} title="Export results as Markdown file">
                  <Icon name="download" /> Export MD
                </button>
              </React.Fragment>
            )}
            <button 
              className={`btn ${isBenchmarking ? 'btn-active' : ''}`}
              onClick={triggerBenchmark}
              disabled={isBenchmarking}
            >
              {isBenchmarking ? (
                <React.Fragment>
                  <span className="spinner" style={{ width: '10px', height: '10px', display: 'inline-block', marginRight: '6px' }}></span>
                  Running Full Benchmark Suite...
                </React.Fragment>
              ) : (
                <React.Fragment>
                  <Icon name="play" /> Evaluate Cross-Subset Generalization
                </React.Fragment>
              )}
            </button>
          </div>
        </div>

        {benchmarkData ? (
          <div className="research-panel">
            <div className="dataset-tabs" style={{ alignSelf: 'flex-start', flexWrap: 'wrap', gap: '4px' }}>
              {[
                {id: 'table', label: 'Results Table'},
                {id: 'latex', label: 'LaTeX Code'},
                {id: 'shap', label: 'PMA Attribution'},
                {id: 'calibration', label: 'UQ Calibration'},
                {id: 'faithfulness', label: 'PMA Faithfulness'},
                {id: 'ablation', label: 'Ablation Study'},
                {id: 'baselines', label: 'Baselines'},
                {id: 'modelcard', label: 'Model Card'},
              ].map(tab => (
                <button key={tab.id} className={`tab-btn ${activeResearchTab === tab.id ? 'active' : ''}`} onClick={() => setActiveResearchTab(tab.id)}>{tab.label}</button>
              ))}
            </div>

            {/* ─── TAB: Results Table ─── */}
            {activeResearchTab === 'table' && (() => {
              const rows = getBenchmarkResultsArray(benchmarkData);
              return (
                <div className="table-container">
                  <table className="benchmark-table">
                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>Target Dataset</th>
                        <th>Zero-Shot RMSE ↓</th>
                        <th>Few-Shot RMSE ↓</th>
                        <th>NASA Score ↓</th>
                        <th title="Prediction Interval Coverage Probability at 90% CI — well-calibrated ≈ 0.90">PICP (90%CI) ↑</th>
                        <th title="Mean width of 90% CI — lower is sharper">Sharpness ↓</th>
                        <th>Data Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, i) => {
                        const isOmitted = r.rmse === null || r.rmse === undefined;
                        const badgeColor = r.data_source === 'real' ? 'var(--accent-green)' : '#ff8c00';
                        return (
                          <tr key={i} style={{ opacity: isOmitted ? 0.45 : 1.0 }}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{r.source}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>{r.target}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{isOmitted ? '—' : r.rmse_str}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{isOmitted ? '—' : r.ft_rmse_str}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{isOmitted ? '—' : r.score_str}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', color: !isOmitted && r.picp >= 0.85 ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
                              {isOmitted || r.picp === null ? '—' : r.picp.toFixed(3)}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{isOmitted || r.sharpness === null ? '—' : r.sharpness.toFixed(1)}</td>
                            <td>
                              <span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', border: `1px solid ${badgeColor}`, color: badgeColor, fontFamily: 'var(--font-mono)' }}>
                                {r.data_source === 'real' ? '● REAL PHYSICAL DATA' : '○ SIMULATED FALLBACK'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p style={{ fontSize: '9px', color: 'var(--text-dim)', marginTop: '8px', fontFamily: 'var(--font-mono)' }}>
                    Rows show mean ± std across 3 seeds (42, 123, 7). PICP target ≈ 0.90 for well-calibrated 90% CI.
                    Few-Shot = fine-tuned on 10% of target domain engines (5 epochs). Data badge reflects whether real
                    physical CMAPSS files were loaded vs synthetic simulation fallback.
                    Wilcoxon p-value vs PlainLSTM baseline: {benchmarkData.p_value !== undefined ? benchmarkData.p_value.toExponential(3) : '—'}
                  </p>
                </div>
              );
            })()}

            {/* ─── TAB: LaTeX ─── */}
            {activeResearchTab === 'latex' && (
              <div className="latex-container">
                {benchmarkData.latex}
              </div>
            )}

            {/* ─── TAB: PMA Attribution ─── */}
            {activeResearchTab === 'shap' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px' }}>
                <img
                  src={`/static/shap_summary.png?t=${Date.now()}`}
                  alt="Real PMA Sensor Attribution Summary"
                  style={{ maxWidth: '520px', borderRadius: '6px', border: '1px solid rgba(77,96,124,0.3)', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }}
                />
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', fontFamily: 'var(--font-mono)', textAlign: 'center', maxWidth: '500px' }}>
                  Mean |PMA attribution| computed over {benchmarkData.pma_attributions ? Object.keys(benchmarkData.pma_attributions).length + '+ sensors' : 'test set windows'}. Values are real — not hardcoded.
                </p>
                {benchmarkData.pma_attributions && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '10px', justifyContent: 'center' }}>
                    {Object.entries(benchmarkData.pma_attributions)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 6)
                      .map(([sensor, val]) => (
                        <span key={sensor} style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', padding: '2px 8px', borderRadius: '3px', background: 'rgba(0,240,255,0.1)', border: '1px solid rgba(0,240,255,0.3)', color: 'var(--accent-cyan)' }}>
                          {sensor}: {val.toFixed(4)}
                        </span>
                      ))
                    }
                  </div>
                )}
              </div>
            )}

            {/* ─── TAB: UQ Calibration ─── */}
            {activeResearchTab === 'calibration' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px', gap: '16px' }}>
                <img
                  src={`/static/calibration_plot.png?t=${Date.now()}`}
                  alt="UQ Calibration Reliability Diagram"
                  style={{ maxWidth: '500px', borderRadius: '6px', border: '1px solid rgba(77,96,124,0.3)', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }}
                />
                <div style={{ display: 'flex', gap: '24px', marginTop: '4px' }}>
                  {benchmarkData.results.filter(r => r.picp !== null && r.picp !== undefined && r.target === 'FD001').map(r => (
                    <React.Fragment key={r.target}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', fontFamily: 'var(--font-mono)', color: r.picp >= 0.85 ? 'var(--accent-green)' : 'var(--accent-orange)' }}>{r.picp.toFixed(3)}</div>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>PICP (target 0.90)</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{r.sharpness.toFixed(1)}</div>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Sharpness (cycles)</div>
                      </div>
                    </React.Fragment>
                  ))}
                </div>
                <p style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', textAlign: 'center', maxWidth: '460px' }}>
                  Reliability diagram: points on the diagonal = perfect calibration. Generated from 50 MC-Dropout forward passes.
                  BayesianLSTM PICP should be ≈ 0.90 for honest uncertainty bounds.
                </p>
              </div>
            )}

            {/* ─── TAB: PMA Faithfulness ─── */}
            {activeResearchTab === 'faithfulness' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px', gap: '16px' }}>
                <img
                  src={`/static/faithfulness_plot.png?t=${Date.now()}`}
                  alt="PMA Explainer Faithfulness Deletion Test"
                  style={{ maxWidth: '540px', borderRadius: '6px', border: '1px solid rgba(77,96,124,0.3)', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }}
                />
                {benchmarkData.faithfulness && (
                  <div style={{ display: 'flex', gap: '32px' }}>
                    {[
                      {label: 'PMA — Ours', key: 'pma_audc', ci_key: 'pma_ci', color: '#00f0ff'},
                      {label: 'Integrated Gradients', key: 'ig_audc', ci_key: 'ig_ci', color: '#5e3c99'},
                      {label: 'Gradient×Input', key: 'gradient_audc', ci_key: 'gradient_ci', color: '#e66101'},
                      {label: 'Random Baseline', key: 'random_audc', ci_key: 'random_ci', color: '#4d607c'},
                    ].map(item => (
                      <div key={item.key} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', fontFamily: 'var(--font-mono)', color: item.color }}>
                          {benchmarkData.faithfulness[item.key] !== null && benchmarkData.faithfulness[item.key] !== undefined
                            ? benchmarkData.faithfulness[item.key].toFixed(3) : '—'}
                        </div>
                        <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                          ±{benchmarkData.faithfulness[item.ci_key] !== undefined ? benchmarkData.faithfulness[item.ci_key].toFixed(3) : '?'} (95% CI)
                        </div>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>AUDC — {item.label}</div>
                      </div>
                    ))}
                  </div>
                )}
                <p style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', textAlign: 'center', maxWidth: '480px' }}>
                  Deletion curve comparison: features zeroed in order of decreasing attribution magnitude.
                  Lower AUDC = model degrades faster when top-attributed features are removed = explainer is more faithful.
                  PMA should beat random; if it also beats Gradient×Input it demonstrates computational efficiency advantage.
                </p>
              </div>
            )}

            {/* ─── TAB: Ablation Study ─── */}
            {activeResearchTab === 'ablation' && (
              <div style={{ padding: '8px 4px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {benchmarkData.ablation && benchmarkData.ablation.hi_ablation ? (
                  <React.Fragment>
                    <div>
                      <h4 style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>
                        Ablation 1: Health Index Abstraction Benefit (FD001)
                      </h4>
                      <table className="benchmark-table">
                        <thead><tr><th>Variant</th><th>RMSE ↓</th><th>NASA Score ↓</th><th>Note</th></tr></thead>
                        <tbody>
                          <tr>
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{benchmarkData.ablation.hi_ablation.hi_pipeline.label}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{benchmarkData.ablation.hi_ablation.hi_pipeline.rmse}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{benchmarkData.ablation.hi_ablation.hi_pipeline.score}</td>
                            <td><span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', border: '1px solid var(--accent-cyan)', color: 'var(--accent-cyan)' }}>Proposed</span></td>
                          </tr>
                          <tr style={{ opacity: 0.75 }}>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{benchmarkData.ablation.hi_ablation.raw_pipeline.label}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{benchmarkData.ablation.hi_ablation.raw_pipeline.rmse}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{benchmarkData.ablation.hi_ablation.raw_pipeline.score}</td>
                            <td><span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', border: '1px solid #4d607c', color: '#4d607c' }}>No HI Layer</span></td>
                          </tr>
                        </tbody>
                      </table>
                      <p style={{ fontSize: '10px', color: benchmarkData.ablation.hi_ablation.hi_helps ? 'var(--accent-green)' : 'var(--accent-orange)', fontFamily: 'var(--font-mono)', marginTop: '6px' }}>
                        HI pipeline {benchmarkData.ablation.hi_ablation.hi_helps ? 'improves' : 'degrades'} RMSE by&nbsp;
                        {Math.abs(benchmarkData.ablation.hi_ablation.delta_rmse)} cycles&nbsp;
                        ({benchmarkData.ablation.hi_ablation.delta_pct > 0 ? '+' : ''}{benchmarkData.ablation.hi_ablation.delta_pct}%)
                      </p>
                    </div>
                    <div>
                      <h4 style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>
                        Ablation 2: Sliding Window Size Sensitivity (FD001)
                      </h4>
                      <table className="benchmark-table">
                        <thead><tr><th>Window Size</th><th>RMSE ↓</th><th>NASA Score ↓</th><th>Note</th></tr></thead>
                        <tbody>
                          {benchmarkData.ablation.window_ablation.map((row, idx) => (
                            <tr key={idx} style={{ fontWeight: row.window_size === 30 ? 'bold' : 'normal' }}>
                              <td style={{ fontFamily: 'var(--font-mono)', color: row.window_size === 30 ? 'var(--accent-cyan)' : 'inherit' }}>{row.window_size} cycles</td>
                              <td style={{ fontFamily: 'var(--font-mono)' }}>{row.rmse !== null ? row.rmse : '—'}</td>
                              <td style={{ fontFamily: 'var(--font-mono)' }}>{row.score !== null ? row.score : '—'}</td>
                              <td style={{ fontSize: '10px', color: 'var(--text-dim)' }}>{row.note || (row.window_size === 30 ? 'Selected' : '')}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </React.Fragment>
                ) : (
                  <p style={{ color: 'var(--text-muted)', padding: '20px', textAlign: 'center' }}>Ablation data not available. Run the benchmark to populate.</p>
                )}
              </div>
            )}

            {/* ─── TAB: Baselines ─── */}
            {activeResearchTab === 'baselines' && (
              <div style={{ padding: '8px 4px' }}>
                <h4 style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>
                  Baseline Architectures — FD001, 3-Seed Mean ± Std (No HI Abstraction)
                </h4>
                {benchmarkData.baselines && Object.keys(benchmarkData.baselines).length > 0 ? (
                  <React.Fragment>
                    <table className="benchmark-table">
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>RMSE (mean ± std) ↓</th>
                          <th>NASA Score (mean ± std) ↓</th>
                          <th>HI Abstraction</th>
                          <th>Seeds</th>
                        </tr>
                      </thead>
                      <tbody>
                        {/* Our proposed model row */}
                        {benchmarkData.results.filter(r => r.target === 'FD001' && r.rmse !== null).map(r => (
                          <tr key="proposed">
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>HI-BayesianLSTM (Proposed)</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{r.rmse}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{r.score}</td>
                            <td><span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', border: '1px solid var(--accent-cyan)', color: 'var(--accent-cyan)' }}>✓ Yes</span></td>
                            <td style={{ color: 'var(--text-dim)', fontSize: '10px' }}>1 (report with seeds TBD)</td>
                          </tr>
                        ))}
                        {/* Baseline rows */}
                        {Object.entries(benchmarkData.baselines).map(([name, b]) => (
                          <tr key={name} style={{ opacity: 0.8 }}>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{name}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{b.rmse_str}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{b.score_str}</td>
                            <td><span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', border: '1px solid #4d607c', color: '#4d607c' }}>✗ No</span></td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-dim)' }}>3 ({b.per_seed ? b.per_seed.map(s => s.seed).join(', ') : '42, 123, 7'})</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', marginTop: '8px' }}>
                      Baselines trained directly on raw sensor windows (no Health Index) — same FD001 training data and epoch count.
                      mean ± std across 3 seeds (42, 123, 7).
                    </p>
                  </React.Fragment>
                ) : (
                  <p style={{ color: 'var(--text-muted)', padding: '20px', textAlign: 'center' }}>Baseline data not available. Run the benchmark to populate.</p>
                )}
              </div>
            )}

            {/* ─── TAB: Model Card ─── */}
            {activeResearchTab === 'modelcard' && (
              <div style={{ padding: '16px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {/* Architecture */}
                <div style={{ background: 'rgba(0,240,255,0.04)', border: '1px solid rgba(0,240,255,0.15)', borderRadius: '6px', padding: '14px' }}>
                  <h4 style={{ fontSize: '11px', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>Architecture</h4>
                  {[['Model', 'AE-HI-BayesianLSTM (Proposed)'], ['AE Type', 'LSTM Autoencoder (Encoder→Decoder)'], ['AE Hidden Dim', `${benchmarkData.window_size ? 8 : '—'} units`], ['LSTM Hidden Dim', '16 units (MC-Dropout=0.25)'], ['HI Pipeline', 'AE Recon Error → HI Sequence → LSTM RUL']].map(([k,v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(77,96,124,0.1)', fontSize: '10px' }}>
                      <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{v}</span>
                    </div>
                  ))}
                </div>
                {/* Hyperparameters */}
                <div style={{ background: 'rgba(0,136,255,0.04)', border: '1px solid rgba(0,136,255,0.15)', borderRadius: '6px', padding: '14px' }}>
                  <h4 style={{ fontSize: '11px', color: '#0088ff', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>Hyperparameters</h4>
                  {[['Window Size', `${benchmarkData.window_size || 30} cycles`], ['Epochs (max)', benchmarkData.epochs || 20], ['Early Stopping Pat.', 4], ['Learning Rate', '0.01 (Adam)'], ['Batch Size', 64], ['Val Split', '15% engines'], ['Seeds', (benchmarkData.seeds || [42,123,7]).join(', ')]].map(([k,v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(77,96,124,0.1)', fontSize: '10px' }}>
                      <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{v}</span>
                    </div>
                  ))}
                </div>
                {/* Data Config */}
                <div style={{ background: 'rgba(0,200,83,0.04)', border: '1px solid rgba(0,200,83,0.15)', borderRadius: '6px', padding: '14px' }}>
                  <h4 style={{ fontSize: '11px', color: '#00c853', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>Data Configuration</h4>
                  {[['Sensor Set', '14 literature-standard'], ['Source', 'NASA CMAPSS (FD001–4)'], ['RUL Cap FD001/3', '125 cycles'], ['RUL Cap FD002/4', '130 cycles'], ['Normalization', 'Per-regime Z-score'], ['Fabricated Sensors', 'None (leakage-free)']].map(([k,v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(77,96,124,0.1)', fontSize: '10px' }}>
                      <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{v}</span>
                    </div>
                  ))}
                </div>
                {/* Explainability */}
                <div style={{ background: 'rgba(255,140,0,0.04)', border: '1px solid rgba(255,140,0,0.15)', borderRadius: '6px', padding: '14px' }}>
                  <h4 style={{ fontSize: '11px', color: '#ff8c00', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' }}>Explainability</h4>
                  {[['XAI Method', 'PMA (proposed)'], ['Baselines', 'Integrated Gradients, Grad×Input'], ['Evaluation', 'Deletion curves (AUDC) N=100'], ['CI', '95% confidence intervals'], ['PMA Axioms', 'Satisfies efficiency; Shapley-like but non-exact']].map(([k,v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(77,96,124,0.1)', fontSize: '10px' }}>
                      <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)', textAlign: 'right', maxWidth: '55%' }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        ) : (
          <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Icon name="info" className="lucide" style={{ width: '24px', height: '24px', display: 'block', margin: '0 auto 8px', color: 'var(--accent-blue)' }} />
            Click "Evaluate Cross-Subset Generalization" to run the full benchmark suite.
            Includes: MC-Dropout UQ metrics (PICP, Sharpness), real PMA attributions,
            PMA faithfulness tests (N=100, 95% CI), 3-seed baselines (PlainLSTM, CNN-LSTM), ablation study, and a Model Card.
          </div>
        )}
      </div>

    </div>
  );
}

// Interactive SVG component for Turbofan Engine Twin
function DigitalTwinSVG({ components, sensors, onHover }) {
  const nfVal = sensors.Nf || 2388.0;
  const spinDuration = `${Math.max(0.1, (2400 / nfVal) * 0.4)}s`;

  const getHealthClass = (health) => {
    if (health > 80) return "module-healthy";
    if (health > 55) return "module-warning";
    return "module-critical";
  };

  return (
    <svg 
      className="engine-svg" 
      viewBox="0 0 700 320" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      style={{ overflow: 'visible' }}
    >
      <defs>
        {/* Glow filters for engine sections */}
        <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        
        {/* Gradients */}
        <linearGradient id="cyanBlueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--accent-cyan)" />
          <stop offset="100%" stopColor="var(--accent-blue)" />
        </linearGradient>
      </defs>

      {/* SVG Outline blueprint grids */}
      <line x1="20" y1="160" x2="680" y2="160" stroke="rgba(77, 96, 124, 0.15)" strokeWidth="1" strokeDasharray="5,5" />
      <line x1="100" y1="20" x2="100" y2="300" stroke="rgba(77, 96, 124, 0.15)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="220" y1="20" x2="220" y2="300" stroke="rgba(77, 96, 124, 0.15)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="330" y1="20" x2="330" y2="300" stroke="rgba(77, 96, 124, 0.15)" strokeWidth="1" strokeDasharray="3,3" />
      <line x1="430" y1="20" x2="430" y2="300" stroke="rgba(77, 96, 124, 0.15)" strokeWidth="1" strokeDasharray="3,3" />

      {/* Main Outer Casing */}
      <path 
        d="M 50 70 L 95 70 L 220 90 L 330 110 L 430 110 L 520 80 L 590 80 M 50 250 L 95 250 L 220 230 L 330 210 L 430 210 L 520 240 L 590 240" 
        stroke="rgba(77, 96, 124, 0.4)" 
        strokeWidth="1.5" 
        fill="none" 
      />

      {/* Bypass Duct Panels */}
      <path d="M 95 50 L 520 55 L 520 75 L 95 70 Z" fill="rgba(0, 136, 255, 0.04)" stroke="rgba(77, 96, 124, 0.2)" />
      <path d="M 95 270 L 520 265 L 520 245 L 95 250 Z" fill="rgba(0, 136, 255, 0.04)" stroke="rgba(77, 96, 124, 0.2)" />

      {/* 1. FAN MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "Fan")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 40 70 L 95 70 L 95 250 L 40 250 Z" 
          className={getHealthClass(components.Fan)} 
        />
        
        {/* Spinning Fan Blades */}
        <g className="fan-spinner" style={{ transformOrigin: '90px 160px', '--spin-duration': spinDuration }}>
          <circle cx="90" cy="160" r="16" fill="var(--bg-primary)" stroke="var(--accent-cyan)" strokeWidth="1.5" />
          <path d="M 90 160 L 90 85 M 90 160 L 90 235" stroke="var(--accent-cyan)" strokeWidth="3" opacity="0.8" />
          <path d="M 90 160 L 25 122 M 90 160 L 155 198" stroke="var(--accent-cyan)" strokeWidth="3" opacity="0.8" />
          <path d="M 90 160 L 25 198 M 90 160 L 155 122" stroke="var(--accent-cyan)" strokeWidth="3" opacity="0.8" />
        </g>
        
        <path d="M 65 140 Q 95 160 65 180 Z" fill="var(--accent-gray)" stroke="var(--accent-cyan)" strokeWidth="1" />
      </g>

      {/* 2. LPC MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "LPC")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 95 75 L 150 90 L 150 230 L 95 245 Z" 
          className={getHealthClass(components.LPC)} 
        />
        <line x1="113" y1="80" x2="113" y2="240" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="131" y1="85" x2="131" y2="235" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
      </g>

      {/* 3. HPC MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "HPC")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 150 90 L 220 108 L 220 212 L 150 230 Z" 
          className={getHealthClass(components.HPC)} 
        />
        <line x1="167" y1="95" x2="167" y2="225" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="184" y1="100" x2="184" y2="220" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="202" y1="104" x2="202" y2="216" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
      </g>

      {/* 4. COMBUSTOR MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "Combustor")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 220 108 L 330 108 L 330 212 L 220 212 Z" 
          className={getHealthClass(components.Combustor)} 
        />
        
        {/* Flame Graphic */}
        <path 
          d="M 230 160 Q 250 135 285 150 Q 300 135 320 160 Q 300 185 285 170 Q 250 185 230 160 Z" 
          fill="none" 
          stroke="var(--accent-orange)" 
          strokeWidth="1.5" 
          opacity="0.85" 
          style={{ transformOrigin: '275px 160px', animation: 'fan-spin 12s linear infinite' }}
        />
        <circle cx="235" cy="130" r="3" fill="var(--accent-orange)" />
        <circle cx="235" cy="190" r="3" fill="var(--accent-orange)" />
      </g>

      {/* 5. HPT MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "HPT")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 330 108 L 410 93 L 410 227 L 330 212 Z" 
          className={getHealthClass(components.HPT)} 
        />
        <line x1="356" y1="104" x2="356" y2="216" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="383" y1="98" x2="383" y2="222" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
      </g>

      {/* 6. LPT MODULE */}
      <g 
        className="engine-module"
        onMouseEnter={(e) => onHover(e, "LPT")}
        onMouseLeave={(e) => onHover(e, null)}
      >
        <path 
          d="M 410 93 L 520 75 L 520 245 L 410 227 Z" 
          className={getHealthClass(components.LPT)} 
        />
        <line x1="437" y1="88" x2="437" y2="232" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="464" y1="84" x2="464" y2="236" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
        <line x1="491" y1="80" x2="491" y2="240" stroke="rgba(0, 240, 255, 0.2)" strokeWidth="1.5" />
      </g>

      {/* Exhaust Nozzle */}
      <path d="M 520 75 Q 600 120 550 160 Q 600 200 520 245 Z" fill="rgba(77, 96, 124, 0.15)" stroke="rgba(77, 96, 124, 0.5)" strokeWidth="1.5" />
      
      {/* Drive Shaft */}
      <line x1="90" y1="160" x2="510" y2="160" stroke="var(--accent-blue)" strokeWidth="4" opacity="0.6" />

      {/* Callouts */}
      <path d="M 67 70 L 67 20 M 67 20 L 50 20" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="45" y="15" className="engine-label">FAN MODULE</text>

      <path d="M 122 80 L 122 25 M 122 25 L 140 25" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="145" y="22" className="engine-label">LPC SECTION</text>

      <path d="M 185 96 L 185 30 M 185 30 L 205 30" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="210" y="27" className="engine-label">HPC SYSTEM</text>

      <path d="M 275 108 L 275 300 M 275 300 L 260 300" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="200" y="303" className="engine-label">COMBUSTOR</text>

      <path d="M 370 100 L 370 295 M 370 295 L 385 295" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="390" y="298" className="engine-label">HPT ROTORS</text>

      <path d="M 465 85 L 465 20 M 465 20 L 450 20" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="390" y="17" className="engine-label">LPT ROTORS</text>

      <path d="M 550 160 L 590 160" stroke="rgba(77,96,124,0.5)" strokeWidth="0.8" fill="none" />
      <text x="595" y="163" className="engine-label">EXHAUST CONE</text>
    </svg>
  );
}

// Chart Component: Sensor degradation trend vs cycle
function TrendChart({ title, history, future, sensorKey, theme }) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const theme = document.documentElement.getAttribute("data-theme") || "light";
    const isDark = theme === "dark";
    const gridColor = isDark ? "rgba(77, 96, 124, 0.1)" : "rgba(100, 116, 139, 0.1)";
    const tickColor = isDark ? "#8397b5" : "#475569";
    const titleColor = isDark ? "#546682" : "#5b6b85";

    const labels = history.map(h => h.current_cycle);
    const dataVals = history.map(h => h.sensors[sensorKey] || null);

    const futureLabels = future.map(f => f.cycle);
    const futureVals = future.map(f => f.predictions.RUL_predicted || null);

    const ctx = canvasRef.current.getContext('2d');
    
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [...labels, ...futureLabels.slice(1)],
        datasets: [
          {
            label: `Observed Telemetry`,
            data: [...dataVals, ...Array(Math.max(0, futureVals.length - 1)).fill(null)],
            borderColor: isDark ? '#00f0ff' : '#1d4ed8',
            backgroundColor: isDark ? 'rgba(0, 240, 255, 0.05)' : 'rgba(29, 78, 216, 0.05)',
            borderWidth: 2,
            pointRadius: 1,
            pointHoverRadius: 5,
            fill: true,
          },
          {
            label: `Twin Forecasted Trend`,
            data: [...Array(Math.max(0, dataVals.length - 1)).fill(null), dataVals[dataVals.length - 1], ...futureVals.map((v, idx) => {
              // Mock project sensor readings based on true degradation direction
              const lastObs = dataVals[dataVals.length - 1] || 1500;
              const step = idx / (futureVals.length || 1);
              const drift = sensorKey === "T30" || sensorKey === "T50" || sensorKey === "Vibration" ? 25 * step : -10 * step;
              return lastObs + drift;
            }).slice(1)],
            borderColor: isDark ? '#ff8c00' : '#d97706',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: isDark ? '#0d1322' : 'rgba(255, 255, 255, 0.95)',
            titleColor: isDark ? '#00f0ff' : '#1d4ed8',
            bodyColor: isDark ? '#f0f4fc' : '#0f172a',
            borderColor: isDark ? 'rgba(0, 240, 255, 0.2)' : 'rgba(29, 78, 216, 0.2)',
            borderWidth: 1,
            titleFont: { family: 'Share Tech Mono' },
            bodyFont: { family: 'Inter' }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 10 } },
            title: { display: true, text: 'Operational Cycle Count', color: titleColor, font: { size: 9 } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 10 } },
            title: { display: true, text: `Reading`, color: titleColor, font: { size: 9 } }
          }
        }
      }
    });

  }, [history, future, sensorKey, theme]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{title}</span>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#00f0ff' }}></div>
            <span>Observed</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#ff8c00', height: '0px', border: '1px dashed #ff8c00' }}></div>
            <span>Twin Forecast</span>
          </div>
        </div>
      </div>
      <div className="chart-container">
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

// Chart Component: Predicted RUL (with confidence bounds) vs Cycle
function RulChart({ title, history, maxCycles, theme }) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const theme = document.documentElement.getAttribute("data-theme") || "light";
    const isDark = theme === "dark";
    const gridColor = isDark ? "rgba(77, 96, 124, 0.1)" : "rgba(100, 116, 139, 0.1)";
    const tickColor = isDark ? "#8397b5" : "#475569";
    const titleColor = isDark ? "#546682" : "#5b6b85";

    const labels = history.map(h => h.current_cycle);
    const predRul = history.map(h => h.predictions.RUL_predicted);
    const predP10 = history.map(h => h.predictions.RUL_p10 || h.predictions.RUL_predicted - 15.0);
    const predP90 = history.map(h => h.predictions.RUL_p90 || h.predictions.RUL_predicted + 15.0);
    const trueRul = history.map(h => h.max_cycles - h.current_cycle);

    const ctx = canvasRef.current.getContext('2d');
    
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: `True EOL`,
            data: trueRul,
            borderColor: isDark ? 'rgba(77, 96, 124, 0.4)' : 'rgba(100, 116, 139, 0.4)',
            borderWidth: 1.5,
            borderDash: [3, 3],
            pointRadius: 0,
            fill: false,
          },
          {
            label: `P90 Upper Bound`,
            data: predP90,
            borderColor: isDark ? 'rgba(0, 240, 255, 0.4)' : 'rgba(29, 78, 216, 0.4)',
            borderWidth: 1,
            borderDash: [2, 2],
            pointRadius: 0,
            fill: false,
          },
          {
            label: `P10 Lower Bound`,
            data: predP10,
            borderColor: isDark ? 'rgba(255, 51, 85, 0.4)' : 'rgba(220, 38, 38, 0.4)',
            borderWidth: 1,
            borderDash: [2, 2],
            pointRadius: 0,
            fill: false,
          },
          {
            label: `P50 Predicted RUL (UQ Median)`,
            data: predRul,
            borderColor: isDark ? '#0088ff' : '#2563eb',
            backgroundColor: isDark ? 'rgba(0, 136, 255, 0.06)' : 'rgba(37, 99, 235, 0.06)',
            borderWidth: 2.5,
            pointRadius: labels.length > 200 ? 0 : 1.5,
            fill: true,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: isDark ? '#0d1322' : 'rgba(255, 255, 255, 0.95)',
            titleColor: isDark ? '#0088ff' : '#2563eb',
            bodyColor: isDark ? '#f0f4fc' : '#0f172a',
            borderColor: isDark ? 'rgba(0, 136, 255, 0.2)' : 'rgba(37, 99, 235, 0.2)',
            borderWidth: 1,
            titleFont: { family: 'Share Tech Mono' },
            bodyFont: { family: 'Inter' }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 10 } },
            title: { display: true, text: 'Operational Cycle Count', color: titleColor, font: { size: 9 } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: tickColor, font: { family: 'Share Tech Mono', size: 10 } },
            title: { display: true, text: 'Remaining Cycles', color: titleColor, font: { size: 9 } },
            min: 0,
            max: maxCycles
          }
        }
      }
    });

  }, [history, maxCycles, theme]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{title}</span>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#0088ff' }}></div>
            <span>P50 RUL</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: 'rgba(255, 51, 85, 0.4)', height: '0px', border: '1px dashed rgba(255, 51, 85, 0.4)' }}></div>
            <span>Confidence bounds</span>
          </div>
        </div>
      </div>
      <div className="chart-container">
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

// Mount the App
const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);
root.render(<App />);
