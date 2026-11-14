const DEFAULT_SETTINGS = {
  enabled: false,
  sourceLang: "auto",
  targetLang: "es",
  voice: "auto",
  ttsBackend: "xtts_v2",
  diarization: true,
  autoSync: true,
  captureMode: "auto",
  delayMs: 2500,
  duckDb: -14,
  speed: 1.08,
  useCaptions: true
};

const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");
const tabHint = document.getElementById("tabHint");
const btnRetry = document.getElementById("btnRetry");
const btnDirBoth = document.getElementById("btnDirBoth");
const btnDirRxOnly = document.getElementById("btnDirRxOnly");
const btnDirTxOnly = document.getElementById("btnDirTxOnly");
const btnDirMuteAll = document.getElementById("btnDirMuteAll");

const chkEnabled = document.getElementById("chkEnabled");
const selSourceLang = document.getElementById("selSourceLang");
const selTargetLang = document.getElementById("selTargetLang");
const selTtsBackend = document.getElementById("selTtsBackend");
const selVoice = document.getElementById("selVoice");
const chkDiarization = document.getElementById("chkDiarization");
const selCaptureMode = document.getElementById("selCaptureMode");
const chkAutoSync = document.getElementById("chkAutoSync");
const syncModeTag = document.getElementById("syncModeTag");
const delayFieldBox = document.getElementById("delayFieldBox");
const rngDelay = document.getElementById("rngDelay");
const lblDelay = document.getElementById("lblDelay");
const rngDuck = document.getElementById("rngDuck");
const lblDuck = document.getElementById("lblDuck");
const rngSpeed = document.getElementById("rngSpeed");
const lblSpeed = document.getElementById("lblSpeed");

function renderDesktopChannels(muteTx, muteRx) {
  const buttons = [
    [btnDirBoth, !muteTx && !muteRx, "active-both"],
    [btnDirRxOnly, muteTx && !muteRx, "active-rx"],
    [btnDirTxOnly, !muteTx && muteRx, "active-tx"],
    [btnDirMuteAll, muteTx && muteRx, "active-mute"]
  ];
  for (const [btn, isActive, cls] of buttons) {
    if (!btn) continue;
    btn.className = isActive ? `dir-btn ${cls}` : "dir-btn";
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
}

function setDesktopChannels(muteTx, muteRx) {
  renderDesktopChannels(Boolean(muteTx), Boolean(muteRx));
  chrome.runtime.sendMessage(
    {
      type: "PB_SET_CHANNELS",
      payload: { mute_tx: Boolean(muteTx), mute_rx: Boolean(muteRx) }
    },
    (resp) => {
      if (chrome.runtime.lastError || !resp || !resp.ok || !resp.data) {
        statusBox.className = "status-box status-err";
        statusText.textContent = "No se detecta Ignait abierto";
        return;
      }
      renderDesktopChannels(Boolean(resp.data.mute_tx), Boolean(resp.data.mute_rx));
      setTimeout(queryActiveTabStatus, 180);
    }
  );
}

function pingDesktopEngine() {
  statusBox.className = "status-box";
  statusText.textContent = "Conectando con Ignait...";
  chrome.runtime.sendMessage({ type: "PB_PING" }, (resp) => {
    if (chrome.runtime.lastError || !resp || !resp.ok) {
      statusBox.className = "status-box status-err";
      statusText.textContent = "No se detecta Ignait abierto";
      return;
    }
    const info = resp.data || {};
    statusBox.className = "status-box status-ok";
    const statusLabel = info.mute_tx && info.mute_rx
      ? "Todo en pausa"
      : info.mute_rx
        ? "Audio de videos en pausa"
        : info.mute_tx
          ? "Solo videos (micrófono apagado)"
          : "Audio y videos listos";
    statusText.textContent = `Conectado • ${statusLabel}`;
    renderDesktopChannels(Boolean(info.mute_tx), Boolean(info.mute_rx));
  });
}

function queryActiveTabStatus() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0] || typeof tabs[0].id !== "number") return;
    chrome.tabs.sendMessage(tabs[0].id, { type: "PB_GET_STATUS" }, (resp) => {
      if (chrome.runtime.lastError || !resp) {
        tabHint.textContent = "Abre un video y recarga la pestaña si es necesario.";
        return;
      }
      if (!resp.hasVideo) {
        tabHint.textContent = "No hay ningún video abierto en esta pestaña.";
        return;
      }
      const dubState = resp.wsConnected ? "Traduciendo video activo" : "Doblaje en pausa";
      const syncInfo = resp.syncStrategy === "PREFETCH_ZERO_LAG"
        ? "Sincronización instantánea"
        : resp.syncStrategy === "AUDIO_ADAPTIVE_EWMA"
          ? "Sincronización automática"
          : `Retardo ${resp.effectiveDelayMs || 0} ms`;
      tabHint.textContent = `${dubState} • ${syncInfo}`;
    });
  });
}

function updateLabels() {
  const autoOn = chkAutoSync.checked;
  rngDelay.disabled = autoOn;
  delayFieldBox.style.opacity = autoOn ? "0.48" : "1.0";
  syncModeTag.textContent = autoOn ? "AUTO" : "MANUAL";
  syncModeTag.style.color = autoOn ? "#34d399" : "#fbbf24";
  lblDelay.textContent = autoOn ? "Auto EWMA" : `${rngDelay.value} ms`;
  lblDuck.textContent = `${rngDuck.value} dB`;
  lblSpeed.textContent = `${Number(rngSpeed.value).toFixed(2)}x`;
}

function collectAndSaveSettings() {
  updateLabels();
  const capMode = selCaptureMode.value;
  const next = {
    enabled: chkEnabled.checked,
    sourceLang: selSourceLang.value,
    targetLang: selTargetLang.value,
    ttsBackend: selTtsBackend.value,
    voice: selVoice.value,
    diarization: chkDiarization.checked,
    autoSync: chkAutoSync.checked,
    captureMode: capMode,
    useCaptions: capMode !== "audio",
    delayMs: Number(rngDelay.value),
    duckDb: Number(rngDuck.value),
    speed: Number(rngSpeed.value)
  };
  chrome.storage.local.set(next, () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && typeof tabs[0].id === "number") {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "PB_APPLY_SETTINGS", settings: next },
          () => {
            void chrome.runtime.lastError;
            setTimeout(queryActiveTabStatus, 180);
          }
        );
      }
    });
  });
}

function init() {
  chrome.storage.local.get(DEFAULT_SETTINGS, (stored) => {
    const s = { ...DEFAULT_SETTINGS, ...stored };
    chkEnabled.checked = Boolean(s.enabled);
    selSourceLang.value = s.sourceLang;
    selTargetLang.value = s.targetLang;
    selTtsBackend.value = s.ttsBackend || "xtts_v2";
    selVoice.value = s.voice;
    chkDiarization.checked = s.diarization !== false;
    chkAutoSync.checked = s.autoSync !== false;
    selCaptureMode.value = s.captureMode || (s.useCaptions ? "auto" : "audio");
    rngDelay.value = String(s.delayMs);
    rngDuck.value = String(s.duckDb);
    rngSpeed.value = String(s.speed);
    updateLabels();
  });

  [
    chkEnabled,
    selSourceLang,
    selTargetLang,
    selTtsBackend,
    selVoice,
    chkDiarization,
    chkAutoSync,
    selCaptureMode,
    rngDelay,
    rngDuck,
    rngSpeed
  ].forEach((el) => {
    el.addEventListener("input", updateLabels);
    el.addEventListener("change", collectAndSaveSettings);
  });

  if (btnDirBoth) btnDirBoth.addEventListener("click", () => setDesktopChannels(false, false));
  if (btnDirRxOnly) btnDirRxOnly.addEventListener("click", () => setDesktopChannels(true, false));
  if (btnDirTxOnly) btnDirTxOnly.addEventListener("click", () => setDesktopChannels(false, true));
  if (btnDirMuteAll) btnDirMuteAll.addEventListener("click", () => setDesktopChannels(true, true));

  btnRetry.addEventListener("click", () => {
    pingDesktopEngine();
    queryActiveTabStatus();
  });

  pingDesktopEngine();
  queryActiveTabStatus();
}

init();
