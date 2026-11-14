(() => {
  if (window.__pbMediaBridgeLoaded) {
    return;
  }
  window.__pbMediaBridgeLoaded = true;

  const TARGET_RATE = 16000;
  const FRAME_SAMPLES = 512;
  const MIN_SPEECH_RMS_DBFS = -40.0;
  const SILENCE_HANGOVER_MS = 380;
  const MIN_PHRASE_SAMPLES = Math.floor(TARGET_RATE * 0.45);
  const MAX_PHRASE_SAMPLES = Math.floor(TARGET_RATE * 2.6);
  const PREROLL_SAMPLES = Math.floor(TARGET_RATE * 0.16);
  const MAX_IN_FLIGHT = 2;

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

  let settings = { ...DEFAULT_SETTINGS };
  let activeVideo = null;
  let audioGraph = null;
  const boundVideos = new WeakMap();

  // Streaming WebSocket por runtime.Port
  let wsPort = null;
  let wsConnected = false;
  let wsErrorCount = 0;

  // Fallback HTTP si el WebSocket falla repetidamente
  let prerollChunks = [];
  let prerollLen = 0;
  let speechChunks = [];
  let speechLen = 0;
  let isSpeaking = false;
  let silenceAccumMs = 0;
  let inFlightCount = 0;

  // Planificador sample-clock + sincronía automática
  const activeTtsSources = new Set();
  const seqScheduleMap = new Map();
  const seqRateMap = new Map();
  const droppedSeqs = new Set();
  let nextScheduledTime = 0;
  let duckRestoreTimer = null;

  const syncState = {
    strategy: "AUDIO_ADAPTIVE_EWMA",
    ewmaLatencyMs: 1400,
    recommendedDelayMs: 2200,
    lastAppliedDelaySec: -1,
    prefetchedVideoKey: "",
    prefetchedCues: [],
    captionSeq: 1000000,
    playbackGen: 0,
    lastVideoTimeSec: null,
    sentPrefetchKeys: new Set(),
    lastClockSentTs: 0,
    lastSentVoiceProbeOnly: null,
    desktopRxMuted: false
  };

  function getStreamKey(meta) {
    const pgen = typeof meta.playback_gen === "number" ? meta.playback_gen : syncState.playbackGen;
    const kind = meta.source_kind || (meta.is_prefetched ? "caption" : "audio");
    return `${pgen}:${kind}:${Number(meta.seq || 0)}`;
  }

  // Fallback DOM YouTube cuando no hay timedtext JSON3
  let captionTimer = null;
  let lastCaptionSent = "";
  let lastCaptionSeen = "";
  let lastCaptionActiveTs = 0;

  // HUD / Overlay
  let overlayRoot = null;
  let overlayStatusBtn = null;
  let overlayResetBtn = null;
  let overlaySubtitleBox = null;
  let subtitleHideTimer = null;
  let currentOverlayStatus = "LISTO";

  function uint8ToBase64(bytes) {
    let binary = "";
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) {
      const slice = bytes.subarray(i, Math.min(bytes.length, i + step));
      binary += String.fromCharCode.apply(null, slice);
    }
    return btoa(binary);
  }

  function base64ToArrayBuffer(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function isMusicOrEmptyCue(text) {
    const clean = String(text || "")
      .replace(/[♪♫]+/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (clean.length < 2) return true;
    return /^\[(?:music|applause|laughter|música|aplausos|risas)\]$/i.test(clean);
  }

  function mergeTimedTextEvents(events) {
    if (!Array.isArray(events) || events.length === 0) return [];
    const rawCues = [];
    for (const ev of events) {
      if (!ev || typeof ev.tStartMs !== "number") continue;
      const segs = Array.isArray(ev.segs) ? ev.segs : [];
      const text = segs
        .map((s) => (s && typeof s.utf8 === "string" ? s.utf8 : ""))
        .join("")
        .replace(/[\r\n]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      if (!text || isMusicOrEmptyCue(text)) continue;
      const startSec = Math.max(0, ev.tStartMs / 1000.0);
      const durMs = typeof ev.dDurationMs === "number" && ev.dDurationMs > 0 ? ev.dDurationMs : 2200;
      const endSec = startSec + Math.max(0.4, durMs / 1000.0);
      rawCues.push({ startSec, endSec, text });
    }
    if (rawCues.length === 0) return [];
    rawCues.sort((a, b) => a.startSec - b.startSec);

    const merged = [];
    let cur = { ...rawCues[0] };

    for (let i = 1; i < rawCues.length; i++) {
      const nxt = rawCues[i];
      if (nxt.text === cur.text && nxt.startSec <= cur.endSec + 0.6) {
        cur.endSec = Math.max(cur.endSec, nxt.endSec);
        continue;
      }
      const gapSec = nxt.startSec - cur.endSec;
      const combinedWords = `${cur.text} ${nxt.text}`.trim().split(/\s+/).length;
      const spanSec = Math.max(cur.endSec, nxt.endSec) - cur.startSec;
      const curDurationSec = cur.endSec - cur.startSec;
      const endsSentence = /[.?!…]["')\]]*$/.test(cur.text);

      const shouldSplit =
        gapSec > 0.65 ||
        combinedWords > 15 ||
        spanSec > 4.2 ||
        (endsSentence && curDurationSec >= 1.0);

      if (shouldSplit) {
        merged.push(cur);
        cur = { ...nxt };
      } else {
        const joined = nxt.text.startsWith(cur.text) ? nxt.text : `${cur.text} ${nxt.text}`.replace(/\s+/g, " ").trim();
        cur = {
          startSec: cur.startSec,
          endSec: Math.max(cur.endSec, nxt.endSec),
          text: joined
        };
      }
    }
    merged.push(cur);
    return merged;
  }

  function ingestTrackTextCues(cuesList, keyLabel = "html5") {
    if (!Array.isArray(cuesList) || cuesList.length === 0) return;
    syncState.prefetchedCues = cuesList;
    syncState.prefetchedVideoKey = keyLabel;
    syncState.sentPrefetchKeys.clear();
    if (settings.enabled) {
      applyEffectiveDelay(false);
      tickAutoPrefetchAndClock();
      updateOverlayBadge();
    }
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const msg = event.data;
    if (!msg || msg.source !== "pb-yt-bridge") return;
    if (msg.type === "PB_TIMEDTEXT_JSON3" && Array.isArray(msg.events)) {
      const merged = mergeTimedTextEvents(msg.events);
      if (merged.length > 0) {
        ingestTrackTextCues(merged, `${msg.videoId || "yt"}:${msg.lang || "cc"}`);
      }
    }
  });

  function requestYouTubeTracks(force = false) {
    if (!window.location.hostname.includes("youtube.com")) return;
    window.postMessage(
      {
        source: "pb-content",
        type: "PB_YT_REQUEST_TRACKS",
        preferredLang: settings.sourceLang || "auto",
        force
      },
      window.location.origin
    );
  }

  function hydrateHtml5TextTracksIfEmpty() {
    if (syncState.prefetchedCues.length > 0 || !activeVideo || !activeVideo.textTracks) return;
    const tracks = activeVideo.textTracks;
    for (let i = 0; i < tracks.length; i++) {
      const tr = tracks[i];
      if (!tr || (tr.kind !== "subtitles" && tr.kind !== "captions")) continue;
      if (tr.mode === "disabled") continue;
      if (!tr.cues || tr.cues.length === 0) continue;
      const converted = [];
      for (let j = 0; j < tr.cues.length; j++) {
        const c = tr.cues[j];
        const text = String(c.text || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
        if (!text || isMusicOrEmptyCue(text)) continue;
        converted.push({
          tStartMs: Math.round(c.startTime * 1000),
          dDurationMs: Math.max(500, Math.round((c.endTime - c.startTime) * 1000)),
          segs: [{ utf8: text }]
        });
      }
      const merged = mergeTimedTextEvents(converted);
      if (merged.length > 0) {
        ingestTrackTextCues(merged, `track:${tr.language || i}`);
        break;
      }
    }
  }

  function getEffectiveSyncConfig() {
    const capMode = settings.captureMode || (settings.useCaptions ? "auto" : "audio");
    const allowPrefetch = settings.autoSync || capMode === "hybrid";
    const canUseAudioOnly = capMode === "audio";

    let hasPrefetchCoverage = false;
    if (!canUseAudioOnly && allowPrefetch && syncState.prefetchedCues.length > 0 && activeVideo) {
      const lastCue = syncState.prefetchedCues[syncState.prefetchedCues.length - 1];
      hasPrefetchCoverage = activeVideo.currentTime <= lastCue.endSec + 2.0;
    }

    if (hasPrefetchCoverage) {
      const horizonSec = Math.max(
        3.8,
        Math.min(8.5, (syncState.ewmaLatencyMs * 2.2 + 1600) / 1000.0)
      );
      syncState.strategy = "PREFETCH_ZERO_LAG";
      return { strategy: "PREFETCH_ZERO_LAG", delayMs: 0, horizonSec };
    }

    if (settings.autoSync) {
      syncState.strategy = "AUDIO_ADAPTIVE_EWMA";
      return {
        strategy: "AUDIO_ADAPTIVE_EWMA",
        delayMs: Math.max(750, Math.min(4500, syncState.recommendedDelayMs)),
        horizonSec: 0
      };
    }

    syncState.strategy = "MANUAL";
    return {
      strategy: "MANUAL",
      delayMs: Math.max(0, Math.min(4500, Number(settings.delayMs || 0))),
      horizonSec: 0
    };
  }

  function applyEffectiveDelay(forceImmediate = false) {
    if (!audioGraph) return;
    const { ctx, delayNode } = audioGraph;
    const syncCfg =
      settings.enabled && !syncState.desktopRxMuted
        ? getEffectiveSyncConfig()
        : { delayMs: 0, strategy: "MANUAL" };
    const targetSec = Math.max(0.0, Math.min(4.5, syncCfg.delayMs / 1000.0));
    const prevSec = syncState.lastAppliedDelaySec;

    if (!forceImmediate && prevSec >= 0 && Math.abs(targetSec - prevSec) < 0.14) {
      return;
    }
    syncState.lastAppliedDelaySec = targetSec;
    const tau = forceImmediate || targetSec === 0 ? 0.07 : 0.42;
    try {
      delayNode.delayTime.setTargetAtTime(targetSec, ctx.currentTime, tau);
    } catch (_e) {
      delayNode.delayTime.value = targetSec;
    }
  }

  function sendWsCommand(cmd) {
    if (!wsPort) return;
    try {
      wsPort.postMessage({ type: "ws_cmd", cmd });
    } catch (_err) {
      wsConnected = false;
      wsPort = null;
    }
  }

  function sendWsConfigure() {
    const { voiceProbeOnly } = computeAudioStreamMode();
    syncState.lastSentVoiceProbeOnly = voiceProbeOnly;
    sendWsCommand({
      type: "configure",
      source_lang: settings.sourceLang,
      target_lang: settings.targetLang,
      voice: settings.voice,
      speed: Number(settings.speed || 1.08),
      tts_backend: settings.ttsBackend || "xtts_v2",
      diarization: settings.diarization !== false,
      auto_sync: settings.autoSync !== false,
      voice_probe_only: voiceProbeOnly,
      playback_gen: syncState.playbackGen
    });
  }

  function ensureStreamPort() {
    if (wsPort) return wsPort;
    const port = chrome.runtime.connect({ name: "pb-media-stream" });
    wsPort = port;

    port.onMessage.addListener((msg) => {
      if (!msg || typeof msg !== "object" || wsPort !== port) return;

      if (msg.type === "ws_open") {
        wsConnected = true;
        wsErrorCount = 0;
        syncState.lastSentVoiceProbeOnly = null;
        currentOverlayStatus = "WS ACTIVO";
        updateOverlayBadge();
        sendWsConfigure();
        if (settings.enabled && activeVideo) {
          sendMediaClock(true);
        }
        return;
      }

      if (msg.type === "ws_close") {
        wsConnected = false;
        if (settings.enabled) {
          currentOverlayStatus = "RECONECTANDO";
          updateOverlayBadge("RECON...");
        }
        return;
      }

      if (msg.type === "ws_error") {
        wsConnected = false;
        wsErrorCount += 1;
        if (settings.enabled) {
          currentOverlayStatus = "ESPERANDO APP";
          updateOverlayBadge("ABRE PROTOBRIDGE");
        }
        return;
      }

      if (msg.type === "ws_json" && msg.data) {
        if (msg.data.type === "hello") {
          if (typeof msg.data.rx_muted === "boolean") {
            setDesktopRxMuted(msg.data.rx_muted);
          }
          if (typeof msg.data.recommended_delay_ms === "number") {
            syncState.recommendedDelayMs = msg.data.recommended_delay_ms;
            applyEffectiveDelay(false);
          }
          if (!syncState.desktopRxMuted) {
            const engLabel = msg.data.xtts_available ? "XTTS GPU" : "Kokoro";
            updateOverlayBadge(engLabel);
          }
        } else if (msg.data.type === "rx_mute_state") {
          setDesktopRxMuted(Boolean(msg.data.rx_muted));
        } else if (msg.data.type === "sync_telemetry") {
          if (
            typeof msg.data.playback_gen === "number" &&
            msg.data.playback_gen !== syncState.playbackGen
          ) {
            return;
          }
          if (typeof msg.data.ewma_latency_ms === "number") {
            syncState.ewmaLatencyMs = msg.data.ewma_latency_ms;
          }
          if (typeof msg.data.recommended_delay_ms === "number") {
            syncState.recommendedDelayMs = msg.data.recommended_delay_ms;
            if (activeTtsSources.size === 0) {
              applyEffectiveDelay(false);
            }
          }
        }
        return;
      }

      if (msg.type === "ws_tts_chunk" && msg.meta) {
        const meta = msg.meta;
        if (typeof meta.playback_gen === "number" && meta.playback_gen !== syncState.playbackGen) {
          return;
        }
        const streamKey = getStreamKey(meta);
        if (typeof meta.ewma_latency_ms === "number") {
          syncState.ewmaLatencyMs = meta.ewma_latency_ms;
        }
        if (typeof meta.recommended_delay_ms === "number") {
          syncState.recommendedDelayMs = meta.recommended_delay_ms;
        }
        if (!meta.is_final && msg.pcmB64 && settings.enabled && !syncState.desktopRxMuted) {
          if (droppedSeqs.has(streamKey)) {
            return;
          }
          const scheduled = scheduleRawPcmChunk(msg.pcmB64, meta);
          if (!scheduled) {
            return;
          }
          const engShort = meta.tts_engine === "xtts_v2" ? "XTTS" : "KOKORO";
          if (meta.latency_ms) {
            currentOverlayStatus = `${meta.latency_ms}ms (${engShort})`;
            updateOverlayBadge();
          }
          if (meta.chunk_index === 0 && meta.translated_text) {
            showSubtitle(
              meta.src_text || "",
              meta.translated_text,
              meta.latency_ms,
              meta.speaker_label || "Locutor",
              engShort
            );
          }
        } else if (meta.is_final) {
          seqScheduleMap.delete(streamKey);
          seqRateMap.delete(streamKey);
          droppedSeqs.delete(streamKey);
          if (activeTtsSources.size === 0) {
            applyEffectiveDelay(false);
          }
        }
      }
    });

    port.onDisconnect.addListener(() => {
      if (wsPort === port) {
        wsPort = null;
        wsConnected = false;
      }
    });

    port.postMessage({ type: "ws_connect" });
    return port;
  }

  function closeStreamPort() {
    wsConnected = false;
    syncState.desktopRxMuted = false;
    if (wsPort) {
      try {
        wsPort.disconnect();
      } catch (_err) {
        // already disconnected
      }
      wsPort = null;
    }
  }

  function loadSettings() {
    chrome.storage.local.get(DEFAULT_SETTINGS, (stored) => {
      settings = { ...DEFAULT_SETTINGS, ...stored };
      applySettingsLive();
    });
  }

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local") return;
    let changed = false;
    let sourceChanged = false;
    for (const [key, val] of Object.entries(changes)) {
      if (key in DEFAULT_SETTINGS) {
        settings[key] = val.newValue;
        changed = true;
        if (key === "sourceLang") sourceChanged = true;
      }
    }
    if (sourceChanged) {
      requestYouTubeTracks(true);
    }
    if (changed) {
      applySettingsLive();
    }
  });

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || typeof msg.type !== "string") return false;
    if (msg.type === "PB_APPLY_SETTINGS" && msg.settings) {
      const srcChanged = msg.settings.sourceLang && msg.settings.sourceLang !== settings.sourceLang;
      settings = { ...settings, ...msg.settings };
      if (srcChanged) {
        requestYouTubeTracks(true);
      }
      applySettingsLive();
      sendResponse({ ok: true });
      return false;
    }
    if (msg.type === "PB_GET_STATUS") {
      const syncCfg = getEffectiveSyncConfig();
      sendResponse({
        ok: true,
        hasVideo: Boolean(findPrimaryVideo()),
        enabled: settings.enabled,
        rxMuted: syncState.desktopRxMuted,
        wsConnected,
        syncStrategy: syncCfg.strategy,
        effectiveDelayMs: syncState.desktopRxMuted ? 0 : syncCfg.delayMs,
        ewmaLatencyMs: Math.round(syncState.ewmaLatencyMs),
        prefetchedCount: syncState.prefetchedCues.length,
        status: currentOverlayStatus
      });
      return false;
    }
    return false;
  });

  function findPrimaryVideo() {
    const videos = Array.from(document.querySelectorAll("video"));
    if (videos.length === 0) return null;
    videos.sort((a, b) => {
      const aPlaying = !a.paused && !a.ended ? 1 : 0;
      const bPlaying = !b.paused && !b.ended ? 1 : 0;
      if (aPlaying !== bPlaying) return bPlaying - aPlaying;
      const aRect = a.getBoundingClientRect();
      const bRect = b.getBoundingClientRect();
      return bRect.width * bRect.height - aRect.width * aRect.height;
    });
    const best = videos[0];
    const rect = best.getBoundingClientRect();
    if (rect.width < 120 || rect.height < 80) {
      return videos.find((v) => !v.paused) || null;
    }
    return best;
  }

  function ensureOverlay() {
    const targetVideo = findPrimaryVideo();
    if (!targetVideo && !settings.enabled) {
      if (overlayRoot) {
        overlayRoot.style.display = "none";
      }
      return;
    }

    if (!overlayRoot) {
      overlayRoot = document.createElement("div");
      overlayRoot.id = "pb-media-overlay";
      Object.assign(overlayRoot.style, {
        position: "fixed",
        right: "18px",
        bottom: "64px",
        zIndex: "2147483647",
        maxWidth: "460px",
        minWidth: "220px",
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        color: "#f8fafc",
        pointerEvents: "none",
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: "8px"
      });

      overlaySubtitleBox = document.createElement("div");
      Object.assign(overlaySubtitleBox.style, {
        display: "none",
        width: "100%",
        background: "rgba(15, 23, 42, 0.86)",
        border: "1px solid rgba(148, 163, 184, 0.24)",
        borderRadius: "14px",
        padding: "10px 14px",
        boxShadow: "0 12px 32px rgba(0, 0, 0, 0.45)",
        backdropFilter: "blur(12px)"
      });

      const bar = document.createElement("div");
      Object.assign(bar.style, {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        pointerEvents: "auto"
      });

      overlayStatusBtn = document.createElement("button");
      overlayStatusBtn.type = "button";
      Object.assign(overlayStatusBtn.style, {
        cursor: "pointer",
        background: "rgba(15, 23, 42, 0.82)",
        color: "#e2e8f0",
        border: "1px solid rgba(148, 163, 184, 0.28)",
        borderRadius: "9999px",
        padding: "6px 14px",
        fontSize: "12px",
        fontWeight: "600",
        letterSpacing: "0.01em",
        fontFamily: "inherit",
        backdropFilter: "blur(10px)",
        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.35)",
        transition: "all 0.15s ease"
      });

      overlayStatusBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        const nextEnabled = !settings.enabled;
        settings.enabled = nextEnabled;
        chrome.storage.local.set({ enabled: nextEnabled });
        applySettingsLive();
      });

      overlayResetBtn = document.createElement("button");
      overlayResetBtn.type = "button";
      overlayResetBtn.title = "Reiniciar voces de hablantes";
      overlayResetBtn.textContent = "↻";
      Object.assign(overlayResetBtn.style, {
        cursor: "pointer",
        background: "rgba(15, 23, 42, 0.82)",
        color: "#cbd5e1",
        border: "1px solid rgba(148, 163, 184, 0.28)",
        borderRadius: "9999px",
        width: "28px",
        height: "28px",
        display: "none",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "13px",
        fontFamily: "inherit",
        backdropFilter: "blur(10px)",
        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.35)"
      });
      overlayResetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        clearPlaybackQueue();
        sendWsCommand({ type: "reset_speakers" });
        updateOverlayBadge("Voces reiniciadas");
      });

      bar.appendChild(overlayStatusBtn);
      bar.appendChild(overlayResetBtn);
      overlayRoot.appendChild(overlaySubtitleBox);
      overlayRoot.appendChild(bar);
    }

    const hostParent =
      document.fullscreenElement ||
      document.querySelector(".html5-video-player") ||
      document.body;

    if (hostParent && overlayRoot.parentElement !== hostParent) {
      overlayRoot.style.position = hostParent === document.body ? "fixed" : "absolute";
      hostParent.appendChild(overlayRoot);
    }

    overlayRoot.style.display = targetVideo || settings.enabled ? "flex" : "none";
    updateOverlayBadge();
  }

  function updateOverlayBadge(extraLabel = "") {
    if (!overlayStatusBtn) return;
    if (overlayResetBtn) {
      overlayResetBtn.style.display = settings.enabled ? "inline-flex" : "none";
    }
    const langCode = (settings.targetLang || "es").toUpperCase();
    if (settings.enabled) {
      if (syncState.desktopRxMuted) {
        overlayStatusBtn.style.background = "rgba(127, 29, 29, 0.85)";
        overlayStatusBtn.style.borderColor = "rgba(251, 113, 133, 0.55)";
        overlayStatusBtn.style.color = "#ffe4e6";
        overlayStatusBtn.textContent = `Doblaje ${langCode} · En pausa`;
        return;
      }
      overlayStatusBtn.style.background = "rgba(8, 51, 68, 0.86)";
      overlayStatusBtn.style.borderColor = "rgba(34, 211, 238, 0.55)";
      overlayStatusBtn.style.color = "#ecfeff";
      overlayStatusBtn.textContent = extraLabel
        ? `Doblaje ${langCode} · ${extraLabel}`
        : `Doblaje ${langCode}`;
    } else {
      overlayStatusBtn.style.background = "rgba(15, 23, 42, 0.82)";
      overlayStatusBtn.style.borderColor = "rgba(148, 163, 184, 0.28)";
      overlayStatusBtn.style.color = "#e2e8f0";
      overlayStatusBtn.textContent = "Traducir video";
    }
  }

  function showSubtitle(srcText, translatedText, _latencyMs, speakerLabel = "", _engineTag = "") {
    ensureOverlay();
    if (!overlaySubtitleBox) return;

    overlaySubtitleBox.innerHTML = "";

    if (speakerLabel) {
      const tagEl = document.createElement("div");
      Object.assign(tagEl.style, {
        fontSize: "11px",
        fontWeight: "600",
        color: "#7dd3fc",
        marginBottom: "3px"
      });
      tagEl.textContent = speakerLabel;
      overlaySubtitleBox.appendChild(tagEl);
    }

    if (srcText) {
      const origEl = document.createElement("div");
      Object.assign(origEl.style, {
        fontSize: "12px",
        color: "#cbd5e1",
        opacity: "0.78",
        marginBottom: "4px",
        lineHeight: "1.35"
      });
      origEl.textContent = srcText;
      overlaySubtitleBox.appendChild(origEl);
    }

    const tgtEl = document.createElement("div");
    Object.assign(tgtEl.style, {
      fontSize: "14px",
      fontWeight: "600",
      color: "#f8fafc",
      lineHeight: "1.4"
    });
    tgtEl.textContent = translatedText;
    overlaySubtitleBox.appendChild(tgtEl);

    overlaySubtitleBox.style.display = "block";
    if (subtitleHideTimer) clearTimeout(subtitleHideTimer);
    subtitleHideTimer = setTimeout(() => {
      if (overlaySubtitleBox) {
        overlaySubtitleBox.style.display = "none";
      }
    }, 5500);
  }

  function resetSpeechBuffers() {
    prerollChunks = [];
    prerollLen = 0;
    speechChunks = [];
    speechLen = 0;
    isSpeaking = false;
    silenceAccumMs = 0;
  }

  function clearPlaybackQueue() {
    syncState.playbackGen += 1;
    if (duckRestoreTimer) {
      clearTimeout(duckRestoreTimer);
      duckRestoreTimer = null;
    }
    for (const src of activeTtsSources) {
      try {
        src.onended = null;
        src.stop();
      } catch (_e) {
        // already stopped
      }
    }
    activeTtsSources.clear();
    seqScheduleMap.clear();
    seqRateMap.clear();
    droppedSeqs.clear();
    nextScheduledTime = 0;
    restoreBackgroundGain();
    resetSpeechBuffers();
  }

  function resetPlaybackState({ resetSpeakers = false, clearPrefetchCues = false } = {}) {
    clearPlaybackQueue();
    if (captionTimer) {
      clearTimeout(captionTimer);
      captionTimer = null;
    }
    lastCaptionSent = "";
    lastCaptionSeen = "";
    lastCaptionActiveTs = 0;
    syncState.sentPrefetchKeys.clear();
    if (clearPrefetchCues) {
      syncState.prefetchedCues = [];
      syncState.prefetchedVideoKey = "";
    }
    syncState.lastVideoTimeSec = activeVideo ? activeVideo.currentTime : null;
    const graph = activeVideo ? boundVideos.get(activeVideo) : audioGraph;
    if (graph) {
      updateWorkletEnabled(graph);
    }
    sendWsCommand({
      type: resetSpeakers ? "reset_speakers" : "reset_playback",
      playback_gen: syncState.playbackGen,
      current_time_sec: activeVideo ? activeVideo.currentTime : null
    });
  }

  function setDesktopRxMuted(muted) {
    const nextMuted = Boolean(muted);
    const changed = syncState.desktopRxMuted !== nextMuted;
    syncState.desktopRxMuted = nextMuted;
    if (nextMuted) {
      if (changed) {
        resetPlaybackState({ resetSpeakers: false, clearPrefetchCues: false });
      }
      applyEffectiveDelay(true);
      if (overlaySubtitleBox) {
        overlaySubtitleBox.style.display = "none";
      }
      currentOverlayStatus = "RX MUTE (STUDIO)";
      updateOverlayBadge("RX MUTE");
    } else if (changed) {
      const graph = activeVideo ? boundVideos.get(activeVideo) : audioGraph;
      if (graph) {
        updateWorkletEnabled(graph);
      }
      if (settings.enabled) {
        applyEffectiveDelay(true);
        sendMediaClock(true);
        tickAutoPrefetchAndClock();
        currentOverlayStatus = wsConnected ? "WS ACTIVO" : "CONECTANDO";
        updateOverlayBadge();
      }
    }
  }

  function applyDuckGain() {
    if (!audioGraph) return;
    if (duckRestoreTimer) {
      clearTimeout(duckRestoreTimer);
      duckRestoreTimer = null;
    }
    const { ctx, duckGain } = audioGraph;
    const linear = Math.pow(10, Number(settings.duckDb || -14) / 20);
    try {
      duckGain.gain.cancelScheduledValues(ctx.currentTime);
      duckGain.gain.setTargetAtTime(linear, ctx.currentTime, 0.04);
    } catch (_e) {
      duckGain.gain.value = linear;
    }
  }

  function restoreBackgroundGain() {
    if (!audioGraph) return;
    const { ctx, duckGain } = audioGraph;
    try {
      duckGain.gain.cancelScheduledValues(ctx.currentTime);
      duckGain.gain.setTargetAtTime(1.0, ctx.currentTime, 0.16);
    } catch (_e) {
      duckGain.gain.value = 1.0;
    }
  }

  function scheduleRestoreIfIdle() {
    if (duckRestoreTimer) clearTimeout(duckRestoreTimer);
    duckRestoreTimer = setTimeout(() => {
      duckRestoreTimer = null;
      if (activeTtsSources.size === 0) {
        restoreBackgroundGain();
      }
    }, 180);
  }

  function updateWorkletEnabled(graph) {
    if (!graph || !graph.workletNode) return;
    const active = Boolean(
      settings.enabled &&
      !syncState.desktopRxMuted &&
      activeVideo === graph.video &&
      !graph.video.paused &&
      !graph.video.ended &&
      !graph.video.seeking
    );
    try {
      graph.workletNode.port.postMessage({ type: "setEnabled", enabled: active });
    } catch (_err) {
      // node detached
    }
  }

  function shouldSendPcmToWhisper() {
    const syncCfg = getEffectiveSyncConfig();
    if (syncCfg.strategy === "PREFETCH_ZERO_LAG") {
      return false;
    }
    if (settings.useCaptions && Date.now() - lastCaptionActiveTs < 4000) {
      return false;
    }
    return true;
  }

  function computeAudioStreamMode() {
    const needSttAudio = shouldSendPcmToWhisper();
    const voiceClean = String(settings.voice || "auto").trim().toLowerCase();
    const needVoiceProbe = Boolean(
      settings.diarization !== false &&
      (voiceClean === "" || voiceClean === "auto" || voiceClean === "xtts" || voiceClean === "cinematic")
    );
    return {
      needSttAudio,
      needVoiceProbe,
      voiceProbeOnly: !needSttAudio
    };
  }

  function syncVoiceProbeModeIfNeeded() {
    if (!wsConnected || !wsPort) return;
    const { voiceProbeOnly } = computeAudioStreamMode();
    if (syncState.lastSentVoiceProbeOnly !== voiceProbeOnly) {
      sendWsConfigure();
    }
  }

  function attachFallbackProcessor(graph) {
    if (graph.fallbackProcessor) return;
    const { ctx, source, silentSink, video } = graph;
    const processor = ctx.createScriptProcessor(2048, 2, 1);
    source.connect(processor);
    processor.connect(silentSink);
    processor.onaudioprocess = (event) => {
      if (
        !settings.enabled ||
        syncState.desktopRxMuted ||
        activeVideo !== video ||
        video.paused ||
        video.ended ||
        video.seeking
      ) {
        return;
      }
      const { needSttAudio, needVoiceProbe } = computeAudioStreamMode();
      if (!needSttAudio && !(wsConnected && wsPort && needVoiceProbe)) {
        return;
      }
      const inBuf = event.inputBuffer;
      const nCh = inBuf.numberOfChannels;
      if (nCh === 0) return;

      const left = inBuf.getChannelData(0);
      let mono = left;
      if (nCh > 1) {
        const right = inBuf.getChannelData(1);
        mono = new Float32Array(left.length);
        for (let i = 0; i < left.length; i++) {
          mono[i] = 0.5 * (left[i] + right[i]);
        }
      }
      const pcm16 = downsampleTo16kInt16(mono, inBuf.sampleRate);
      if (wsConnected && wsPort) {
        syncVoiceProbeModeIfNeeded();
        for (let off = 0; off + FRAME_SAMPLES <= pcm16.length; off += FRAME_SAMPLES) {
          const frameBytes = new Uint8Array(
            pcm16.buffer,
            pcm16.byteOffset + off * 2,
            FRAME_SAMPLES * 2
          );
          wsPort.postMessage({ type: "pcm_frame", b64: uint8ToBase64(frameBytes) });
        }
      } else if (needSttAudio && wsErrorCount >= 2) {
        processCapturedFrameHttp(mono, inBuf.sampleRate);
      }
    };
    graph.fallbackProcessor = processor;
  }

  function bindVideoAudio(video) {
    if (!video) return null;
    if (boundVideos.has(video)) {
      const existing = boundVideos.get(video);
      audioGraph = existing;
      if (existing.ctx.state === "suspended") {
        existing.ctx.resume().catch(() => {});
      }
      updateWorkletEnabled(existing);
      return existing;
    }

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaElementSource(video);
      const delayNode = ctx.createDelay(5.0);
      const duckGain = ctx.createGain();
      const voiceGain = ctx.createGain();
      const silentSink = ctx.createGain();
      silentSink.gain.value = 0.0;

      // Rama B: Audio del video con retardo Lookahead + Smart Ducking
      source.connect(delayNode);
      delayNode.connect(duckGain);
      duckGain.connect(ctx.destination);

      // Rama D: Salida de voz traducida directo a destino
      voiceGain.connect(ctx.destination);
      silentSink.connect(ctx.destination);

      const graph = {
        video,
        ctx,
        source,
        delayNode,
        duckGain,
        voiceGain,
        silentSink,
        workletNode: null,
        fallbackProcessor: null
      };

      const workletUrl = chrome.runtime.getURL("pcm_worklet.js");
      ctx.audioWorklet
        .addModule(workletUrl)
        .then(() => {
          if (boundVideos.get(video) !== graph) return;
          const workletNode = new AudioWorkletNode(ctx, "protobridge-capture-processor", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            processorOptions: {
              enabled: Boolean(settings.enabled && !video.paused && !video.ended && !video.seeking)
            }
          });
          workletNode.port.onmessage = (event) => {
            if (!(event.data instanceof ArrayBuffer)) return;
            if (
              !settings.enabled ||
              syncState.desktopRxMuted ||
              activeVideo !== video ||
              video.paused ||
              video.ended ||
              video.seeking
            ) {
              return;
            }
            const { needSttAudio, needVoiceProbe } = computeAudioStreamMode();
            if (!needSttAudio && !(wsConnected && wsPort && needVoiceProbe)) return;

            if (wsConnected && wsPort) {
              syncVoiceProbeModeIfNeeded();
              try {
                wsPort.postMessage({
                  type: "pcm_frame",
                  b64: uint8ToBase64(new Uint8Array(event.data))
                });
              } catch (_err) {
                wsConnected = false;
              }
            } else if (needSttAudio && wsErrorCount >= 2) {
              const i16 = new Int16Array(event.data);
              const f32 = new Float32Array(i16.length);
              for (let i = 0; i < i16.length; i++) {
                f32[i] = i16[i] / 32768.0;
              }
              processCapturedFrameHttp(f32, TARGET_RATE);
            }
          };
          source.connect(workletNode);
          workletNode.connect(silentSink);
          graph.workletNode = workletNode;
          updateWorkletEnabled(graph);
        })
        .catch(() => {
          attachFallbackProcessor(graph);
        });

      video.addEventListener("play", () => {
        if (ctx.state === "suspended") {
          ctx.resume().catch(() => {});
        }
        updateWorkletEnabled(graph);
        requestYouTubeTracks(false);
        if (settings.enabled) {
          sendMediaClock(true);
          tickAutoPrefetchAndClock();
        }
      });
      video.addEventListener("playing", () => {
        syncState.lastVideoTimeSec = video.currentTime;
        updateWorkletEnabled(graph);
        if (settings.enabled) {
          sendMediaClock(true);
          tickAutoPrefetchAndClock();
        }
      });
      video.addEventListener("canplay", () => {
        if (settings.enabled && !video.paused && !video.seeking) {
          sendMediaClock(true);
          tickAutoPrefetchAndClock();
        }
      });
      video.addEventListener("pause", () => {
        updateWorkletEnabled(graph);
        resetPlaybackState({ resetSpeakers: false, clearPrefetchCues: false });
      });
      video.addEventListener("seeking", () => {
        resetPlaybackState({ resetSpeakers: false, clearPrefetchCues: false });
      });
      video.addEventListener("seeked", () => {
        syncState.sentPrefetchKeys.clear();
        syncState.lastVideoTimeSec = video.currentTime;
        updateWorkletEnabled(graph);
        if (settings.enabled) {
          sendMediaClock(true);
          tickAutoPrefetchAndClock();
          if (settings.useCaptions) {
            pollCaptionsTick();
          }
        }
      });
      video.addEventListener("emptied", () => {
        updateWorkletEnabled(graph);
        resetPlaybackState({ resetSpeakers: true, clearPrefetchCues: true });
      });

      boundVideos.set(video, graph);
      audioGraph = graph;
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }
      return graph;
    } catch (err) {
      currentOverlayStatus = "ERROR AUDIO";
      updateOverlayBadge(`ERROR CORS/AUDIO: ${err.message}`);
      return null;
    }
  }

  function scheduleRawPcmChunk(pcmB64, meta) {
    if (
      !audioGraph ||
      !settings.enabled ||
      syncState.desktopRxMuted ||
      !activeVideo ||
      activeVideo.seeking
    ) {
      return false;
    }
    if (typeof meta.playback_gen === "number" && meta.playback_gen !== syncState.playbackGen) {
      return false;
    }
    const streamKey = getStreamKey(meta);
    if (droppedSeqs.has(streamKey)) return false;

    const { ctx, voiceGain } = audioGraph;
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    const buf = base64ToArrayBuffer(pcmB64);
    const pcmI16 = new Int16Array(buf);
    if (pcmI16.length === 0) return false;

    const rate = Math.max(8000, Math.min(48000, Number(meta.sample_rate || 24000)));
    const audioBuf = ctx.createBuffer(1, pcmI16.length, rate);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < pcmI16.length; i++) {
      ch[i] = pcmI16[i] / 32768.0;
    }

    const fadeSamples = Math.min(Math.floor(rate * 0.002), Math.floor(ch.length / 4));
    for (let i = 0; i < fadeSamples; i++) {
      const w = i / fadeSamples;
      ch[i] *= w;
      ch[ch.length - 1 - i] *= w;
    }

    if (meta.chunk_index > 0 && !seqScheduleMap.has(streamKey)) {
      droppedSeqs.add(streamKey);
      return false;
    }
    let startAt = Math.max(ctx.currentTime + 0.02, seqScheduleMap.get(streamKey) || nextScheduledTime);
    let playRate = seqRateMap.get(streamKey) || 1.0;

    if (meta.chunk_index === 0) {
      const activeDelaySec = Math.max(0.0, syncState.lastAppliedDelaySec);
      if (meta.is_prefetched && typeof meta.play_at_sec === "number") {
        const durSec =
          typeof meta.duration_sec === "number" && meta.duration_sec > 0 ? meta.duration_sec : 1.5;
        const leadSec = meta.play_at_sec + activeDelaySec - activeVideo.currentTime;
        const endLeadSec = meta.play_at_sec + durSec + activeDelaySec - activeVideo.currentTime;
        if (endLeadSec < -1.2 || leadSec < -Math.max(2.0, durSec + 0.9) || leadSec > 12.0) {
          droppedSeqs.add(streamKey);
          return false;
        }
        const targetTime = ctx.currentTime + Math.max(0.02, leadSec);
        if (targetTime < nextScheduledTime && nextScheduledTime - targetTime > 0.25) {
          const behindSec = nextScheduledTime - targetTime;
          playRate = Math.max(1.0, Math.min(1.18, 1.0 + behindSec * 0.08));
        }
        startAt = Math.max(ctx.currentTime + 0.02, targetTime, nextScheduledTime);
      } else if (typeof meta.capture_media_sec === "number") {
        const capDelay = Number(meta.capture_delay_sec || activeDelaySec);
        const targetSceneTime = meta.capture_media_sec + capDelay;
        const lateness = activeVideo.currentTime - targetSceneTime;
        if (lateness > 4.2 || lateness < -(capDelay + 5.5)) {
          droppedSeqs.add(streamKey);
          return false;
        }
        if (lateness > 0.65) {
          playRate = Math.max(1.0, Math.min(1.18, 1.0 + lateness * 0.07));
        }
        const targetTime = ctx.currentTime + Math.max(0.02, -lateness);
        startAt = Math.max(ctx.currentTime + 0.02, targetTime, nextScheduledTime);
      }
      if (startAt > ctx.currentTime + 14.0) {
        droppedSeqs.add(streamKey);
        return false;
      }
      seqRateMap.set(streamKey, playRate);
    }

    const effectiveDur = audioBuf.duration / playRate;
    const endAt = startAt + effectiveDur;
    seqScheduleMap.set(streamKey, endAt);
    nextScheduledTime = Math.min(ctx.currentTime + 16.0, Math.max(nextScheduledTime, endAt));

    applyDuckGain();

    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.playbackRate.value = playRate;
    src.connect(voiceGain);
    activeTtsSources.add(src);
    src.onended = () => {
      activeTtsSources.delete(src);
      if (activeTtsSources.size === 0) {
        scheduleRestoreIfIdle();
      }
    };
    src.start(startAt);
    return true;
  }

  function downsampleTo16kInt16(floatSamples, inputRate) {
    if (inputRate <= TARGET_RATE) {
      const out = new Int16Array(floatSamples.length);
      for (let i = 0; i < floatSamples.length; i++) {
        const s = Math.max(-1, Math.min(1, floatSamples[i]));
        out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
      }
      return out;
    }

    const ratio = inputRate / TARGET_RATE;
    const outLen = Math.floor(floatSamples.length / ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(floatSamples.length, Math.floor((i + 1) * ratio));
      let acc = 0;
      let cnt = 0;
      for (let j = start; j < end; j++) {
        acc += floatSamples[j];
        cnt++;
      }
      const avg = cnt > 0 ? acc / cnt : floatSamples[start];
      const clamped = Math.max(-1, Math.min(1, avg));
      out[i] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    }
    return out;
  }

  function int16ChunksToBase64(chunks, totalSamples) {
    const merged = new Int16Array(totalSamples);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return uint8ToBase64(new Uint8Array(merged.buffer));
  }

  function processCapturedFrameHttp(monoFloat, sampleRate) {
    const pcm16 = downsampleTo16kInt16(monoFloat, sampleRate);
    if (pcm16.length === 0) return;

    let sumSq = 0;
    for (let i = 0; i < monoFloat.length; i++) {
      const v = monoFloat[i];
      sumSq += v * v;
    }
    const rmsDbfs = 20 * Math.log10(Math.sqrt(sumSq / monoFloat.length) + 1e-9);
    const frameMs = (pcm16.length / TARGET_RATE) * 1000;

    if (!isSpeaking) {
      prerollChunks.push(pcm16);
      prerollLen += pcm16.length;
      while (prerollLen > PREROLL_SAMPLES && prerollChunks.length > 1) {
        const dropped = prerollChunks.shift();
        prerollLen -= dropped.length;
      }
      if (rmsDbfs >= MIN_SPEECH_RMS_DBFS) {
        isSpeaking = true;
        speechChunks = [...prerollChunks];
        speechLen = prerollLen;
        prerollChunks = [];
        prerollLen = 0;
        silenceAccumMs = 0;
      }
      return;
    }

    speechChunks.push(pcm16);
    speechLen += pcm16.length;
    if (rmsDbfs >= MIN_SPEECH_RMS_DBFS) {
      silenceAccumMs = 0;
    } else {
      silenceAccumMs += frameMs;
    }

    const reachedEndOfPhrase =
      silenceAccumMs >= SILENCE_HANGOVER_MS && speechLen >= MIN_PHRASE_SAMPLES;
    const reachedMaxChunk = speechLen >= MAX_PHRASE_SAMPLES;

    if (reachedEndOfPhrase || reachedMaxChunk) {
      const chunksToSend = speechChunks;
      const lenToSend = speechLen;
      resetSpeechBuffers();
      if (lenToSend >= MIN_PHRASE_SAMPLES) {
        sendHttpFallbackTranslate({
          pcm_s16le_b64: int16ChunksToBase64(chunksToSend, lenToSend),
          sample_rate: TARGET_RATE
        });
      }
    }
  }

  function sendHttpFallbackTranslate(partialPayload) {
    if (!settings.enabled || inFlightCount >= MAX_IN_FLIGHT) {
      return;
    }
    inFlightCount++;
    currentOverlayStatus = "TRADUCIENDO GPU...";
    updateOverlayBadge("HTTP...");

    const payload = {
      source_lang: settings.sourceLang,
      target_lang: settings.targetLang,
      voice: settings.voice,
      tts_backend: settings.ttsBackend || "xtts_v2",
      diarization: settings.diarization !== false,
      speed: settings.speed,
      ...partialPayload
    };

    chrome.runtime.sendMessage({ type: "PB_TRANSLATE_MEDIA", payload }, (response) => {
      inFlightCount = Math.max(0, inFlightCount - 1);
      if (chrome.runtime.lastError || !response || !response.ok) {
        currentOverlayStatus = "ESPERANDO APP";
        updateOverlayBadge("ABRE PROTOBRIDGE");
        return;
      }
      const res = response.data;
      if (!res || !res.ok) {
        currentOverlayStatus = "ERROR MOTOR";
        updateOverlayBadge("ERROR IA");
        return;
      }
      if (res.empty) {
        currentOverlayStatus = "ACTIVO";
        updateOverlayBadge();
        return;
      }

      const engShort = res.tts_engine === "xtts_v2" ? "XTTS" : "KOKORO";
      currentOverlayStatus = `${res.latency_ms}ms (${engShort})`;
      updateOverlayBadge();
      showSubtitle(
        res.src_text,
        res.translated_text,
        res.latency_ms,
        res.speaker_label || "Locutor",
        engShort
      );
      if (settings.enabled && res.wav_base64) {
        enqueueTtsWav(res.wav_base64);
      }
    });
  }

  function enqueueTtsWav(wavBase64) {
    if (!audioGraph) return;
    const { ctx, voiceGain } = audioGraph;
    const capturedGen = syncState.playbackGen;
    ctx
      .decodeAudioData(base64ToArrayBuffer(wavBase64))
      .then((audioBuffer) => {
        if (!settings.enabled || capturedGen !== syncState.playbackGen) return;
        const startAt = Math.max(ctx.currentTime + 0.02, nextScheduledTime);
        nextScheduledTime = startAt + audioBuffer.duration;
        applyDuckGain();
        const src = ctx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(voiceGain);
        activeTtsSources.add(src);
        src.onended = () => {
          activeTtsSources.delete(src);
          if (activeTtsSources.size === 0) {
            scheduleRestoreIfIdle();
          }
        };
        src.start(startAt);
      })
      .catch(() => {
        // Ignorar bloque WAV corrupto
      });
  }

  function sendCaptionForTranslation(text, playAtSec = null, durationSec = null, isPrefetched = false) {
    const clean = (text || "").replace(/\s+/g, " ").trim();
    if (clean.length < 3) return;
    if (wsConnected && wsPort) {
      syncState.captionSeq += 1;
      sendWsCommand({
        type: "translate_caption",
        seq: syncState.captionSeq,
        playback_gen: syncState.playbackGen,
        text: clean,
        source_lang: settings.sourceLang,
        target_lang: settings.targetLang,
        voice: settings.voice,
        tts_backend: settings.ttsBackend || "xtts_v2",
        speed: Number(settings.speed || 1.08),
        play_at_sec: playAtSec,
        duration_sec: durationSec,
        is_prefetched: isPrefetched
      });
    } else if (!isPrefetched) {
      sendHttpFallbackTranslate({ text: clean });
    }
  }

  function sendMediaClock(force = false) {
    if (!wsConnected || !wsPort || !activeVideo) return;
    const now = Date.now();
    if (!force && now - syncState.lastClockSentTs < 240) return;
    syncState.lastClockSentTs = now;
    sendWsCommand({
      type: "media_clock",
      current_time_sec: activeVideo.currentTime,
      delay_sec: Math.max(0.0, syncState.lastAppliedDelaySec)
    });
  }

  function tickAutoPrefetchAndClock() {
    if (!settings.enabled || syncState.desktopRxMuted || !activeVideo) return;
    if (activeVideo.seeking) {
      applyEffectiveDelay(false);
      return;
    }
    if (activeVideo.paused || activeVideo.ended) return;

    const nowSec = activeVideo.currentTime;
    if (
      syncState.lastVideoTimeSec !== null &&
      (nowSec < syncState.lastVideoTimeSec - 1.2 || nowSec - syncState.lastVideoTimeSec > 4.0)
    ) {
      resetPlaybackState({ resetSpeakers: false, clearPrefetchCues: false });
    }
    syncState.lastVideoTimeSec = nowSec;

    hydrateHtml5TextTracksIfEmpty();
    syncVoiceProbeModeIfNeeded();
    sendMediaClock(false);

    const syncCfg = getEffectiveSyncConfig();
    applyEffectiveDelay(false);
    if (syncCfg.strategy !== "PREFETCH_ZERO_LAG" || syncState.prefetchedCues.length === 0) {
      return;
    }

    const horizonSec = syncCfg.horizonSec;

    for (const cue of syncState.prefetchedCues) {
      if (cue.endSec < nowSec - 0.15) continue;
      if (cue.startSec > nowSec + horizonSec) break;
      const key = `${Math.round(cue.startSec * 10)}:${cue.text}`;
      if (syncState.sentPrefetchKeys.has(key)) continue;
      syncState.sentPrefetchKeys.add(key);
      if (syncState.sentPrefetchKeys.size > 500) {
        syncState.sentPrefetchKeys.clear();
      }
      lastCaptionActiveTs = Date.now();
      sendCaptionForTranslation(
        cue.text,
        cue.startSec,
        Math.max(0.45, cue.endSec - cue.startSec),
        true
      );
    }
  }

  function readVisibleYouTubeCaptions() {
    const segs = Array.from(document.querySelectorAll(".ytp-caption-segment"));
    if (segs.length === 0) return "";
    return segs
      .map((el) => (el.textContent || "").trim())
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function pollCaptionsTick() {
    if (
      !settings.enabled ||
      syncState.desktopRxMuted ||
      !settings.useCaptions ||
      !activeVideo ||
      activeVideo.paused ||
      activeVideo.seeking ||
      activeVideo.ended
    ) {
      return;
    }
    const syncCfg = getEffectiveSyncConfig();
    if (syncCfg.strategy === "PREFETCH_ZERO_LAG") {
      return;
    }
    const text = readVisibleYouTubeCaptions();
    if (!text) return;

    lastCaptionActiveTs = Date.now();
    if (text === lastCaptionSeen) return;
    lastCaptionSeen = text;

    if (captionTimer) clearTimeout(captionTimer);
    const scheduledGen = syncState.playbackGen;
    captionTimer = setTimeout(() => {
      if (
        !settings.enabled ||
        syncState.desktopRxMuted ||
        !settings.useCaptions ||
        scheduledGen !== syncState.playbackGen ||
        !activeVideo ||
        activeVideo.seeking ||
        activeVideo.paused
      ) {
        return;
      }
      const latest = readVisibleYouTubeCaptions();
      if (!latest || latest === lastCaptionSent) return;

      let delta = latest;
      if (lastCaptionSent) {
        if (latest.startsWith(lastCaptionSent)) {
          delta = latest.slice(lastCaptionSent.length).trim();
        } else {
          const prevWords = lastCaptionSent.split(" ");
          const curWords = latest.split(" ");
          let overlap = 0;
          const maxCheck = Math.min(prevWords.length, curWords.length);
          for (let k = maxCheck; k >= 1; k--) {
            let match = true;
            for (let i = 0; i < k; i++) {
              if (prevWords[prevWords.length - k + i] !== curWords[i]) {
                match = false;
                break;
              }
            }
            if (match) {
              overlap = k;
              break;
            }
          }
          if (overlap > 0) {
            delta = curWords.slice(overlap).join(" ").trim();
          }
        }
      }
      lastCaptionSent = latest;
      if (delta.length >= 3) {
        sendCaptionForTranslation(delta, null, null, false);
      }
    }, 280);
  }

  function applySettingsLive() {
    const video = findPrimaryVideo();
    if (video) {
      activeVideo = video;
    }

    if (settings.enabled) {
      ensureStreamPort();
      sendWsConfigure();
      requestYouTubeTracks(false);
      if (activeVideo) {
        const graph = bindVideoAudio(activeVideo);
        if (graph) {
          applyEffectiveDelay(true);
          updateWorkletEnabled(graph);
          if (activeTtsSources.size > 0) {
            applyDuckGain();
          } else {
            restoreBackgroundGain();
          }
        }
      }
      currentOverlayStatus = syncState.desktopRxMuted
        ? "RX MUTE (STUDIO)"
        : wsConnected
          ? "WS ACTIVO"
          : "CONECTANDO";
    } else {
      if (audioGraph) {
        syncState.lastAppliedDelaySec = 0.0;
        try {
          audioGraph.delayNode.delayTime.setTargetAtTime(0.0, audioGraph.ctx.currentTime, 0.05);
        } catch (_e) {
          audioGraph.delayNode.delayTime.value = 0.0;
        }
        updateWorkletEnabled(audioGraph);
      }
      clearPlaybackQueue();
      closeStreamPort();
      if (overlaySubtitleBox) {
        overlaySubtitleBox.style.display = "none";
      }
      currentOverlayStatus = "LISTO";
    }

    ensureOverlay();
  }

  const domObserver = new MutationObserver(() => {
    const v = findPrimaryVideo();
    if (v && v !== activeVideo) {
      activeVideo = v;
      if (settings.enabled) {
        applySettingsLive();
      } else {
        ensureOverlay();
      }
    }
    if (settings.enabled && settings.useCaptions) {
      pollCaptionsTick();
    }
  });

  if (document.documentElement) {
    domObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
      characterData: true
    });
  }

  document.addEventListener("fullscreenchange", ensureOverlay);
  window.addEventListener("yt-navigate-finish", () => {
    resetPlaybackState({ resetSpeakers: true, clearPrefetchCues: true });
    setTimeout(() => {
      requestYouTubeTracks(true);
      applySettingsLive();
    }, 380);
  });

  setInterval(() => {
    if (settings.enabled) {
      tickAutoPrefetchAndClock();
    }
    const v = findPrimaryVideo();
    if (v !== activeVideo || (v && (!overlayRoot || overlayRoot.style.display === "none"))) {
      activeVideo = v;
      ensureOverlay();
      if (settings.enabled && v) {
        bindVideoAudio(v);
      }
    }
  }, 260);

  loadSettings();
})();
