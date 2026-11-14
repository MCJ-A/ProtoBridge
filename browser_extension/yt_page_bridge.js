(() => {
  if (window.__pbYtPageBridgeLoaded) {
    return;
  }
  window.__pbYtPageBridgeLoaded = true;

  let preferredLang = "auto";
  let lastFetchedKey = "";

  function getCurrentVideoId() {
    try {
      const url = new URL(window.location.href);
      if (url.searchParams.has("v")) {
        return url.searchParams.get("v") || "";
      }
      const m = url.pathname.match(/\/shorts\/([^/?]+)/);
      if (m) return m[1];
    } catch (_e) {
      // ignore invalid URL
    }
    return "";
  }

  function decodeHtmlEntities(str) {
    return String(str || "")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
  }

  function parseTimedTextXml(xmlText) {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xmlText, "text/xml");
      const nodes = Array.from(doc.querySelectorAll("text, p"));
      const events = [];
      for (const node of nodes) {
        const rawStart = node.getAttribute("start") || node.getAttribute("t");
        const rawDur = node.getAttribute("dur") || node.getAttribute("d") || "2.0";
        if (rawStart === null) continue;
        const isMs = node.hasAttribute("t");
        const tStartMs = isMs ? Number(rawStart) : Math.round(Number(rawStart) * 1000);
        const dDurationMs = isMs ? Number(rawDur) : Math.round(Number(rawDur) * 1000);
        const text = decodeHtmlEntities(node.textContent || "").replace(/\s+/g, " ").trim();
        if (!Number.isFinite(tStartMs) || !text) continue;
        events.push({
          tStartMs,
          dDurationMs: Number.isFinite(dDurationMs) && dDurationMs > 0 ? dDurationMs : 2000,
          segs: [{ utf8: text }]
        });
      }
      return events;
    } catch (_err) {
      return [];
    }
  }

  function emitEvents(videoId, lang, vssId, events) {
    if (!Array.isArray(events) || events.length === 0) return;
    window.postMessage(
      {
        source: "pb-yt-bridge",
        type: "PB_TIMEDTEXT_JSON3",
        videoId: videoId || getCurrentVideoId(),
        lang: String(lang || ""),
        vssId: String(vssId || ""),
        events
      },
      window.location.origin
    );
  }

  function handleTimedTextPayload(urlStr, textBody) {
    if (!textBody) return;
    let lang = "";
    let vssId = "";
    let videoId = getCurrentVideoId();
    try {
      const u = new URL(urlStr, window.location.origin);
      lang = u.searchParams.get("tlang") || u.searchParams.get("lang") || "";
      videoId = u.searchParams.get("v") || videoId;
      vssId = u.searchParams.get("kind") === "asr" ? `a.${lang}` : `.${lang}`;
    } catch (_e) {
      // ignore
    }

    const trimmed = textBody.trim();
    if (trimmed.startsWith("{")) {
      try {
        const data = JSON.parse(trimmed);
        if (data && Array.isArray(data.events)) {
          emitEvents(videoId, lang, vssId, data.events);
        }
      } catch (_err) {
        // invalid json
      }
      return;
    }

    if (trimmed.startsWith("<")) {
      const events = parseTimedTextXml(trimmed);
      emitEvents(videoId, lang, vssId, events);
    }
  }

  // Interceptar fetch() de YouTube para capturar pistas /api/timedtext con POT/firma válida
  const origFetch = window.fetch;
  if (typeof origFetch === "function") {
    window.fetch = async function (...args) {
      const resp = await origFetch.apply(this, args);
      try {
        const url =
          typeof args[0] === "string"
            ? args[0]
            : args[0] && typeof args[0].url === "string"
              ? args[0].url
              : "";
        if (url.includes("/api/timedtext")) {
          const clone = resp.clone();
          clone
            .text()
            .then((body) => handleTimedTextPayload(url, body))
            .catch(() => {});
        }
      } catch (_err) {
        // pass-through transparent
      }
      return resp;
    };
  }

  // Interceptar XMLHttpRequest por si el reproductor usa XHR en /api/timedtext
  const origXhrOpen = XMLHttpRequest.prototype.open;
  const origXhrSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__pbUrl = typeof url === "string" ? url : String(url || "");
    return origXhrOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    if (this.__pbUrl && this.__pbUrl.includes("/api/timedtext")) {
      this.addEventListener("load", () => {
        if (this.status === 200 && typeof this.responseText === "string") {
          handleTimedTextPayload(this.__pbUrl, this.responseText);
        }
      });
    }
    return origXhrSend.apply(this, args);
  };

  function getPlayerResponse() {
    try {
      const player = document.getElementById("movie_player") || document.querySelector(".html5-video-player");
      if (player && typeof player.getPlayerResponse === "function") {
        const pr = player.getPlayerResponse();
        if (pr) return pr;
      }
    } catch (_e) {
      // fallback to window global
    }
    if (window.ytInitialPlayerResponse && typeof window.ytInitialPlayerResponse === "object") {
      return window.ytInitialPlayerResponse;
    }
    return null;
  }

  function pickBestTrack(tracks, targetPref) {
    if (!Array.isArray(tracks) || tracks.length === 0) return null;
    const pref = String(targetPref || "auto").toLowerCase();
    const isAsr = (t) => t.kind === "asr" || String(t.vssId || "").startsWith("a.");
    const langOf = (t) => String(t.languageCode || "").toLowerCase();

    if (pref && pref !== "auto") {
      const exactManual = tracks.find((t) => !isAsr(t) && langOf(t).startsWith(pref));
      if (exactManual) return exactManual;
      const exactAsr = tracks.find((t) => langOf(t).startsWith(pref));
      if (exactAsr) return exactAsr;
    }

    const manualEn = tracks.find((t) => !isAsr(t) && langOf(t).startsWith("en"));
    if (manualEn) return manualEn;
    const anyManual = tracks.find((t) => !isAsr(t));
    if (anyManual) return anyManual;
    const asrEn = tracks.find((t) => langOf(t).startsWith("en"));
    return asrEn || tracks[0];
  }

  async function prefetchFromPlayerResponse(force = false) {
    const videoId = getCurrentVideoId();
    const pr = getPlayerResponse();
    if (!pr || !pr.captions || !pr.captions.playerCaptionsTracklistRenderer) {
      return;
    }
    const tracks = pr.captions.playerCaptionsTracklistRenderer.captionTracks;
    const chosen = pickBestTrack(tracks, preferredLang);
    if (!chosen || !chosen.baseUrl) return;

    const key = `${videoId}:${chosen.vssId || chosen.languageCode}:${preferredLang}`;
    if (!force && key === lastFetchedKey) return;
    lastFetchedKey = key;

    try {
      const u = new URL(chosen.baseUrl, window.location.origin);
      u.searchParams.set("fmt", "json3");
      const resp = await origFetch.call(window, u.toString(), { credentials: "include" });
      if (!resp.ok) return;
      const raw = await resp.text();
      handleTimedTextPayload(u.toString(), raw);
    } catch (_err) {
      // Si la URL firmada expira/falla, el interceptor de fetch/XHR la capturará al activarse CC
    }
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const msg = event.data;
    if (!msg || msg.source !== "pb-content") return;
    if (msg.type === "PB_YT_REQUEST_TRACKS") {
      if (msg.preferredLang) {
        preferredLang = String(msg.preferredLang);
      }
      void prefetchFromPlayerResponse(Boolean(msg.force));
    }
  });

  window.addEventListener("yt-navigate-finish", () => {
    lastFetchedKey = "";
    setTimeout(() => void prefetchFromPlayerResponse(false), 500);
  });
  window.addEventListener("yt-page-data-updated", () => {
    setTimeout(() => void prefetchFromPlayerResponse(false), 400);
  });

  setTimeout(() => void prefetchFromPlayerResponse(false), 800);
})();
