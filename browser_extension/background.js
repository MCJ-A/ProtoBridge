const CANDIDATE_PORTS = [8770, 8771, 8772, 8773, 8774, 8775, 8776];
let activePort = 8770;

async function fetchWithTimeout(url, options = {}, timeoutMs = 1500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function pingPort(port) {
  const resp = await fetchWithTimeout(
    `http://127.0.0.1:${port}/api/media/ping`,
    {
      method: "GET",
      headers: {
        "X-ProtoBridge-Client": "BrowserExtension"
      }
    },
    900
  );
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const data = await resp.json();
  if (!data || !data.ok) {
    throw new Error("Respuesta inválida de ProtoBridge");
  }
  return data;
}

async function findActivePort() {
  const ordered = [activePort, ...CANDIDATE_PORTS.filter((p) => p !== activePort)];
  for (const port of ordered) {
    try {
      const data = await pingPort(port);
      activePort = port;
      return { ok: true, port, data };
    } catch (_err) {
      // Siguiente puerto candidato
    }
  }
  return {
    ok: false,
    error: "No se detectó ProtoBridge.exe ejecutándose en 127.0.0.1:8770-8776."
  };
}

async function postMediaEndpoint(port, path, payload, timeoutMs = 18000) {
  const resp = await fetchWithTimeout(
    `http://127.0.0.1:${port}${path}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-ProtoBridge-Client": "BrowserExtension"
      },
      body: JSON.stringify(payload || {})
    },
    timeoutMs
  );
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return await resp.json();
}

async function sendWithPortRetry(path, payload, timeoutMs = 18000) {
  try {
    const data = await postMediaEndpoint(activePort, path, payload, timeoutMs);
    return { ok: true, port: activePort, data };
  } catch (_firstErr) {
    const found = await findActivePort();
    if (!found.ok) {
      return found;
    }
    try {
      const data = await postMediaEndpoint(found.port, path, payload, timeoutMs);
      return { ok: true, port: found.port, data };
    } catch (secondErr) {
      return { ok: false, error: String(secondErr) };
    }
  }
}

function base64ToUint8Array(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function uint8ArrayToBase64(bytes) {
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    const slice = bytes.subarray(i, Math.min(bytes.length, i + step));
    binary += String.fromCharCode.apply(null, slice);
  }
  return btoa(binary);
}

const UTF8_DECODER = new TextDecoder("utf-8");

function unpackMediaPacket(arrayBuffer) {
  if (!(arrayBuffer instanceof ArrayBuffer) || arrayBuffer.byteLength < 4) {
    return null;
  }
  const view = new DataView(arrayBuffer);
  const hdrLen = view.getUint32(0, true);
  if (hdrLen === 0 || 4 + hdrLen > arrayBuffer.byteLength) {
    return null;
  }
  const hdrBytes = new Uint8Array(arrayBuffer, 4, hdrLen);
  const meta = JSON.parse(UTF8_DECODER.decode(hdrBytes));
  const pcmBytes = new Uint8Array(arrayBuffer, 4 + hdrLen);
  const pcmB64 = pcmBytes.byteLength > 0 ? uint8ArrayToBase64(pcmBytes) : "";
  return { meta, pcmB64 };
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "pb-media-stream") {
    return;
  }

  let ws = null;
  let connecting = false;
  let disconnected = false;
  let lastConfigCmd = null;

  function safePost(msg) {
    if (disconnected) return;
    try {
      port.postMessage(msg);
    } catch (_err) {
      disconnected = true;
    }
  }

  async function ensureWebSocket() {
    if (disconnected || connecting) return;
    if (ws && ws.readyState === WebSocket.OPEN) return;

    connecting = true;
    try {
      const found = await findActivePort();
      if (!found.ok || disconnected) {
        safePost({ type: "ws_error", error: found.error || "ProtoBridge no detectado" });
        return;
      }

      const socket = new WebSocket(`ws://127.0.0.1:${found.port}/ws/media`);
      socket.binaryType = "arraybuffer";
      ws = socket;

      socket.onopen = () => {
        if (disconnected || ws !== socket) {
          try {
            socket.close();
          } catch (_e) {
            // ignore
          }
          return;
        }
        safePost({ type: "ws_open", port: found.port, info: found.data });
        if (lastConfigCmd) {
          try {
            socket.send(JSON.stringify(lastConfigCmd));
          } catch (_e) {
            // handled on close
          }
        }
      };

      socket.onmessage = async (event) => {
        if (disconnected || ws !== socket) return;
        if (typeof event.data === "string") {
          try {
            safePost({ type: "ws_json", data: JSON.parse(event.data) });
          } catch (_e) {
            // ignore malformed JSON
          }
          return;
        }
        try {
          const buf =
            event.data instanceof ArrayBuffer
              ? event.data
              : event.data && typeof event.data.arrayBuffer === "function"
                ? await event.data.arrayBuffer()
                : null;
          if (!buf) return;
          const unpacked = unpackMediaPacket(buf);
          if (unpacked) {
            safePost({ type: "ws_tts_chunk", ...unpacked });
          }
        } catch (err) {
          safePost({ type: "ws_error", error: String(err) });
        }
      };

      socket.onerror = () => {
        if (ws === socket) {
          safePost({ type: "ws_error", error: "Error en WebSocket /ws/media" });
        }
      };

      socket.onclose = () => {
        if (ws === socket) {
          ws = null;
          safePost({ type: "ws_close" });
        }
      };
    } finally {
      connecting = false;
    }
  }

  port.onMessage.addListener((msg) => {
    if (!msg || typeof msg !== "object" || disconnected) return;

    if (msg.type === "ws_connect") {
      void ensureWebSocket();
      return;
    }

    if (msg.type === "ws_cmd" && msg.cmd && typeof msg.cmd === "object") {
      if (msg.cmd.type === "configure") {
        lastConfigCmd = { ...(lastConfigCmd || {}), ...msg.cmd };
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(msg.cmd));
        } catch (_e) {
          void ensureWebSocket();
        }
      } else {
        void ensureWebSocket();
      }
      return;
    }

    if (msg.type === "pcm_frame" && typeof msg.b64 === "string" && msg.b64.length > 0) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(base64ToUint8Array(msg.b64));
        } catch (_e) {
          void ensureWebSocket();
        }
      } else {
        void ensureWebSocket();
      }
    }
  });

  port.onDisconnect.addListener(() => {
    disconnected = true;
    if (ws) {
      try {
        ws.close();
      } catch (_e) {
        // ignore
      }
      ws = null;
    }
  });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message.type !== "string") {
    return false;
  }

  if (message.type === "PB_PING") {
    findActivePort()
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "PB_TRANSLATE_MEDIA") {
    sendWithPortRetry("/api/media/translate", message.payload || {}, 18000)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "PB_SET_CHANNELS") {
    sendWithPortRetry("/api/media/channels", message.payload || {}, 2500)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});
