#!/usr/bin/env python3
import base64
import copy
import io
import os
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from transformers import pipeline


BRAIN_BASE_URL = os.getenv("BRAIN_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
BRAIN_MODEL = os.getenv("BRAIN_MODEL", "").strip()
BRAIN_SYSTEM_PROMPT = os.getenv(
    "BRAIN_SYSTEM_PROMPT",
    "You are a helpful conversational assistant. Keep answers concise, clear, and natural for spoken playback.",
)
BRAIN_MAX_TOKENS = int(os.getenv("VOICE_MODE_BRAIN_MAX_TOKENS", "96"))
ASR_MODEL = os.getenv("VOICE_MODE_ASR_MODEL", "microsoft/VibeVoice-ASR-HF")
ASR_FALLBACK_MODEL = os.getenv("VOICE_MODE_ASR_FALLBACK_MODEL", "openai/whisper-small")
TTS_MODEL = os.getenv("VOICE_MODE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
TTS_CFG_SCALE = float(os.getenv("VOICE_MODE_TTS_CFG_SCALE", "1.5"))
TTS_MAX_NEW_TOKENS = int(os.getenv("VOICE_MODE_TTS_MAX_NEW_TOKENS", "512"))
TTS_VOICE_PROMPT_URL = os.getenv(
    "VOICE_MODE_TTS_VOICE_PROMPT_URL",
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model/en-Carter_man.pt",
)
TTS_VOICE_PROMPT_PATH = Path(
    os.getenv(
        "VOICE_MODE_TTS_VOICE_PROMPT_PATH",
        "/workspace/models/voice_prompts/en-Carter_man.pt",
    )
)
VOICE_MODE_HOST = os.getenv("VOICE_MODE_HOST", "0.0.0.0")
VOICE_MODE_PORT = int(os.getenv("VOICE_MODE_PORT", "9001"))


PAGE_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Voice Mode</title>
  <style>
    :root {
      --bg0: #0d1b1e;
      --bg1: #102a2f;
      --panel: rgba(12, 30, 34, 0.8);
      --line: rgba(139, 193, 177, 0.35);
      --txt: #e5f4f0;
      --accent: #5ee3b6;
      --danger: #ff8c7a;
      --muted: #9db8b1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at 20% 15%, #184750 0%, var(--bg1) 38%, var(--bg0) 100%);
      color: var(--txt);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      display: grid;
      place-items: center;
      padding: 16px;
    }
    .wrap {
      width: min(720px, 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      background: var(--panel);
      backdrop-filter: blur(8px);
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    h1 { margin: 0 0 8px; font-size: clamp(24px, 3vw, 34px); }
    p { margin: 0; color: var(--muted); }
    .row { margin-top: 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    button {
      border: none;
      border-radius: 12px;
      padding: 13px 18px;
      font-weight: 700;
      font-size: 16px;
      background: var(--accent);
      color: #04231d;
      cursor: pointer;
      transition: transform 0.14s ease, box-shadow 0.2s ease;
      box-shadow: 0 8px 22px rgba(94, 227, 182, 0.35);
    }
    button:hover { transform: translateY(-1px); }
    button.recording { background: var(--danger); color: #2a0700; box-shadow: 0 8px 22px rgba(255,140,122,0.38); }
    .status { font-size: 14px; color: var(--muted); }
    .box {
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(0,0,0,0.16);
    }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    .text { white-space: pre-wrap; line-height: 1.45; }
    audio { width: 100%; margin-top: 14px; }
  </style>
</head>
<body>
  <main class=\"wrap\">
    <h1>Voice Mode</h1>
    <p>Click once to start speaking. Click again to stop. You will get a spoken response after recording ends.</p>

    <div class=\"row\">
      <button id=\"talkBtn\">Start Speaking</button>
      <span class=\"status\" id=\"status\">Idle</span>
    </div>

    <div class=\"box\">
      <div class=\"label\">You said</div>
      <div class=\"text\" id=\"transcript\">-</div>
    </div>

    <div class=\"box\">
      <div class=\"label\">Assistant thought</div>
      <div class=\"text\" id=\"reply\">-</div>
    </div>

    <audio id=\"audioPlayer\" controls></audio>
  </main>

  <script>
    const talkBtn = document.getElementById('talkBtn');
    const statusEl = document.getElementById('status');
    const transcriptEl = document.getElementById('transcript');
    const replyEl = document.getElementById('reply');
    const audioPlayer = document.getElementById('audioPlayer');

    let mediaRecorder = null;
    let chunks = [];
    let stream = null;
    let recording = false;

    function setStatus(text) { statusEl.textContent = text; }

    function audioBufferToWavBlob(audioBuffer) {
      const numChannels = audioBuffer.numberOfChannels;
      const sampleRate = audioBuffer.sampleRate;
      const samples = audioBuffer.length;
      const bytesPerSample = 2;
      const blockAlign = numChannels * bytesPerSample;
      const buffer = new ArrayBuffer(44 + samples * blockAlign);
      const view = new DataView(buffer);

      let offset = 0;
      function writeString(s) {
        for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
        offset += s.length;
      }

      writeString('RIFF');
      view.setUint32(offset, 36 + samples * blockAlign, true); offset += 4;
      writeString('WAVE');
      writeString('fmt ');
      view.setUint32(offset, 16, true); offset += 4;
      view.setUint16(offset, 1, true); offset += 2;
      view.setUint16(offset, numChannels, true); offset += 2;
      view.setUint32(offset, sampleRate, true); offset += 4;
      view.setUint32(offset, sampleRate * blockAlign, true); offset += 4;
      view.setUint16(offset, blockAlign, true); offset += 2;
      view.setUint16(offset, 16, true); offset += 2;
      writeString('data');
      view.setUint32(offset, samples * blockAlign, true); offset += 4;

      const channels = [];
      for (let c = 0; c < numChannels; c++) channels.push(audioBuffer.getChannelData(c));
      for (let i = 0; i < samples; i++) {
        for (let c = 0; c < numChannels; c++) {
          const s = Math.max(-1, Math.min(1, channels[c][i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
          offset += 2;
        }
      }

      return new Blob([buffer], { type: 'audio/wav' });
    }

    async function webmToWavBlob(webmBlob) {
      const arr = await webmBlob.arrayBuffer();
      const audioCtx = new AudioContext();
      const decoded = await audioCtx.decodeAudioData(arr.slice(0));
      const wavBlob = audioBufferToWavBlob(decoded);
      await audioCtx.close();
      return wavBlob;
    }

    async function ensureRecorder() {
      if (mediaRecorder) return;
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? { mimeType: 'audio/webm;codecs=opus' }
        : {};
      mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
    }

    async function startRecording() {
      await ensureRecorder();
      chunks = [];
      mediaRecorder.start();
      recording = true;
      talkBtn.classList.add('recording');
      talkBtn.textContent = 'Stop Speaking';
      setStatus('Recording...');
    }

    async function stopRecording() {
      if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
      setStatus('Processing...');
      recording = false;
      talkBtn.classList.remove('recording');
      talkBtn.textContent = 'Start Speaking';

      const stopDone = new Promise((resolve) => {
        mediaRecorder.onstop = resolve;
      });
      mediaRecorder.stop();
      await stopDone;

      const webmBlob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      const wavBlob = await webmToWavBlob(webmBlob);

      const form = new FormData();
      form.append('audio_file', wavBlob, 'speech.wav');

      const response = await fetch('/api/voice-mode/respond', { method: 'POST', body: form });
      if (!response.ok) {
        const err = await response.text();
        throw new Error(err || 'Voice mode request failed');
      }

      const data = await response.json();
      transcriptEl.textContent = data.transcript || '-';
      replyEl.textContent = data.reply || '-';
      if (data.audio_b64) {
        audioPlayer.src = `data:${data.audio_mime || 'audio/wav'};base64,${data.audio_b64}`;
        await audioPlayer.play().catch(() => {});
      }

      setStatus('Done');
    }

    talkBtn.addEventListener('click', async () => {
      try {
        if (!recording) {
          await startRecording();
        } else {
          await stopRecording();
        }
      } catch (err) {
        setStatus('Error: ' + err.message);
      }
    });
  </script>
</body>
</html>
"""


class VoiceModeEngine:
    def __init__(self) -> None:
        self._asr = None
        self._tts = None
        self._tts_backend = ""
        self._model_id = BRAIN_MODEL
        self._lock = threading.Lock()

    @staticmethod
    def _is_unknown_model_type_error(exc: Exception, model_type: str) -> bool:
        message = str(exc).lower()
        return f"model type `{model_type}`" in message or f"model type '{model_type}'" in message

    @staticmethod
    def _pick_torch_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _pick_torch_dtype(device: str) -> torch.dtype:
        if device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _load_vibevoice_prompt(self, device: str) -> dict[str, Any]:
        prompt_path = TTS_VOICE_PROMPT_PATH
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        if not prompt_path.exists():
            response = requests.get(TTS_VOICE_PROMPT_URL, timeout=60)
            response.raise_for_status()
            prompt_path.write_bytes(response.content)

        prompt = torch.load(str(prompt_path), map_location=device, weights_only=False)
        if not isinstance(prompt, dict):
            raise RuntimeError("VibeVoice prompt file is invalid")
        return prompt

    def _init_vibevoice_tts_backend(self) -> None:
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        except Exception as exc:
            raise RuntimeError(
                "VibeVoice realtime backend is unavailable. Install with: "
                "pip install \"vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git\""
            ) from exc

        device = self._pick_torch_device()
        torch_dtype = self._pick_torch_dtype(device)
        attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"

        processor = VibeVoiceStreamingProcessor.from_pretrained(TTS_MODEL)
        try:
            if device == "mps":
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    TTS_MODEL,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                model.to("mps")
            else:
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    TTS_MODEL,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    attn_implementation=attn_impl,
                )
        except Exception:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                TTS_MODEL,
                torch_dtype=torch_dtype,
                device_map=(device if device in {"cuda", "cpu"} else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")

        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)

        self._tts = {
            "model": model,
            "processor": processor,
            "device": device,
            "prompt": self._load_vibevoice_prompt(device),
        }
        self._tts_backend = "vibevoice_realtime"

    def _ensure_models(self) -> None:
        if self._asr is not None and self._tts is not None:
            return
        with self._lock:
            if self._asr is None:
                try:
                    self._asr = pipeline(
                        "automatic-speech-recognition",
                        model=ASR_MODEL,
                        trust_remote_code=True,
                    )
                except Exception as exc:
                    if not self._is_unknown_model_type_error(exc, "vibevoice_asr"):
                        raise
                    self._asr = pipeline(
                        "automatic-speech-recognition",
                        model=ASR_FALLBACK_MODEL,
                    )
            if self._tts is None:
                try:
                    self._tts = pipeline(
                        "text-to-audio",
                        model=TTS_MODEL,
                        trust_remote_code=True,
                    )
                    self._tts_backend = "pipeline"
                except Exception as exc:
                    if not self._is_unknown_model_type_error(exc, "vibevoice_streaming"):
                        raise
                    self._init_vibevoice_tts_backend()

    def _resolve_model_id(self) -> str:
        if self._model_id:
            return self._model_id

        url = f"{BRAIN_BASE_URL}/models"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            raise RuntimeError("No models available from brain service")

        self._model_id = data[0].get("id")
        if not self._model_id:
            raise RuntimeError("Could not resolve model id from brain service")
        return self._model_id

    @staticmethod
    def _read_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if sampwidth != 2:
            raise RuntimeError("Only 16-bit PCM wav is supported")

        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, sample_rate

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_models()
        audio_array, sample_rate = self._read_wav(audio_bytes)
        result = self._asr({"array": audio_array, "sampling_rate": sample_rate})
        text = result.get("text", "").strip()
        if not text:
            raise RuntimeError("Speech could not be transcribed")
        return text

    def think(self, user_text: str) -> str:
        model_id = self._resolve_model_id()
        url = f"{BRAIN_BASE_URL}/chat/completions"
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": BRAIN_MAX_TOKENS,
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Brain response contained no choices")
        message = choices[0].get("message") or {}
        answer = (message.get("content") or "").strip()
        if not answer:
            raise RuntimeError("Brain response was empty")
        return answer

    @staticmethod
    def _to_wav_bytes(audio_array: np.ndarray, sampling_rate: int) -> bytes:
        clipped = np.clip(audio_array, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sampling_rate)
            wav.writeframes(pcm.tobytes())
        return out.getvalue()

    def speak(self, text: str) -> bytes:
        self._ensure_models()
        if self._tts_backend == "pipeline":
            tts_out = self._tts(text)
            audio = np.asarray(tts_out["audio"], dtype=np.float32)
            sr = int(tts_out["sampling_rate"])
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            return self._to_wav_bytes(audio, sr)

        if self._tts_backend == "vibevoice_realtime":
            model = self._tts["model"]
            processor = self._tts["processor"]
            prompt = self._tts["prompt"]
            device = self._tts["device"]

            inputs = processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=prompt,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=TTS_MAX_NEW_TOKENS,
                cfg_scale=TTS_CFG_SCALE,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prompt),
            )

            speech = outputs.speech_outputs[0]
            if torch.is_tensor(speech):
                speech = speech.detach().cpu().float().numpy()
            audio = np.asarray(speech, dtype=np.float32).squeeze()
            if audio.size == 0:
                raise RuntimeError("VibeVoice produced empty audio")
            return self._to_wav_bytes(audio, 24000)

        raise RuntimeError("No TTS backend initialized")


engine = VoiceModeEngine()
app = FastAPI(title="Voice Mode Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE_HTML


@app.post("/api/voice-mode/respond")
async def voice_mode_respond(audio_file: UploadFile = File(...)) -> dict[str, str]:
    try:
        payload = await audio_file.read()
        transcript = engine.transcribe(payload)
        reply = engine.think(transcript)
        audio_bytes = engine.speak(reply)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Brain request failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_mime": "audio/wav",
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=VOICE_MODE_HOST, port=VOICE_MODE_PORT)
