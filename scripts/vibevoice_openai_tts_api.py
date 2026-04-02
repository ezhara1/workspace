#!/usr/bin/env python3
import copy
import io
import os
import re
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


HOST = os.getenv("VIBEVOICE_API_HOST", "0.0.0.0")
STRIP_THINK_FOR_TTS = os.getenv("VIBEVOICE_STRIP_THINK_FOR_TTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUIRE_CUDA = os.getenv("VIBEVOICE_REQUIRE_CUDA", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PORT = int(os.getenv("VIBEVOICE_API_PORT", "9101"))
TTS_MODEL = os.getenv("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
TTS_CFG_SCALE = float(os.getenv("VIBEVOICE_TTS_CFG_SCALE", "1.5"))
TTS_MAX_NEW_TOKENS = int(os.getenv("VIBEVOICE_TTS_MAX_NEW_TOKENS", "512"))
TTS_VOICE_PROMPT_URL = os.getenv(
    "VIBEVOICE_TTS_VOICE_PROMPT_URL",
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model/en-Carter_man.pt",
)
TTS_VOICE_PROMPT_PATH = Path(
    os.getenv(
        "VIBEVOICE_TTS_VOICE_PROMPT_PATH",
        "/workspace/models/voice_prompts/en-Carter_man.pt",
    )
)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str | None = "wav"


app = FastAPI(title="VibeVoice OpenAI-Compatible TTS API")
_tts_backend: dict[str, Any] | None = None
_tts_lock = threading.Lock()
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


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


def _strip_think_blocks(text: str) -> str:
    cleaned = THINK_BLOCK_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _pick_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if REQUIRE_CUDA:
        raise RuntimeError("CUDA is required for VibeVoice TTS, but no CUDA device is available")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_voice_prompt(device: str) -> dict[str, Any]:
    TTS_VOICE_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TTS_VOICE_PROMPT_PATH.exists():
        response = requests.get(TTS_VOICE_PROMPT_URL, timeout=60)
        response.raise_for_status()
        TTS_VOICE_PROMPT_PATH.write_bytes(response.content)

    prompt = torch.load(str(TTS_VOICE_PROMPT_PATH), map_location=device, weights_only=False)
    if not isinstance(prompt, dict):
        raise RuntimeError("VibeVoice prompt file is invalid")
    return prompt


def _ensure_tts_backend() -> dict[str, Any]:
    global _tts_backend
    if _tts_backend is not None:
        return _tts_backend

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    device = _pick_torch_device()
    torch_dtype = _pick_torch_dtype(device)
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

    _tts_backend = {
        "model": model,
        "processor": processor,
        "device": device,
        "prompt": _load_voice_prompt(device),
    }
    return _tts_backend


@app.on_event("startup")
def warmup_tts_backend() -> None:
    _ensure_tts_backend()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": TTS_MODEL, "object": "model"}]}


@app.get("/v1/audio/models")
def audio_models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": TTS_MODEL, "object": "model"}]}


@app.get("/v1/audio/voices")
def audio_voices() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "alloy", "object": "voice", "name": "alloy"}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    raw_text = (req.input or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="'input' is required")
    text = _strip_think_blocks(raw_text) if STRIP_THINK_FOR_TTS else raw_text
    if not text:
        text = raw_text

    fmt = (req.response_format or "wav").lower()
    # OpenWebUI/OpenAI clients often default to mp3. We currently generate wav/pcm,
    # so normalize common compressed formats to wav for compatibility.
    if fmt in {"mp3", "opus", "aac", "flac"}:
        fmt = "wav"
    elif fmt not in {"wav", "pcm"}:
        fmt = "wav"

    try:
        with _tts_lock:
            backend = _ensure_tts_backend()
            model = backend["model"]
            processor = backend["processor"]
            prompt = backend["prompt"]
            device = backend["device"]

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
            speech_out = outputs.speech_outputs[0]
            if torch.is_tensor(speech_out):
                speech_out = speech_out.detach().cpu().float().numpy()
            audio = np.asarray(speech_out, dtype=np.float32).squeeze()
            if audio.size == 0:
                raise RuntimeError("VibeVoice produced empty audio")
            sr = 24000

        if fmt == "wav":
            wav_bytes = _to_wav_bytes(audio, sr)
            return Response(content=wav_bytes, media_type="audio/wav")

        # pcm s16le
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16).tobytes()
        return Response(content=pcm, media_type="application/octet-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
