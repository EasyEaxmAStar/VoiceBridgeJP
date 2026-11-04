"""Resident native Python 3.11 inference worker; framed PCM JSON travels on pipes.

Only this child imports GPT-SoVITS dependencies. Stdout is reserved for IPC;
upstream prints/progress and tracebacks go to stderr. No network listener.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    # Duplicate the pipe before redirecting fd 1 as well as Python stdout.
    # Native phonemizer libraries can use C printf, bypassing sys.stdout.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(value):
        wire.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
        wire.flush()

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf8"))
        vendor = Path(config["root"]).resolve()
        manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf8"))
        if manifest["version"] != "v2ProPlus" or manifest["sovits"] != "v2Pro/s2Gv2ProPlus.pth":
            raise ValueError("Only the pinned v2 ProPlus manifest is accepted")
        import subprocess
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
        if revision != manifest["source_revision"]:
            raise ValueError("GPT-SoVITS source revision differs from the pinned manifest")
        weights = vendor / "GPT_SoVITS/pretrained_models"
        for artifact in manifest["artifacts"]:
            path = weights / artifact["path"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError(f"Checkpoint integrity mismatch: {artifact['path']}")
        from ort.pipeline.personal_voice import load_personal_profile,check_personal_reference,personal_metadata
        personal=None
        if config.get('personal_profile'):
            personal=load_personal_profile(config['personal_profile'],base_manifest=config['manifest'])
            if personal['profile_sha256']!=config['personal_profile_sha256']:
                raise ValueError('Personal deployment profile changed during worker startup')
            check_personal_reference(personal,config['reference_wav'],config['reference_text'],config['reference_language'])
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        os.chdir(vendor)
        sys.path[:0] = [str(vendor), str(vendor / "GPT_SoVITS")]
        # The upstream Chinese frontend imports jieba_fast unconditionally.
        # Use its API-compatible Python implementation on Windows without MSVC.
        import jieba
        import jieba.posseg
        sys.modules["jieba_fast"] = jieba
        sys.modules["jieba_fast.posseg"] = jieba.posseg
        import numpy as np
        import torch
        import socket

        def offline_only(*args, **kwargs):
            raise RuntimeError("Network connections are disabled in the local GPT-SoVITS inference worker")

        socket.socket.connect = offline_only
        socket.socket.connect_ex = offline_only
        from ort.pipeline.gptsovits_reference import validate_reference
        import shutil
        import tools.portable_runtime as portable
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required; install it explicitly before starting the worker")
        portable.FFMPEG_EXE = Path(ffmpeg)
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        # Even all_ja routes Latin words through the English frontend. Validate
        # its offline data before allocating models rather than fail mid-session.
        import nltk
        nltk.data.path.insert(0, str(PROJECT / ".model-cache/gptsovits-nltk"))
        try:
            for resource in ("corpora/cmudict.zip", "taggers/averaged_perceptron_tagger.zip",
                             "taggers/averaged_perceptron_tagger_eng/"):
                nltk.data.find(resource)
            from text import english
            english._g2p("Python MySQL 2026")
        except LookupError as exc:
            raise RuntimeError("Missing offline English phonemizer data; run "
                               ".venv/Scripts/python.exe tools/setup_gptsovits_nltk.py") from exc
        from ort.gptsovits_pcm import validate_pcm
        from ort.tts_controls import controls, BoundaryPCM, DEFAULT_SPEED
        from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast
        from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
        import fast_langdetect.infer as language_detection
        # split-lang requests the large detector even for forced Japanese.
        # Explicitly use the official package's bundled small model, avoiding
        # any attempted runtime network download or classifier fallback.
        language_detection._default_detector.config.custom_model_path = str(language_detection._LOCAL_SMALL_MODEL_PATH)
        language_detection._default_detector.config.allow_fallback = False

        reference = validate_reference(config["reference_wav"], config["reference_text"], config["reference_language"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is disabled")
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        settings = {"device": "cuda", "is_half": True, "version": "v2ProPlus",
                    "t2s_weights_path": str(weights / manifest["gpt"]),
                    "vits_weights_path": str(weights / manifest["sovits"]),
                    "bert_base_path": str(weights / "chinese-roberta-wwm-ext-large"),
                    "cnhuhbert_base_path": str(weights / "chinese-hubert-base")}
        if personal:
            settings.update(t2s_weights_path=personal['gpt']['path'],vits_weights_path=personal['sovits']['path'])
        for key in ("t2s_weights_path", "vits_weights_path", "bert_base_path", "cnhuhbert_base_path"):
            if not Path(settings[key]).exists():
                raise ValueError(f"Required artifact missing: {settings[key]}")
        tts_config = TTS_Config({"custom": settings})
        tts_config.configs_path = str(Path(args.config).with_suffix(".upstream.yaml"))
        pipeline = TTS(tts_config)
        identity = get_sovits_version_from_path_fast(settings["vits_weights_path"])
        if pipeline.configs.version != "v2ProPlus" or identity[1] != "v2ProPlus" or pipeline.configs.use_vocoder:
            raise RuntimeError(f"Loaded checkpoint is not v2 ProPlus: {identity}")
        if personal:
            # TTS_Config exposes the paths actually initialized by official TTS.
            if Path(pipeline.configs.t2s_weights_path).resolve()!=Path(personal['gpt']['path']).resolve() or Path(pipeline.configs.vits_weights_path).resolve()!=Path(personal['sovits']['path']).resolve():
                raise RuntimeError('Upstream substituted personal checkpoint paths; inference rejected')
        if next(pipeline.vits_model.parameters()).device.type != "cuda":
            raise RuntimeError("Upstream silently selected a non-CUDA device")
        if next(pipeline.vits_model.parameters()).dtype != torch.float16:
            raise RuntimeError("FP16 model initialization failed")
        if next(pipeline.t2s_model.parameters()).dtype != torch.float16:
            raise RuntimeError("GPT FP16 model initialization failed")
        language = {"ja": "all_ja", "en": "en", "zh": "all_zh"}[reference.language]
        conditioning_start = time.perf_counter()
        with torch.inference_mode():
            pipeline.set_ref_audio(reference.path)
            prompt_text = reference.text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "." if language == "en" else "。"
            phones, bert, normalized = pipeline.text_preprocessor.segment_and_extract_feature_for_text(
                prompt_text, language, pipeline.configs.version)
            pipeline.prompt_cache.update(prompt_text=prompt_text, prompt_lang=language,
                                         phones=phones, bert_features=bert, norm_text=normalized)
            # The pinned upstream caches semantic codes/specs/text, but computes
            # its speaker embedding for every fragment. Reuse that pure result
            # only for the exact resident static-reference tensor.
            ref_tensor = pipeline.prompt_cache["refer_spec"][0][1]
            original_embedding = pipeline.sv_model.compute_embedding3
            embedding = original_embedding(ref_tensor)
            embedding_hits = [0]

            def static_embedding(wav):
                if wav is ref_tensor:
                    embedding_hits[0] += 1
                    return embedding
                return original_embedding(wav)

            pipeline.sv_model.compute_embedding3 = static_embedding
            pipeline._streaming_ref_model = id(pipeline.vits_model)
        torch.cuda.synchronize(0)
        ready = {"type": "ready", "version": pipeline.configs.version,
                 "source_revision": revision, "model_revision": manifest["model_revision"],
                 "gpt_checkpoint": manifest["gpt"], "sovits_checkpoint": manifest["sovits"],
                 "checkpoint_identity": list(identity), "device": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "dtype": "float16", "sample_rate": pipeline.configs.sampling_rate,
                 "initialization_seconds": time.perf_counter() - started,
                 "conditioning_seconds": time.perf_counter() - conditioning_start,
                 "reference_fingerprint": reference.fingerprint,
                 "conditioning_builds": 1, "pid": os.getpid(), "offline_network_guard": True}
        ready.update(initialization_peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                     initialization_peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
                     inference_path="official native streaming mode 2; CUDA graphs and FlashAttention disabled")
        if personal:
            ready.update(gpt_checkpoint=settings['t2s_weights_path'],sovits_checkpoint=settings['vits_weights_path'],
                personal_voice=personal_metadata(personal))
        cache_identity = tuple(id(pipeline.prompt_cache[key]) for key in ("prompt_semantic", "refer_spec", "phones", "bert_features"))
        frontend_trace = []
        segmentation_trace = []
        acoustic_trace = []
        if config.get("frontend_diagnostics", False):
            from text.symbols2 import symbols as phone_symbols
            original_clean = pipeline.text_preprocessor.clean_text_inf

            def observed_clean(text, language, version="v2"):
                result = original_clean(text, language, version)
                if len(frontend_trace) >= 256:
                    raise RuntimeError("Frontend diagnostics exceed the bounded 256 segment limit")
                ids = [int(i) for i in result[0]]
                frontend_trace.append({"text": text, "language": language,
                    "normalized_text": result[2], "phone_ids": ids,
                    "phones": [phone_symbols[i] for i in ids]})
                return result

            # Instance-local observation of the actual frontend calls, after
            # static conditioning. Does not change results or global G2P state.
            pipeline.text_preprocessor.clean_text_inf = observed_clean
            original_segment=pipeline.text_preprocessor.pre_seg_text
            def observed_segment(*args,**kwargs):
                result=original_segment(*args,**kwargs)
                if len(segmentation_trace)+len(result)>256:raise RuntimeError('Segmentation diagnostics exceeded bound')
                segmentation_trace.extend(result)
                return result
            pipeline.text_preprocessor.pre_seg_text=observed_segment
            original_decode=pipeline.vits_model.decode_streaming
            def observed_decode(*args,**kwargs):
                result=original_decode(*args,**kwargs)
                if len(acoustic_trace)>=256:raise RuntimeError('Acoustic diagnostics exceeded bound')
                acoustic_trace.append(dict(semantic_context_tokens=int(args[0].shape[-1]),
                    result_length=kwargs.get('result_length'),speed=float(kwargs.get('speed',1.)),
                    audio_samples=int(result[0].shape[-1]),
                    overlap_latent_frames=int(kwargs['overlap_frames'].shape[-1]) if kwargs.get('overlap_frames') is not None else 0))
                return result
            pipeline.vits_model.decode_streaming=observed_decode
        send(ready)
        requests = queue.Queue(maxsize=4)
        shutting_down = threading.Event()
        active_id = [None]

        def listen():
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    if message.get("type") == "cancel":
                        # A late cancel for the preceding sentence must not
                        # interrupt the next resident-worker request.
                        if message.get("id") == active_id[0]:
                            pipeline.stop_flag = True
                    elif message.get("type") == "shutdown":
                        pipeline.stop_flag = True
                        shutting_down.set()
                        return
                    else:
                        requests.put(message, timeout=1)
                except Exception:
                    pipeline.stop_flag = True
                    shutting_down.set()
                    return
            pipeline.stop_flag = True
            shutting_down.set()

        threading.Thread(target=listen, daemon=True, name="gptsovits-controls").start()
        while not shutting_down.is_set():
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            request_id = request.get("id")
            active_id[0] = request_id
            try:
                if request.get("type") != "synthesize" or not isinstance(request.get("text"), str):
                    raise ValueError("Invalid synthesis request")
                if request.get("language") != "ja":
                    raise ValueError("Production output language is Japanese (ja)")
                text_language = request.get("text_language", "all_ja")
                if text_language not in {"all_ja", "ja"}:
                    raise ValueError("Unsupported Japanese frontend mode")
                frontend_trace.clear()
                segmentation_trace.clear();acoustic_trace.clear()
                current = validate_reference(reference.path, reference.text, reference.language)
                if current.fingerprint != reference.fingerprint:
                    raise ValueError("Static reference changed; restart with the selected reference")
                inputs = {"text": request["text"], "text_lang": text_language, "ref_audio_path": reference.path,
                          "prompt_text": reference.text, "prompt_lang": language, "text_split_method": "cut5",
                          "batch_size": 1, "split_bucket": False, "seed": 42, "top_k": 15,
                          "top_p": 1.0, "temperature": 1.0, "speed_factor": DEFAULT_SPEED,
                          "fragment_interval": 0.3, "parallel_infer": True,
                          # Official api_v2 streaming_mode=2 maps to this native
                          # generator path; it needs no AR accelerator extension.
                          "streaming_mode": True, "fixed_length_chunk": False,
                          "gpt_streaming": False, "min_chunk_length": 16,
                          "overlap_length": 2, "use_cuda_graph": False, "use_flash_attention": False}
                options = controls(request.get("options"))
                from ort.tts_sampling import sampling_parameters
                sampling=sampling_parameters(request.get('sampling'))
                if sampling['seed']<0:raise ValueError('Sampling seed must be frozen by parent per utterance')
                inputs.update(sampling)
                from ort.tts_generation import generation_parameters
                generation=generation_parameters(request.get('generation'))
                inputs.update(generation)
                inputs.update(speed_factor=options["speed_factor"],fragment_interval=options["fragment_interval"])
                processor = BoundaryPCM(ready["sample_rate"],options["trim_digital_silence"])
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                started = time.perf_counter()
                first = None
                digest = hashlib.sha256()
                total_samples = 0
                chunks = 0
                prior_embedding_hits = embedding_hits[0]
                with torch.inference_mode():
                    def processed():
                        from ort.tts_random import isolated_utterance_rng
                        # The resident child owns one serialized model stack.
                        # Scope the upstream seed once around the ENTIRE iterator,
                        # never around individual PCM or semantic chunks.
                        with isolated_utterance_rng():
                            for position, (rate, samples) in enumerate(pipeline.run(inputs)):
                                samples = validate_pcm(rate, samples, ready["sample_rate"], position)
                                output = processor.add(samples)
                                if processor.metrics["raw_samples"] > 120 * rate:
                                    raise RuntimeError("Raw synthesis exceeds the bounded 120 second output limit")
                                yield rate, output
                        yield ready["sample_rate"], processor.add(np.zeros(0,dtype=np.int16),final=True)
                    for rate, samples in processed():
                        if not samples.size:
                            continue
                        torch.cuda.synchronize(0)
                        if first is None:
                            first = time.perf_counter() - started
                        pcm = samples.astype("<i2", copy=False).tobytes()
                        digest.update(pcm)
                        total_samples += samples.size
                        if total_samples > 120 * rate:
                            raise RuntimeError("Synthesis exceeds the bounded 120 second output limit")
                        send({"type": "chunk", "id": request_id, "sequence": chunks,
                              "sample_rate": int(rate), "samples": samples.size,
                              "sha256": hashlib.sha256(pcm).hexdigest(),
                              "pcm": base64.b64encode(pcm).decode("ascii")})
                        chunks += 1
                torch.cuda.synchronize(0)
                if first is None:
                    raise RuntimeError("GPT-SoVITS produced no playable PCM")
                send({"type": "done", "id": request_id, "chunks": chunks, "samples": total_samples,
                      "frontend_mode": text_language,"frontend_trace": frontend_trace,
                      "controls": options,"sampling":sampling,"generation":generation,
                      "text_segments":segmentation_trace,"acoustic_trace":acoustic_trace,
                      "effective_parallel_infer":generation['parallel_infer'] and not generation['streaming_mode'],
                      "effective_split_bucket":False,
                      "boundary_processing":{**processor.metrics,"raw_sha256":processor.raw_hash.hexdigest(),"processed_samples":total_samples},
                      "sha256": digest.hexdigest(), "synthesis_seconds": time.perf_counter() - started,
                      "first_playable_pcm_seconds": first, "conditioning_builds": 1,
                      "speaker_embedding_cache_hits": embedding_hits[0],
                      "speaker_embedding_reused": embedding_hits[0] > prior_embedding_hits,
                      "reference_features_reused": cache_identity == tuple(
                          id(pipeline.prompt_cache[key]) for key in ("prompt_semantic", "refer_spec", "phones", "bert_features")),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20})
            except Exception as exc:
                traceback.print_exc()
                send({"type": "error", "id": request_id, "message": str(exc)})
            finally:
                active_id[0] = None
    except Exception as exc:
        traceback.print_exc()
        send({"type": "error", "id": None, "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
