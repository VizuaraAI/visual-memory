"""Modal app: visual memory telemetry on Zamba2-VL-7B.

Zamba2-VL-7B is a hybrid vision-language model: 81 decoder layers, of which 68
are Mamba2 (fixed-size recurrent state) and 13 are shared-attention "hybrid"
blocks at layer indices 6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77.
That 5.2:1 ratio makes it the most recurrence-dominated open VLM available.

Usage:
    modal run modal_vlab.py::download_model    # CPU, populates HF cache volume
    modal run modal_vlab.py::e0                # GPU, instrument gate
"""

from __future__ import annotations

import json
import logging

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Zyphra/Zamba2-VL-7B"
ZYPHRA_TRANSFORMERS = "git+https://github.com/Zyphra/transformers.git@zamba2-vl"

# Prebuilt wheels matching torch 2.7 / cu12 / cp311. Building these from source
# on Modal costs 20+ minutes and frequently fails; the previous project settled
# on the same approach for flash-attn.
FLASH_ATTN_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE"
    "-cp311-cp311-linux_x86_64.whl"
)

app = modal.App("visual-memory-lab")
hf_cache = modal.Volume.from_name("hf-cache-vlab", create_if_missing=True)
results_vol = modal.Volume.from_name("visual-memory-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1")
    .pip_install(
        "accelerate>=1.0",
        "einops",
        "safetensors",
        "hf_transfer",
        "huggingface_hub",
        "pillow",
        "requests",
        "numpy",
        "qwen-vl-utils==0.0.2",
    )
    .pip_install(ZYPHRA_TRANSFORMERS)
    .pip_install(FLASH_ATTN_WHL)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/hf",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir("lab", "/root/lab")
)

# The second-family work needs stock recent transformers rather than the Zyphra
# fork this project pins for Zamba2-VL, because Bamba and Granite 4.0 were added
# upstream after that fork. Keeping them in separate images avoids disturbing a
# working environment for the experiments already collected.
image_hybrid = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1")
    .pip_install(
        "transformers>=4.56",
        "accelerate>=1.0",
        "einops",
        "safetensors",
        "hf_transfer",
        "huggingface_hub",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/hf",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir("lab", "/root/lab")
)

VOLUMES = {"/hf": hf_cache, "/results": results_vol}

# COCO val2017 image 39769: two cats on a pink couch with two remotes.
# Same corpus the real experiments use, and it serves without a User-Agent gate.
SMOKE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


@app.function(image=image, volumes=VOLUMES, timeout=7200)
def download_model(model_id: str = MODEL_ID) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    hf_cache.commit()
    logger.info("downloaded to %s", path)
    return path


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=3600)
def e0() -> str:
    """Instrument gate.

    Answers the questions the rest of the project depends on:
      1. Does the Zyphra fork load the model at all, without mamba-ssm?
      2. What is the cache object, and what are its state attributes called?
      3. What are the shapes of the Mamba2 recurrent states and attention KV?
      4. How many tokens does one image occupy?
      5. Does the model produce a sane answer about the image?
    """
    import io

    import requests
    import torch
    from transformers import AutoProcessor
    from transformers.models.zamba2_vl.modeling_zamba2_vl import (
        Zamba2_VLForConditionalGeneration,
    )

    report: dict = {}

    from PIL import Image as PILImage

    logger.info("loading processor")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("loading model")
    model = Zamba2_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    text_cfg = model.config.text_config
    block_types = list(text_cfg.layers_block_type)
    report["n_layers"] = len(block_types)
    report["n_mamba"] = sum(1 for t in block_types if t == "mamba")
    report["n_hybrid"] = sum(1 for t in block_types if t == "hybrid")
    report["hybrid_layer_ids"] = list(text_cfg.hybrid_layer_ids)
    report["mamba_d_state"] = text_cfg.mamba_d_state
    report["mamba_headdim"] = text_cfg.mamba_headdim
    report["n_mamba_heads"] = text_cfg.n_mamba_heads
    report["model_class"] = type(model).__name__

    logger.info("fetching smoke image")
    resp = requests.get(
        SMOKE_IMAGE_URL, timeout=60, headers={"User-Agent": "visual-memory-lab/0.1"}
    )
    resp.raise_for_status()
    raw = resp.content
    pil = PILImage.open(io.BytesIO(raw)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What animal is in this picture? Answer in one word."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt").to("cuda")
    report["prompt_token_count"] = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    cache = out.past_key_values
    report["cache_class"] = type(cache).__name__
    report["cache_attrs"] = sorted(
        a for a in dir(cache) if not a.startswith("_") and not callable(getattr(cache, a, None))
    )

    def describe(obj, name):
        if obj is None:
            return {"name": name, "kind": "None"}
        if torch.is_tensor(obj):
            return {"name": name, "kind": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, (list, tuple)):
            info = {"name": name, "kind": type(obj).__name__, "len": len(obj)}
            for i, el in enumerate(obj):
                if torch.is_tensor(el):
                    info[f"elem_{i}_shape"] = list(el.shape)
                    break
            return info
        if isinstance(obj, dict):
            info = {"name": name, "kind": "dict", "len": len(obj)}
            for k, v in obj.items():
                if torch.is_tensor(v):
                    info["sample_key"] = str(k)
                    info["sample_shape"] = list(v.shape)
                    break
            return info
        return {"name": name, "kind": type(obj).__name__}

    state_report = []
    for attr in ("ssm_states", "conv_states", "key_cache", "value_cache", "layers"):
        state_report.append(describe(getattr(cache, attr, None), attr))
    report["cache_states"] = state_report

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    answer = processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    report["smoke_answer"] = answer.strip()

    report["cuda_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    with open("/results/e0_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()

    logger.info("E0 REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=3600)
def e1() -> str:
    """Second gate: the internals we need to build a readout.

    The Mamba2 recurrence is h_t = h_{t-1} * exp(dt A) + dt B_t x_t^T and
    y_t = C_t^T h_t, so a readout of "what is in the state" needs the write
    pair (B, x) and the read vector C. This dumps the mixer module tree and
    confirms that the recurrent state actually responds to the image.
    """
    import io

    import requests
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor
    from transformers.models.zamba2_vl.modeling_zamba2_vl import (
        Zamba2_VLForConditionalGeneration,
    )

    report: dict = {}

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Zamba2_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # Locate the decoder layer list and dump one mamba and one hybrid layer.
    lm = model.language_model if hasattr(model, "language_model") else model.model
    report["top_level_children"] = [n for n, _ in model.named_children()]
    report["lm_type"] = type(lm).__name__
    report["lm_children"] = [n for n, _ in lm.named_children()]

    layers = None
    for candidate in (lm, getattr(lm, "model", None)):
        if candidate is not None and hasattr(candidate, "layers"):
            layers = candidate.layers
            break
    report["n_decoder_layers"] = len(layers) if layers is not None else None

    if layers is not None:
        report["layer0_type"] = type(layers[0]).__name__
        report["layer0_tree"] = [n for n, _ in layers[0].named_modules()][:40]
        report["layer6_type"] = type(layers[6]).__name__
        report["layer6_tree"] = [n for n, _ in layers[6].named_modules()][:40]
        # Parameter shapes on the mamba mixer tell us the projection layout.
        mixer = None
        for name, mod in layers[0].named_modules():
            if "mamba" in type(mod).__name__.lower() and hasattr(mod, "in_proj"):
                mixer = mod
                report["mixer_path"] = name
                report["mixer_type"] = type(mod).__name__
                break
        if mixer is not None:
            report["mixer_params"] = {
                n: list(p.shape) for n, p in mixer.named_parameters()
            }
            report["mixer_attrs"] = {
                a: getattr(mixer, a)
                for a in (
                    "intermediate_size",
                    "ssm_state_size",
                    "conv_kernel_size",
                    "num_heads",
                    "head_dim",
                    "n_groups",
                    "chunk_size",
                )
                if isinstance(getattr(mixer, a, None), int)
            }

    # Does the recurrent state actually respond to the image?
    resp = requests.get(
        SMOKE_IMAGE_URL, timeout=60, headers={"User-Agent": "visual-memory-lab/0.1"}
    )
    resp.raise_for_status()
    pil = PILImage.open(io.BytesIO(resp.content)).convert("RGB")

    def run(with_image: bool):
        if with_image:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the scene."},
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[pil], return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Describe the scene."},
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt], return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        return inputs, out.past_key_values

    inp_img, cache_img = run(True)
    inp_txt, cache_txt = run(False)
    report["tokens_with_image"] = int(inp_img["input_ids"].shape[1])
    report["tokens_without_image"] = int(inp_txt["input_ids"].shape[1])
    report["image_token_cost"] = (
        report["tokens_with_image"] - report["tokens_without_image"]
    )

    keys = sorted(cache_img.ssm_states.keys(), key=lambda k: int(k))
    mamba_ids = [int(k) for k in keys if int(k) not in set(model.config.text_config.hybrid_layer_ids)]
    report["n_mamba_state_layers"] = len(mamba_ids)
    sample = []
    for lid in mamba_ids[:5]:
        a = cache_img.ssm_states[lid].float()
        b = cache_txt.ssm_states[lid].float()
        sample.append({
            "layer": lid,
            "norm_with_image": round(a.norm().item(), 3),
            "norm_without_image": round(b.norm().item(), 3),
            "relative_difference": round(((a - b).norm() / (b.norm() + 1e-6)).item(), 4),
        })
    report["state_responds_to_image"] = sample
    report["transformer_layers"] = list(getattr(cache_img, "transformer_layers", []))
    kc = [i for i, t in enumerate(cache_img.key_cache) if torch.is_tensor(t) and t.numel() > 0]
    report["layers_with_kv"] = kc
    if kc:
        report["kv_shape"] = list(cache_img.key_cache[kc[0]].shape)

    with open("/results/e1_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    results_vol.commit()
    logger.info("E1 REPORT\n%s", json.dumps(report, indent=2, default=str))
    return json.dumps(report, indent=2, default=str)


GUTENBERG_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def prepare_corpus() -> str:
    """Fetch COCO instance annotations and the filler corpus, once."""
    import os
    import zipfile

    import requests

    os.makedirs("/results/corpus", exist_ok=True)
    ann_path = "/results/corpus/instances_val2017.json"
    if not os.path.exists(ann_path):
        logger.info("downloading COCO annotations")
        zip_path = "/tmp/ann.zip"
        with requests.get(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            stream=True,
            timeout=1200,
        ) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("annotations/instances_val2017.json") as src:
                with open(ann_path, "wb") as dst:
                    dst.write(src.read())
        os.remove(zip_path)

    filler_path = "/results/corpus/filler.txt"
    if not os.path.exists(filler_path):
        logger.info("downloading filler corpus")
        resp = requests.get(
            GUTENBERG_URL, timeout=300, headers={"User-Agent": "visual-memory-lab/0.1"}
        )
        resp.raise_for_status()
        text = resp.text
        start = text.find("It is a truth universally acknowledged")
        body = text[start:] if start > 0 else text
        with open(filler_path, "w") as fh:
            fh.write(body)

    results_vol.commit()
    sizes = {
        "annotations_mb": round(os.path.getsize(ann_path) / 1e6, 1),
        "filler_kb": round(os.path.getsize(filler_path) / 1e3, 1),
    }
    logger.info("corpus ready %s", sizes)
    return json.dumps(sizes)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=3600)
def e2() -> str:
    """Third gate: end-to-end pipeline on a handful of trials, with timing.

    Confirms the codebook builds, the readout produces sane shapes, the
    zero-shot decoder does something at distance zero, the pre-image control
    runs, and gives the seconds-per-trial number the full run is budgeted on.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, vtelemetry as vt

    report: dict = {}
    t0 = time.time()
    model, processor = vt.load_model()
    geom = vt.geometry(model)
    matrix = vt.projection_matrix(N_PROJECT_FEATURES, geom)
    report["load_seconds"] = round(time.time() - t0, 1)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    report["n_coco_images"] = len(coco["images"])

    trials = materials.build_trials(coco, ["identity", "count", "position"], n_blocks=1, seed=17)
    report["n_trials_one_block_each"] = len(trials)
    by_family: dict = {}
    for t in trials:
        by_family.setdefault(t.family, []).append(t)
    report["trials_per_family"] = {k: len(v) for k, v in by_family.items()}

    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()
    report["filler_words"] = len(filler_words)

    t0 = time.time()
    codebooks = {}
    for family, group in by_family.items():
        codebooks[family] = decay.build_codebook(model, processor, group[0].candidates, geom)
    report["codebook_seconds"] = round(time.time() - t0, 1)
    report["codebook_shape"] = list(codebooks["count"].shape)

    timings: dict = {}
    zero_shot: dict = {}
    for family, group in by_family.items():
        for distance in (0, 2048):
            t0 = time.time()
            hits = 0
            for trial in group:
                record, scores, _feats = decay.run_trial(
                    model, processor, trial, distance, filler_words,
                    codebooks, matrix, "/results/images", seed=17, geom=geom,
                )
                pooled = scores.float().mean(dim=(0, 1))
                if int(torch.argmax(pooled)) == trial.answer_index:
                    hits += 1
            timings[f"{family}-d{distance}"] = round((time.time() - t0) / len(group), 2)
            zero_shot[f"{family}-d{distance}"] = f"{hits}/{len(group)}"
    report["seconds_per_trial"] = timings
    report["zero_shot_argmax"] = zero_shot
    report["chance"] = "1/8"

    ctrl_hits = 0
    for family, group in by_family.items():
        for trial in group:
            _, scores, _feats = decay.run_pre_image_control(
                model, processor, trial, filler_words, codebooks, matrix,
                seed=17, geom=geom,
            )
            pooled = scores.float().mean(dim=(0, 1))
            if int(torch.argmax(pooled)) == trial.answer_index:
                ctrl_hits += 1
    report["pre_image_control"] = f"{ctrl_hits}/{len(trials)}"
    report["cuda_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    with open("/results/e2_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    results_vol.commit()
    logger.info("E2 REPORT\n%s", json.dumps(report, indent=2, default=str))
    return json.dumps(report, indent=2, default=str)


# Width of the per-layer random projection of the recurrent state. The signal
# gate ran at 128 and found count at chance; the capacity control (e4) showed
# that was partly narrowness, since count reaches 19.0% (p = 0.006) at 512
# while a coarse control property reaches 71.0% on the same features. Every
# reported comparison therefore uses the same width.
N_PROJECT_FEATURES = 512


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=10800)
def e3(n_blocks: int = 15) -> str:
    """Signal gate, run at distance zero only.

    Three questions, and the project only continues if all three answer yes:
      1. Can the model itself answer these questions right after the image?
         If not, the questions are wrong, not the model's memory.
      2. Does a linear probe recover the answer from the recurrent state?
      3. Do the pre-image and label-permutation controls sit at chance?
    """
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, probe, vtelemetry as vt

    report: dict = {"n_blocks": n_blocks, "n_project_features": N_PROJECT_FEATURES}
    model, processor = vt.load_model()
    geom = vt.geometry(model)
    matrix = vt.projection_matrix(N_PROJECT_FEATURES, geom)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(
        coco, ["identity", "count", "position"], n_blocks=n_blocks, seed=17
    )
    by_family: dict = {}
    for t in trials:
        by_family.setdefault(t.family, []).append(t)
    report["trials_per_family"] = {k: len(v) for k, v in by_family.items()}

    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    codebooks = {
        family: decay.build_codebook(model, processor, group[0].candidates, geom)
        for family, group in by_family.items()
    }

    results: dict = {}
    for family, group in by_family.items():
        t0 = time.time()
        feats, labels, groups, behaviour_hits, behaviour_seen = [], [], [], 0, 0
        for trial in group:
            record, _scores, features = decay.run_trial(
                model, processor, trial, 0, filler_words, codebooks, matrix,
                "/results/images", seed=17, geom=geom,
            )
            feats.append(features.float().reshape(-1))
            labels.append(trial.answer_index)
            groups.append(trial.block_id)
            behaviour_seen += 1
            behaviour_hits += int(record["behaviour"]["correct"])

        control_feats, control_labels, control_groups = [], [], []
        for trial in group:
            _record, _scores, features = decay.run_pre_image_control(
                model, processor, trial, filler_words, codebooks, matrix, seed=17, geom=geom
            )
            control_feats.append(features.float().reshape(-1))
            control_labels.append(trial.answer_index)
            control_groups.append(trial.block_id)

        x = torch.stack(feats)
        y = torch.tensor(labels)
        main = probe.cross_validated_accuracy(x, y, groups, n_classes=8)
        perm = probe.permuted_accuracy(x, y, groups, n_classes=8, n_repeats=3)
        control = probe.cross_validated_accuracy(
            torch.stack(control_feats), torch.tensor(control_labels),
            control_groups, n_classes=8,
        )

        results[family] = {
            "behaviour_accuracy": round(behaviour_hits / max(behaviour_seen, 1), 4),
            "behaviour": f"{behaviour_hits}/{behaviour_seen}",
            "probe_accuracy": round(main["accuracy"], 4),
            "probe_p_value": main["p_value"],
            "permutation_mean": round(perm["mean"], 4),
            "pre_image_accuracy": round(control["accuracy"], 4),
            "chance": 0.125,
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%s -> %s", family, json.dumps(results[family]))

    report["families"] = results
    report["cuda_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    with open("/results/e3_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    results_vol.commit()
    logger.info("E3 REPORT\n%s", json.dumps(report, indent=2, default=str))
    return json.dumps(report, indent=2, default=str)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=14400)
def e4(n_blocks: int = 25, n_features: int = 512) -> str:
    """Capacity control for the count and position nulls.

    A failure to decode is only evidence of absence if the features could have
    carried the answer. Two checks, both on the identical feature vectors:

      1. Widen the projection from 128 to `n_features` per layer. If count is
         still at chance with four times the capacity, narrowness is not the
         explanation.
      2. Decode a coarse property (scene complexity, median-split on the
         number of distinct COCO categories present) that varies within block.
         If that decodes while count does not, the features demonstrably carry
         visual content and the count null is about the state, not the probe.
    """
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, probe, vtelemetry as vt

    report: dict = {"n_blocks": n_blocks, "n_project_features": n_features}
    model, processor = vt.load_model()
    geom = vt.geometry(model)
    matrix = vt.projection_matrix(n_features, geom)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["count", "position"], n_blocks=n_blocks, seed=17)
    by_family: dict = {}
    for t in trials:
        by_family.setdefault(t.family, []).append(t)

    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()
    codebooks = {
        family: decay.build_codebook(model, processor, group[0].candidates, geom)
        for family, group in by_family.items()
    }

    def complexity(image_id: int) -> int:
        return len(coco["by_image"].get(image_id, {}))

    results: dict = {}
    for family, group in by_family.items():
        t0 = time.time()
        feats, labels, groups, complexities = [], [], [], []
        for trial in group:
            _record, _scores, features = decay.run_trial(
                model, processor, trial, 0, filler_words, codebooks, matrix,
                "/results/images", seed=17, geom=geom, do_behaviour=False,
            )
            feats.append(features.float().reshape(-1))
            labels.append(trial.answer_index)
            groups.append(trial.block_id)
            complexities.append(complexity(trial.image_id))

        x = torch.stack(feats)
        main = probe.cross_validated_accuracy(
            x, torch.tensor(labels), groups, n_classes=8
        )

        median = float(torch.tensor(complexities, dtype=torch.float32).median())
        binary = torch.tensor([int(c > median) for c in complexities])
        balance = float(binary.float().mean())
        positive = probe.cross_validated_accuracy(
            x, binary, groups, n_classes=2
        )

        results[family] = {
            "target_accuracy": round(main["accuracy"], 4),
            "target_p_value": main["p_value"],
            "target_chance": 0.125,
            "complexity_accuracy": round(positive["accuracy"], 4),
            "complexity_p_value": positive["p_value"],
            "complexity_chance": 0.5,
            "complexity_positive_rate": round(balance, 3),
            "n": len(group),
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%s -> %s", family, json.dumps(results[family]))

    report["families"] = results
    with open("/results/e4_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    results_vol.commit()
    logger.info("E4 REPORT\n%s", json.dumps(report, indent=2, default=str))
    return json.dumps(report, indent=2, default=str)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=43200)
def collect_splice_v3(
    n_pairs: int = 30, distance: int = 0, image_size: int = 448, tag: str = "clean"
) -> str:
    """The splice with its two artifact explanations closed off.

    Fix 1: every image is resized to one square geometry, so both runs emit
    the same number of vision tokens and the two attention caches are the same
    length. Pairs whose prompts still differ in length are dropped rather than
    spliced, and the count of dropped pairs is reported.

    Fix 2: the recurrent swap is verified to have taken effect, by measuring
    the state before and after. "Replacing the recurrent memory changes
    nothing" is only meaningful if the replacement demonstrably happened.

    A fourth condition, both channels swapped, is the positive control: it
    should follow the donor, confirming the machinery can flip the answer.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, vtelemetry as vt

    model, processor = vt.load_model()
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)
    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    def prefill_context(trial, filler):
        img = decay.fetch_image(trial.file_name, "/results/images", size=image_size)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": filler}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        return out.past_key_values, int(inputs["input_ids"].shape[1])

    def ask_through(cache, trial, max_new_tokens: int = 10) -> str:
        ids = processor.tokenizer(
            trial.question(), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        language_model = model.language_model
        with torch.no_grad():
            out = None
            for position in range(ids.shape[1]):
                out = language_model(
                    input_ids=ids[:, position:position + 1],
                    past_key_values=cache, use_cache=True,
                )
            generated = []
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(max_new_tokens):
                generated.append(int(token))
                out = language_model(
                    input_ids=token, past_key_values=cache, use_cache=True
                )
                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def state_fingerprint(cache) -> float:
        return float(sum(cache.ssm_states[i].float().norm() for i in range(0, 81, 8)))

    rng = random.Random(23)
    records: list[dict] = []
    dropped = 0
    swap_checks: list[dict] = []
    started = time.time()

    for family in ("identity", "count", "position"):
        trials = materials.build_trials(coco, [family], n_blocks=20, seed=23)
        by_answer: dict = {}
        for t in trials:
            by_answer.setdefault(t.answer, []).append(t)
        answers = sorted(by_answer)

        made = 0
        attempts = 0
        while made < n_pairs and attempts < n_pairs * 6:
            attempts += 1
            a, b = rng.sample(answers, 2)
            host, donor = rng.choice(by_answer[a]), rng.choice(by_answer[b])
            filler = decay.filler_text(filler_words, distance, random.Random(attempts))

            donor_cache, donor_len = prefill_context(donor, filler)
            probe_cache, host_len = prefill_context(host, filler)
            if donor_len != host_len:
                dropped += 1
                continue

            row = {
                "family": family, "pair": made,
                "host_answer": host.answer, "donor_answer": donor.answer,
                "distance": distance, "prompt_tokens": host_len,
            }

            for condition in ("none", "recurrent", "attention", "both"):
                cache, _ = prefill_context(host, filler)
                if condition == "none":
                    text = ask_through(cache, host)
                elif condition == "both":
                    with vt.channel_swapped(cache, donor_cache, "recurrent", attn_ids, geom):
                        with vt.channel_swapped(cache, donor_cache, "attention", attn_ids, geom):
                            text = ask_through(cache, host)
                else:
                    before = state_fingerprint(cache)
                    with vt.channel_swapped(cache, donor_cache, condition, attn_ids, geom):
                        if condition == "recurrent":
                            after = state_fingerprint(cache)
                            swap_checks.append({
                                "family": family,
                                "changed": abs(after - before) > 1e-3,
                                "before": round(before, 2),
                                "after": round(after, 2),
                            })
                        text = ask_through(cache, host)
                scored = decay.score_answer(text, host)
                row[condition] = {
                    "raw": text,
                    "chosen": scored["chosen"],
                    "follows_host": scored["chosen"] == host.answer,
                    "follows_donor": scored["chosen"] == donor.answer,
                }
            records.append(row)
            made += 1
            if made % 10 == 0:
                logger.info(
                    "%s %d/%d, %.1f min", family, made, n_pairs,
                    (time.time() - started) / 60,
                )

    with open(f"/results/splice_{tag}.jsonl", "w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")

    summary: dict = {
        "n_pairs_per_family": n_pairs,
        "distance": distance,
        "image_size": image_size,
        "dropped_length_mismatch": dropped,
        "recurrent_swap_took_effect": (
            f"{sum(c['changed'] for c in swap_checks)}/{len(swap_checks)}"
        ),
        "swap_check_examples": swap_checks[:3],
    }
    for family in ("identity", "count", "position"):
        rows = [r for r in records if r["family"] == family]
        summary[family] = {
            condition: {
                "follows_host": sum(r[condition]["follows_host"] for r in rows),
                "follows_donor": sum(r[condition]["follows_donor"] for r in rows),
                "n": len(rows),
            }
            for condition in ("none", "recurrent", "attention", "both")
        }
    summary["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/splice_{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    results_vol.commit()
    logger.info("SPLICE v3 SUMMARY\n%s", json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)


@app.function(image=image, volumes=VOLUMES, timeout=1200)
def inspect_source() -> str:
    """Print the parts of the Zamba2 forward chain the splice has to satisfy."""
    import inspect as inspect_mod

    from transformers.models.zamba2 import modeling_zamba2 as mz

    out = []
    src = inspect_mod.getsource(mz)
    lines = src.splitlines()

    def window(centre: int, before: int = 12, after: int = 22) -> str:
        lo, hi = max(0, centre - before), min(len(lines), centre + after)
        return "\n".join(f"{i+1:5d}| {lines[i]}" for i in range(lo, hi))

    out.append("=== Zamba2MambaMixer.torch_forward head ===")
    out.append(window(751))
    out.append("\n=== around line 1109 (layer forward) ===")
    out.append(window(1108))
    out.append("\n=== Zamba2HybridLayer.forward signature ===")
    for name in ("Zamba2HybridLayer", "Zamba2MambaDecoderLayer", "Zamba2Model", "Zamba2ForCausalLM"):
        cls = getattr(mz, name, None)
        if cls is not None:
            out.append(f"{name}.forward{inspect_mod.signature(cls.forward)}")
    text = "\n".join(out)
    print(text, flush=True)
    return text


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def inspect_dynamics() -> str:
    """Dump exactly how Zamba2 forms A and dt, so E5 computes the real thing.

    The half-life derivation needs three facts we must not guess: the name and
    sign convention of the A parameter, the transform applied to the dt slice
    of in_proj (softplus, bias, clamping), and where the time-step limits live
    on the config. Reading them from the installed source costs one CPU
    container and removes a whole class of quietly wrong numbers.
    """
    import inspect as inspect_mod
    import re

    from transformers.models.zamba2 import modeling_zamba2 as mz

    out = []
    mixer = getattr(mz, "Zamba2MambaMixer", None)
    if mixer is None:
        raise RuntimeError("Zamba2MambaMixer not found in installed source")

    out.append("=== Zamba2MambaMixer.__init__ ===")
    out.append(inspect_mod.getsource(mixer.__init__))

    src = inspect_mod.getsource(mixer)
    lines = src.splitlines()
    for pattern in (r"A_log", r"softplus", r"time_step", r"dt_bias", r"dt_limit", r"clamp"):
        hits = [i for i, line in enumerate(lines) if re.search(pattern, line)]
        out.append(f"\n=== lines matching {pattern!r} ({len(hits)} hits) ===")
        for i in hits[:14]:
            lo, hi = max(0, i - 2), min(len(lines), i + 3)
            out.append("\n".join(f"{j:5d}| {lines[j]}" for j in range(lo, hi)))
            out.append("   ---")

    out.append("\n=== torch_forward state update, verbatim ===")
    for lo, hi in ((270, 300), (325, 360)):
        out.append("\n".join(f"{j:5d}| {lines[j]}" for j in range(lo, min(hi, len(lines)))))
        out.append("   ---")

    cfg_cls = getattr(mz, "Zamba2Config", None)
    if cfg_cls is None:
        from transformers.models.zamba2 import configuration_zamba2 as cz

        cfg_cls = cz.Zamba2Config
    out.append("\n=== Zamba2Config.__init__ signature ===")
    out.append(str(inspect_mod.signature(cfg_cls.__init__)))

    text = "\n".join(out)
    print(text, flush=True)
    return text


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def e5b(
    model_id: str = MODEL_ID,
    dynamics_path: str = "/results/e5_dynamics.npz",
    n_blocks: int = 15,
    n_features: int = 512,
    stratify: str = "halflife",
    distances_json: str = "[0,2,4,8,16,32,64]",
    tag: str = "7b",
) -> str:
    """E5b: does the predicted half-life of a head govern what it still holds?

    E5 showed that the measured forgetting horizon matches the one implied by
    A and dt. That is a curve comparison, and a curve comparison cannot rule
    out a coincidence. This is the within-model test: split the heads into
    thirds by their predicted half-life and probe each third separately. Decay
    predicts that the slow third keeps decodable identity to larger distances
    than the fast third, with the crossings near their respective predicted
    half-lives. If the three strata behave identically, decay is not what is
    driving the collapse and the interference branch takes over.

    One forward pass serves all three strata, because they are three different
    projections of the same recurrent state.

    `stratify="gain_matched"` replaces the tercile split with the two
    amplitude-controlled strata from `dynamics.gain_matched_masks`: heads whose
    mean gate sits in a narrow band (so write gain is matched by construction)
    split by |A| alone. This is the split under which the original inversion
    claim must be re-measured, because the tercile split confounds decay rate
    with write amplitude.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from lab import decay as decay_mod
    from lab import dynamics as dyn
    from lab import materials
    from lab import vtelemetry as vt

    started = time.time()
    distances = tuple(json.loads(distances_json))
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)

    stored = np.load(dynamics_path)
    half_lives = torch.from_numpy(stored["half_life_first32"])
    if tuple(half_lives.shape) != (geom.n_layers, geom.n_heads):
        raise RuntimeError(
            f"dynamics file has half-lives of shape {tuple(half_lives.shape)}, "
            f"model geometry is [{geom.n_layers}, {geom.n_heads}]"
        )

    if stratify == "gain_matched":
        fast_np, slow_np, gain_report = dyn.gain_matched_masks(
            stored["abs_a"], stored["mean_dt_first32"]
        )
        masks = {
            "fast": torch.from_numpy(fast_np),
            "slow": torch.from_numpy(slow_np),
        }
        strata = ("fast", "slow")
    else:
        masks = dyn.head_tercile_masks(half_lives)
        strata = ("fast", "middle", "slow")
        gain_report = None
    report: dict = {
        "model_id": model_id,
        "distances": list(distances),
        "n_features": n_features,
        "stratify": stratify,
        "gain_matched": gain_report,
        "strata": {s: dyn.stratum_summary(half_lives, masks[s]) for s in strata},
    }
    logger.info("strata\n%s", json.dumps(report["strata"], indent=2))

    # One shared projection, the same basis the unstratified probe uses, so the
    # strata are comparable to it and to each other.
    matrix = vt.projection_matrix(n_features, geom, device=model.device)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=n_blocks, seed=17)
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    records: list[dict] = []
    features = {s: [] for s in strata}
    for distance in distances:
        for trial in trials:
            rng = random.Random(f"{trial.trial_id}-{distance}-17")
            filler = materials.filler_text(filler_words, distance, rng)
            picture = decay_mod.fetch_image(trial.file_name, "/results/images")
            prompt = decay_mod._readout_prompt(processor, filler)
            inputs = processor(
                text=[prompt], images=[picture], return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                out = model(**inputs, use_cache=True)
            cache = out.past_key_values
            for stratum in strata:
                projected = vt.project_state_masked(
                    cache, matrix, masks[stratum], geom
                )
                features[stratum].append(
                    projected.float().reshape(-1).numpy().astype(np.float16)
                )
            records.append(
                {
                    "trial_id": trial.trial_id,
                    "family": "identity",
                    "block_id": trial.block_id,
                    "image_id": trial.image_id,
                    "distance": distance,
                    "answer_index": trial.answer_index,
                    "candidates": list(trial.candidates),
                }
            )
        logger.info(
            "d=%d done, %.1f min elapsed", distance, (time.time() - started) / 60
        )

    for stratum in strata:
        base = f"/results/e5b_{tag}_{stratum}"
        with open(f"{base}.jsonl", "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        np.savez_compressed(f"{base}_features.npz", features=np.stack(features[stratum]))

    report["n_records"] = len(records)
    report["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/e5b_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E5B REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def e9(
    n_blocks: int = 15,
    retentions_json: str = "[1.0,0.5,0.25,0.1,0.05,0.0]",
    distances_json: str = "[0,32,256]",
    strategy: str = "random",
    model_id: str = MODEL_ID,
    tag: str = "7b",
) -> str:
    """E9: what happens when the visual KV entries are evicted.

    Methods that prune visual tokens or evict visual KV rest on the premise
    that those entries are largely redundant. Our splice says that in this
    hybrid the attention layers hold the only copy of the visual particulars,
    because the recurrent channel demonstrably does not carry them. If that is
    right, eviction should be far more damaging here than on a dense
    transformer of comparable size.

    This also supplies the positive control the splice has been missing. At
    zero retention the model has no visual information in the attention
    channel at all, so whatever accuracy survives is what the recurrent state
    alone can support. A null recurrent splice is only interpretable against
    that number.

    Two strategies, neither of them a specific published scorer: `random`
    keeps a uniform sample of visual positions, `recent` keeps the final
    fraction. The claim in the paper is about the redundancy premise, not
    about beating any particular method.

    The distance axis is the point of the experiment. A pilot at distance zero
    found the model almost unharmed by evicting 95% of the visual entries,
    which on its own would say the redundancy premise transfers to hybrids
    intact. But at distance zero the recurrent state still holds the picture,
    so the attention entries are not the only copy yet. E5 says that backup
    is gone within single-digit tokens. The prediction this experiment tests
    is therefore an interaction: eviction should be harmless close to the
    image and damaging once the recurrent state has emptied.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay as decay_mod
    from lab import materials
    from lab import vtelemetry as vt

    if strategy not in ("random", "recent"):
        raise ValueError(f"unknown strategy {strategy!r}")

    started = time.time()
    retentions = [float(r) for r in json.loads(retentions_json)]
    distances = [int(d) for d in json.loads(distances_json)]
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=n_blocks, seed=17)
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    def build_inputs(picture, trial, filler: str):
        """The full prompt the model normally sees, image and question together.

        The eviction is applied part way through this exact sequence rather
        than to an improvised image-only prompt. That matters: feeding a bare
        question through an image-only cache drops unevicted accuracy to a
        quarter, which would leave the sweep no dynamic range to measure.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"{filler}\n\n{trial.question()}"
                            if filler
                            else trial.question()
                        ),
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(
            text=[prompt], images=[picture], return_tensors="pt"
        ).to(model.device)

    def prefill_through(inputs, split: int):
        """Run the prompt up to `split`, which is the token after the image."""
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"][:, :split],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                use_cache=True,
            )
        return out.past_key_values

    def finish(cache, inputs, split: int, max_new_tokens: int = 10) -> str:
        """Feed the remaining prompt tokens, then decode greedily.

        Zamba2's naive path accepts one token at a time once the cache holds a
        previous state: torch_forward does input_states.squeeze(1) under
        has_previous_state, so a multi-token feed makes the gate 4D.
        """
        ids = inputs["input_ids"]
        language_model = model.language_model
        with torch.no_grad():
            out = None
            for position in range(split, ids.shape[1]):
                out = language_model(
                    input_ids=ids[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
            produced = []
            for _ in range(max_new_tokens):
                token = int(torch.argmax(out.logits[0, -1]))
                produced.append(token)
                if token == processor.tokenizer.eos_token_id:
                    break
                out = language_model(
                    input_ids=torch.tensor([[token]], device=model.device),
                    past_key_values=cache,
                    use_cache=True,
                )
        return processor.tokenizer.decode(produced, skip_special_tokens=True)

    records: list[dict] = []
    failures = 0
    for index, trial in enumerate(trials):
        picture = decay_mod.fetch_image(trial.file_name, "/results/images")
        for distance in distances:
            filler = materials.filler_text(
                filler_words, distance, random.Random(f"{trial.trial_id}-{distance}")
            )
            inputs = build_inputs(picture, trial, filler)
            img_start, img_end = _image_token_span(inputs["input_ids"], model, processor)
            n_visual = img_end - img_start + 1
            if n_visual < 64:
                raise RuntimeError(f"implausible visual span of {n_visual} tokens")
            split = img_end + 1
            cache = prefill_through(inputs, split)

            saved_recurrent = vt.snapshot_recurrent(cache, geom)
            saved_k = {i: cache.key_cache[i].clone() for i in attn_ids}
            saved_v = {i: cache.value_cache[i].clone() for i in attn_ids}
            total_positions = saved_k[attn_ids[0]].shape[2]

            for retention in retentions:
                vt.restore_recurrent(cache, saved_recurrent)
                n_keep = int(round(retention * n_visual))
                rng = random.Random(f"{trial.trial_id}-{retention}-e9")
                if strategy == "random":
                    kept_visual = sorted(rng.sample(range(n_visual), n_keep))
                else:
                    kept_visual = list(range(n_visual - n_keep, n_visual))
                keep = (
                    list(range(0, img_start))
                    + [img_start + offset for offset in kept_visual]
                    + list(range(img_end + 1, total_positions))
                )
                index_tensor = torch.tensor(keep, device=model.device)
                for i in attn_ids:
                    cache.key_cache[i] = saved_k[i].index_select(2, index_tensor)
                    cache.value_cache[i] = saved_v[i].index_select(2, index_tensor)

                try:
                    answer = finish(cache, inputs, split)
                    scored = decay_mod.score_answer(answer, trial)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.warning("trial %s retention %s failed: %s", trial.trial_id, retention, exc)
                    continue

                records.append(
                    {
                        "trial_id": trial.trial_id,
                        "image_id": trial.image_id,
                        "block_id": trial.block_id,
                        "retention": retention,
                        "distance": distance,
                        "strategy": strategy,
                        "n_visual_tokens": int(n_visual),
                        "n_visual_kept": int(n_keep),
                        "answer": answer,
                        "correct": bool(scored["correct"]),
                    }
                )

            # Put the cache back so nothing leaks into the next trial.
            for i in attn_ids:
                cache.key_cache[i] = saved_k[i]
                cache.value_cache[i] = saved_v[i]
        if index % 20 == 0:
            logger.info("trial %d/%d, %.1f min", index, len(trials), (time.time() - started) / 60)

    by_retention: dict = {}
    for record in records:
        key = f"d{record['distance']}_r{record['retention']}"
        cell = by_retention.setdefault(key, {"n": 0, "hits": 0})
        cell["n"] += 1
        cell["hits"] += int(record["correct"])
    for cell in by_retention.values():
        cell["accuracy"] = round(cell["hits"] / cell["n"], 4) if cell["n"] else None

    report = {
        "model_id": model_id,
        "strategy": strategy,
        "distances": distances,
        "retentions": retentions,
        "n_trials": len(trials),
        "n_records": len(records),
        "failures": failures,
        "by_retention": by_retention,
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(f"/results/e9_{tag}_{strategy}.jsonl", "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    with open(f"/results/e9_report_{tag}_{strategy}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E9 REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def e10(
    n_pairs: int = 120,
    image_size: int = 448,
    family: str = "identity",
    model_id: str = MODEL_ID,
    tag: str = "scaled",
) -> str:
    """E10: the splice at scale, with the ceiling removed and a dose response.

    Two problems with the original splice. Its host baseline was 17 of 30, so
    the control topped out at 57% and invited the question of what the other
    43% were doing. And the manipulation was all or nothing, which shows that
    the attention channel matters without showing how much of it matters.

    Both are fixed here. Pairs are admitted only if the host answers its own
    question correctly and the donor answers its own question correctly, so the
    ceiling is 100% by construction rather than by luck. And the attention swap
    is run over subsets of the attention layers, which turns a binary result
    into a curve and answers whether the picture is held redundantly across all
    of them or concentrated in a few.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, splice
    from lab import vtelemetry as vt

    started = time.time()
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)
    subsets = splice.attention_subsets(attn_ids)
    coco = materials.load_coco("/results/corpus/instances_val2017.json")

    trials = materials.build_trials(coco, [family], n_blocks=40, seed=23)
    by_answer: dict = {}
    for trial in trials:
        by_answer.setdefault(trial.answer, []).append(trial)
    answers = sorted(by_answer)

    def context(trial):
        """Prefill to the token after the image, and return what is needed to
        finish the real prompt afterwards."""
        picture = decay.fetch_image(trial.file_name, "/results/images", size=image_size)
        inputs = splice.full_prompt_inputs(model, processor, picture, trial.question())
        _, img_end = _image_token_span(inputs["input_ids"], model, processor)
        split = img_end + 1
        return splice.prefill_to(model, inputs, split), inputs, split

    def answers_own_question(trial) -> bool:
        cache, inputs, split = context(trial)
        text = splice.finish_prompt(model, processor, cache, inputs, split)
        return bool(decay.score_answer(text, trial)["correct"])

    rng = random.Random(23)
    records: list[dict] = []
    dropped_length, dropped_baseline = 0, 0
    swap_checks: list[dict] = []
    baseline_cache: dict = {}

    made, attempts = 0, 0
    while made < n_pairs and attempts < n_pairs * 12:
        attempts += 1
        a, b = rng.sample(answers, 2)
        host, donor = rng.choice(by_answer[a]), rng.choice(by_answer[b])

        # Admit the pair only if both images are answered correctly on their
        # own. This is what removes the ceiling.
        ok = True
        for trial in (host, donor):
            if trial.trial_id not in baseline_cache:
                baseline_cache[trial.trial_id] = answers_own_question(trial)
            ok = ok and baseline_cache[trial.trial_id]
        if not ok:
            dropped_baseline += 1
            continue

        donor_cache, _donor_inputs, donor_split = context(donor)
        _host_cache, host_inputs, host_split = context(host)
        if donor_split != host_split:
            dropped_length += 1
            continue

        row = {
            "family": family,
            "pair": made,
            "host_answer": host.answer,
            "donor_answer": donor.answer,
            "prompt_tokens": host_split,
        }

        conditions = ["none", "recurrent"] + list(subsets) + ["both"]
        for condition in conditions:
            cache = splice.prefill_to(model, host_inputs, host_split)
            finish = lambda c: splice.finish_prompt(  # noqa: E731
                model, processor, c, host_inputs, host_split
            )
            if condition == "none":
                text = finish(cache)
            elif condition == "recurrent":
                before = splice.state_fingerprint(cache)
                with vt.channel_swapped(cache, donor_cache, "recurrent", attn_ids, geom):
                    after = splice.state_fingerprint(cache)
                    swap_checks.append(
                        {"changed": abs(after - before) > 1e-3,
                         "before": round(before, 2), "after": round(after, 2)}
                    )
                    text = finish(cache)
            elif condition == "both":
                with vt.channel_swapped(cache, donor_cache, "recurrent", attn_ids, geom):
                    with vt.channel_swapped(
                        cache, donor_cache, "attention", attn_ids, geom
                    ):
                        text = finish(cache)
            else:
                with vt.channel_swapped(
                    cache, donor_cache, "attention", subsets[condition], geom
                ):
                    text = finish(cache)

            scored = decay.score_answer(text, host)
            row[condition] = {
                "raw": text,
                "chosen": scored["chosen"],
                "follows_host": scored["chosen"] == host.answer,
                "follows_donor": scored["chosen"] == donor.answer,
            }

        records.append(row)
        made += 1
        if made % 10 == 0:
            logger.info(
                "%d/%d pairs, %.1f min", made, n_pairs, (time.time() - started) / 60
            )

    summary: dict = {}
    for condition in ["none", "recurrent"] + list(subsets) + ["both"]:
        rows = [r for r in records if condition in r]
        n = len(rows)
        host_hits = sum(int(r[condition]["follows_host"]) for r in rows)
        donor_hits = sum(int(r[condition]["follows_donor"]) for r in rows)
        summary[condition] = {
            "n": n,
            "follows_host": host_hits,
            "follows_donor": donor_hits,
            "host_rate": round(host_hits / n, 4) if n else None,
            "donor_rate": round(donor_hits / n, 4) if n else None,
            "n_layers_swapped": (
                len(subsets[condition]) if condition in subsets
                else (len(attn_ids) if condition == "both" else 0)
            ),
        }

    report = {
        "model_id": model_id,
        "family": family,
        "n_pairs": len(records),
        "dropped_for_length": dropped_length,
        "dropped_for_baseline": dropped_baseline,
        "attempts": attempts,
        "swap_verified": f"{sum(int(c['changed']) for c in swap_checks)}/{len(swap_checks)}",
        "summary": summary,
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(f"/results/e10_{tag}.jsonl", "w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")
    with open(f"/results/e10_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E10 REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def e5c(
    model_id: str = MODEL_ID,
    dynamics_path: str = "/results/e5_dynamics.npz",
    n_blocks: int = 8,
    n_features: int = 512,
    distances_json: str = "[0,32,64]",
    tag: str = "7b",
) -> str:
    """E5c: is the recurrent state remembering the image, or re-reading it?

    E5b produced a result that inverts the decay prediction. Heads whose
    predicted half-life is about two tokens decode object identity at 62% after
    sixty-four tokens, while heads with a predicted half-life of twenty-nine
    tokens fall to 17% over the same span. The layer composition of the strata
    runs the wrong way to explain it: the fast heads sit in earlier layers that
    decode worse on their own.

    There is only one consistent reading. A head with a two-token half-life
    cannot still be holding something written sixty-four tokens ago, because its
    state is by then composed almost entirely of recent writes. So the identity
    information in that state arrived recently, which means something else in
    the model is still supplying it. The only candidate is the attention
    channel, whose key-value entries still contain the image.

    This function tests that directly. It probes the recurrent state with the
    visual key-value entries intact, and again with them evicted. If the
    re-encoding account is right, evicting the visual entries should collapse
    decodability at distance, and should do so most dramatically for the fast
    heads. If decodability at distance survives eviction, the state really is
    remembering and the account is wrong.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from lab import decay as decay_mod
    from lab import dynamics as dyn
    from lab import materials
    from lab import vtelemetry as vt

    started = time.time()
    distances = [int(d) for d in json.loads(distances_json)]
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)

    stored = np.load(dynamics_path)
    half_lives = torch.from_numpy(stored["half_life_first32"])
    masks = dyn.head_tercile_masks(half_lives)
    strata = ("fast", "middle", "slow", "all")
    masks["all"] = torch.ones_like(masks["fast"])
    matrix = vt.projection_matrix(n_features, geom, device=model.device)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=n_blocks, seed=17)
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    records: list[dict] = []
    features = {s: [] for s in strata}

    for index, trial in enumerate(trials):
        picture = decay_mod.fetch_image(trial.file_name, "/results/images")
        for distance in distances:
            rng = random.Random(f"{trial.trial_id}-{distance}-e5c")
            filler = materials.filler_text(filler_words, distance, rng)
            prompt = decay_mod._readout_prompt(processor, filler)
            inputs = processor(
                text=[prompt], images=[picture], return_tensors="pt"
            ).to(model.device)
            img_start, img_end = _image_token_span(
                inputs["input_ids"], model, processor
            )
            split = img_end + 1

            for evicted in (False, True):
                # Prefill only through the image, so the eviction happens
                # before a single filler token has been read.
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"][:, :split],
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        use_cache=True,
                    )
                cache = out.past_key_values

                if evicted:
                    keep = list(range(0, img_start)) + list(
                        range(img_end + 1, cache.key_cache[attn_ids[0]].shape[2])
                    )
                    keep_tensor = torch.tensor(keep, device=model.device)
                    for i in attn_ids:
                        cache.key_cache[i] = cache.key_cache[i].index_select(
                            2, keep_tensor
                        )
                        cache.value_cache[i] = cache.value_cache[i].index_select(
                            2, keep_tensor
                        )

                language_model = model.language_model
                with torch.no_grad():
                    for position in range(split, inputs["input_ids"].shape[1]):
                        language_model(
                            input_ids=inputs["input_ids"][:, position : position + 1],
                            past_key_values=cache,
                            use_cache=True,
                        )

                for stratum in strata:
                    projected = vt.project_state_masked(
                        cache, matrix, masks[stratum], geom
                    )
                    features[stratum].append(
                        projected.float().reshape(-1).numpy().astype(np.float16)
                    )
                records.append(
                    {
                        "trial_id": trial.trial_id,
                        "family": "identity",
                        "block_id": trial.block_id,
                        "image_id": trial.image_id,
                        "distance": distance,
                        "evicted": bool(evicted),
                        "answer_index": trial.answer_index,
                        "candidates": list(trial.candidates),
                    }
                )
        if index % 8 == 0:
            logger.info(
                "trial %d/%d, %.1f min", index, len(trials), (time.time() - started) / 60
            )

    for stratum in strata:
        base = f"/results/e5c_{tag}_{stratum}"
        with open(f"{base}.jsonl", "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        np.savez_compressed(f"{base}_features.npz", features=np.stack(features[stratum]))

    report = {
        "model_id": model_id,
        "distances": distances,
        "n_records": len(records),
        "strata": list(strata),
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(f"/results/e5c_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E5C REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def e5d(
    model_id: str = MODEL_ID,
    dynamics_path: str = "/results/e5_dynamics.npz",
    n_blocks: int = 8,
    n_features: int = 512,
    n_bins: int = 3,
    n_seeds: int = 1,
    stratify: str = "halflife",
    distances_json: str = "[0,32]",
    conditions_json: str = ('["intact","visual_all","prefill_matched",'
                            '"visual_all_zero","visual_early3_zero","visual_late3_zero"]'),
    pre_filler_tokens: int = 700,
    tag: str = "ctrl",
) -> str:
    """E5d: the controls that decide whether the relay result survives review.

    E5c showed that evicting the visual key-value entries collapses what a probe
    reads from the recurrent state at distance. The obvious objection is that the
    eviction also shortens every attention cache by about 400 positions and
    changes the softmax normalisation for everything downstream, so some of the
    drop could be the perturbation rather than the missing picture.

    The matched control needs non-visual positions to evict, and a prompt that
    begins with the image does not have any. So the prompt here carries a long
    text passage BEFORE the image. That gives two eviction sets of comparable
    size, the image positions and an equal number of pre-image text positions,
    and the difference between them isolates the image.

    The same run answers the routing question by evicting the visual entries
    from only the earliest or only the latest attention layers.

    `n_bins` controls the head stratification: 3 reproduces E5c's terciles, 10
    turns the dependence of eviction sensitivity on decay rate into a measured
    relationship rather than a three-point ordering. `n_seeds` repeats every
    projection under independent Gaussian bases so the basis-dependence of the
    probe can be reported.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from lab import decay as decay_mod
    from lab import dynamics as dyn
    from lab import materials
    from lab import vtelemetry as vt
    from lab.materials import PROBE_SUFFIX

    started = time.time()
    distances = [int(d) for d in json.loads(distances_json)]
    conditions = list(json.loads(conditions_json))
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)

    stored = np.load(dynamics_path)
    half_lives = torch.from_numpy(stored["half_life_first32"])
    if stratify == "gain_matched":
        # Splitting by half-life also splits by write amplitude, because the
        # same gate scales the write term and the decay term: the fast third
        # writes about seven times harder than the slow third. Any relay result
        # from that split could be explained by signal amplitude instead. Here
        # the gate is held inside a narrow band so write gain is matched, and
        # only |A| varies, which isolates decay from amplitude.
        fast_np, slow_np, gain_report = dyn.gain_matched_masks(
            stored["abs_a"], stored["mean_dt_first32"]
        )
        masks = {
            "fast": torch.from_numpy(fast_np),
            "slow": torch.from_numpy(slow_np),
        }
    elif n_bins == 3:
        masks = dyn.head_tercile_masks(half_lives)
        gain_report = None
    else:
        masks = dyn.head_quantile_masks(half_lives, n_bins)
        gain_report = None
    strata = sorted(masks)
    matrices = {
        s: vt.projection_matrix(n_features, geom, seed=20260812 + 977 * s,
                                device=model.device)
        for s in range(n_seeds)
    }

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=n_blocks, seed=17)
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    report: dict = {
        "model_id": model_id,
        "conditions": conditions,
        "distances": distances,
        "n_bins": n_bins,
        "n_seeds": n_seeds,
        "stratify": stratify,
        "gain_matched": gain_report,
        "strata": {
            s: dyn.stratum_summary(half_lives, masks[s]) for s in strata
        },
    }

    def build(picture, pre: str, post: str):
        """Prompt with a long text passage before the image.

        Zamba2-VL's chat template emits every image in a message before any of
        that message's text, regardless of the order the content list is
        written in, so text-before-image is impossible within one message. Two
        user messages give the ordering the template will honour, and the
        matched control needs it: without pre-image positions there is nothing
        to evict as a control for evicting the image.
        """
        messages = [
            {"role": "user", "content": [{"type": "text", "text": pre}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": post}],
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(
            text=[prompt + PROBE_SUFFIX], images=[picture], return_tensors="pt"
        ).to(model.device)

    records: list[dict] = []
    features = {(s, seed): [] for s in strata for seed in range(n_seeds)}
    checked_span = False

    for index, trial in enumerate(trials):
        picture = decay_mod.fetch_image(trial.file_name, "/results/images")
        pre = materials.filler_text(
            filler_words, pre_filler_tokens, random.Random(f"pre-{trial.trial_id}")
        )
        for distance in distances:
            post = materials.filler_text(
                filler_words, distance, random.Random(f"post-{trial.trial_id}-{distance}")
            )
            inputs = build(picture, pre, post)
            img_start, img_end = _image_token_span(inputs["input_ids"], model, processor)
            n_visual = img_end - img_start + 1
            if n_visual < 64:
                raise RuntimeError(f"implausible visual span of {n_visual} tokens")
            if img_start < n_visual:
                raise RuntimeError(
                    f"only {img_start} pre-image positions for a {n_visual}-token "
                    "matched control; increase pre_filler_tokens"
                )
            if not checked_span:
                report["example_span"] = [int(img_start), int(img_end), int(n_visual)]
                checked_span = True
            split = img_end + 1

            for condition in conditions:
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"][:, :split],
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        use_cache=True,
                    )
                cache = out.past_key_values
                total = cache.key_cache[attn_ids[0]].shape[2]

                visual = list(range(img_start, img_end + 1))
                # An equal count of pre-image text positions, taken from the end
                # of the passage so they are the most recent non-visual context,
                # which is the hardest case for the control to pass.
                matched = list(range(img_start - n_visual, img_start))

                # Zamba2's 13 attention blocks are shared and the causal mask is
                # built for one sequence length, so truncating the cache of only
                # some layers raises a shape error. Whole-cache conditions use
                # truncation, which E9 validated; per-layer conditions zero the
                # keys and values in place, which leaves every length uniform.
                # `visual_all_zero` exists to check the two methods agree before
                # any per-layer number is interpreted.
                plan = {
                    "intact": ([], (), None),
                    "visual_all": (visual, attn_ids, "truncate"),
                    "prefill_matched": (matched, attn_ids, "truncate"),
                    "visual_all_zero": (visual, attn_ids, "zero"),
                    "visual_early3_zero": (visual, attn_ids[:3], "zero"),
                    "visual_late3_zero": (visual, attn_ids[-3:], "zero"),
                }
                if condition not in plan:
                    raise ValueError(f"unknown condition {condition!r}")
                drop, layers, how = plan[condition]

                if drop and how == "truncate":
                    dropped = set(drop)
                    keep = torch.tensor(
                        [p for p in range(total) if p not in dropped],
                        device=model.device,
                    )
                    for i in layers:
                        cache.key_cache[i] = cache.key_cache[i].index_select(2, keep)
                        cache.value_cache[i] = cache.value_cache[i].index_select(2, keep)
                elif drop and how == "zero":
                    positions = torch.tensor(drop, device=model.device)
                    for i in layers:
                        cache.key_cache[i][:, :, positions, :] = 0
                        cache.value_cache[i][:, :, positions, :] = 0

                language_model = model.language_model
                with torch.no_grad():
                    for position in range(split, inputs["input_ids"].shape[1]):
                        language_model(
                            input_ids=inputs["input_ids"][:, position : position + 1],
                            past_key_values=cache,
                            use_cache=True,
                        )

                for stratum in strata:
                    for seed in range(n_seeds):
                        projected = vt.project_state_masked(
                            cache, matrices[seed], masks[stratum], geom
                        )
                        features[(stratum, seed)].append(
                            projected.float().reshape(-1).numpy().astype(np.float16)
                        )
                records.append(
                    {
                        "trial_id": trial.trial_id,
                        "family": "identity",
                        "block_id": trial.block_id,
                        "image_id": trial.image_id,
                        "distance": distance,
                        "condition": condition,
                        "n_visual": int(n_visual),
                        "n_dropped": len(drop),
                        "n_layers_evicted": len(layers),
                        "answer_index": trial.answer_index,
                    }
                )
        if index % 8 == 0:
            logger.info(
                "trial %d/%d, %.1f min", index, len(trials), (time.time() - started) / 60
            )

    for stratum in strata:
        for seed in range(n_seeds):
            suffix = f"{stratum}" if n_seeds == 1 else f"{stratum}_s{seed}"
            base = f"/results/e5d_{tag}_{suffix}"
            with open(f"{base}.jsonl", "w") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            np.savez_compressed(
                f"{base}_features.npz",
                features=np.stack(features[(stratum, seed)]),
            )

    report["n_records"] = len(records)
    report["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/e5d_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E5D REPORT\n%s", json.dumps(
        {k: v for k, v in report.items() if k != "strata"}, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image_hybrid, volumes=VOLUMES, gpu="H100", timeout=7200)
def e12_gate(model_id: str = "ibm-granite/granite-4.0-h-tiny") -> str:
    """Instrument gate for a second hybrid family, before any experiment.

    Every finding in this paper depends on reading the right slice of the right
    tensor. Zamba2's layout was verified against its source before use, and the
    same discipline applies to any new architecture: what are the layer types,
    where is the Mamba2 mixer, how wide is its input projection, what does the
    cache call its recurrent states, and which layers carry attention.

    This function answers those questions and asserts nothing, so a mismatch
    shows up as a report rather than as a plausible wrong number later.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    report: dict = {"model_id": model_id}
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    report["architectures"] = list(getattr(config, "architectures", []) or [])

    interesting = (
        "num_hidden_layers", "hidden_size", "mamba_n_heads", "mamba_num_heads",
        "mamba_d_head", "mamba_head_dim", "mamba_d_state", "mamba_n_groups",
        "mamba_expand", "expand", "attn_layer_indices", "layer_types",
        "hybrid_override_pattern", "mamba_chunk_size", "n_groups",
    )
    report["config"] = {
        key: str(getattr(config, key))[:120]
        for key in interesting
        if getattr(config, key, None) is not None
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    report["n_layers_found"] = len(layers) if layers is not None else None

    def mixer_of(layer):
        for name in ("mamba", "mixer", "self_mamba"):
            found = getattr(layer, name, None)
            if found is not None:
                return name, found
        return None, None

    kinds = []
    for index, layer in enumerate(layers):
        name, mixer = mixer_of(layer)
        has_attention = any(
            getattr(layer, attr, None) is not None
            for attr in ("self_attn", "attention", "self_attention")
        )
        kinds.append({"layer": index, "mixer_attr": name, "attention": has_attention})
    report["layer_kinds_first8"] = kinds[:8]
    report["n_with_mixer"] = sum(1 for k in kinds if k["mixer_attr"])
    report["n_with_attention"] = sum(1 for k in kinds if k["attention"])
    report["attention_layer_ids"] = [k["layer"] for k in kinds if k["attention"]]

    first = next((k["layer"] for k in kinds if k["mixer_attr"]), None)
    if first is not None:
        _, mixer = mixer_of(layers[first])
        report["mixer_type"] = type(mixer).__name__
        report["mixer_children"] = [n for n, _ in mixer.named_children()][:12]
        in_proj = getattr(mixer, "in_proj", None)
        if in_proj is not None:
            report["in_proj_out_features"] = int(in_proj.out_features)
        for attr in ("A_log", "dt_bias", "D"):
            value = getattr(mixer, attr, None)
            if value is not None:
                report[f"mixer_{attr}_shape"] = list(value.shape)

    ids = tokenizer("The code word is CRIMSON. Remember it.", return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        out = model(**ids, use_cache=True)
    cache = out.past_key_values
    report["cache_class"] = type(cache).__name__
    report["cache_attrs"] = sorted(
        a for a in dir(cache)
        if not a.startswith("_") and not callable(getattr(cache, a, None))
    )[:24]

    for attr in ("ssm_states", "conv_states", "key_cache", "value_cache"):
        value = getattr(cache, attr, None)
        if value is None:
            report[f"cache_{attr}"] = None
        elif torch.is_tensor(value):
            report[f"cache_{attr}"] = {"kind": "tensor", "shape": list(value.shape)}
        elif isinstance(value, (list, tuple)):
            shapes = [
                list(v.shape) if torch.is_tensor(v) and v.numel() else None
                for v in value
            ]
            report[f"cache_{attr}"] = {
                "kind": type(value).__name__, "len": len(value),
                "nonempty_at": [i for i, s in enumerate(shapes) if s][:16],
                "first_nonempty_shape": next((s for s in shapes if s), None),
            }
        elif isinstance(value, dict):
            keys = sorted(value)[:4]
            report[f"cache_{attr}"] = {
                "kind": "dict", "len": len(value),
                "sample": {str(k): list(value[k].shape) for k in keys},
            }

    # Recent transformers replaced the flat ssm_states/key_cache attributes with
    # a per-layer list of cache objects, so the states have to be found there.
    per_layer = getattr(cache, "layers", None)
    if per_layer is not None:
        seen: dict = {}
        for index, entry in enumerate(per_layer):
            kind = type(entry).__name__
            if kind in seen:
                seen[kind]["layers"].append(index)
                continue
            fields = {}
            for attr in sorted(dir(entry)):
                if attr.startswith("_"):
                    continue
                value = getattr(entry, attr, None)
                if torch.is_tensor(value):
                    fields[attr] = list(value.shape)
                elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                    fields[attr] = f"{type(value).__name__}[{len(value)}] first {list(value[0].shape)}"
            seen[kind] = {"layers": [index], "tensor_fields": fields}
        report["cache_layer_kinds"] = {
            k: {"layers": v["layers"][:8], "n_layers": len(v["layers"]),
                "tensor_fields": v["tensor_fields"]}
            for k, v in seen.items()
        }

    # The linear-attention cache entries exposed no tensors, so find out where
    # this implementation actually keeps the recurrent state before writing any
    # experiment against it.
    if per_layer is not None and per_layer:
        entry = per_layer[0]
        report["linear_layer_all_attrs"] = sorted(
            a for a in dir(entry)
            if not a.startswith("_") and not callable(getattr(entry, a, None))
        )
        report["linear_layer_values"] = {
            a: str(getattr(entry, a, None))[:80]
            for a in report["linear_layer_all_attrs"][:20]
        }

    import inspect as inspect_mod
    import re as re_mod

    module = inspect_mod.getmodule(type(model))
    try:
        source = inspect_mod.getsource(module)
    except Exception:  # noqa: BLE001
        source = ""
    if source:
        lines = source.splitlines()
        hits = {}
        for pattern in ("ssm_state", "conv_state", "cache_params", "Cache",
                        "update_conv_state", "update_ssm_state"):
            found = [i for i, line in enumerate(lines) if re_mod.search(pattern, line)]
            hits[pattern] = len(found)
            if found:
                lo = max(0, found[0] - 3)
                hits[f"{pattern}_first"] = "\n".join(
                    lines[j] for j in range(lo, min(len(lines), found[0] + 6))
                )[:600]
        report["source_hits"] = hits
        report["module"] = getattr(module, "__name__", "?")

    text = json.dumps(report, indent=2)
    with open(f"/results/e12_gate_{model_id.split('/')[-1]}.json", "w") as fh:
        fh.write(text)
    results_vol.commit()
    logger.info("E12 GATE\n%s", text)
    return text


@app.function(image=image_hybrid, volumes=VOLUMES, gpu="H100", timeout=86400)
def e13(
    model_id: str = "ibm-granite/granite-4.0-h-tiny",
    n_blocks: int = 8,
    n_features: int = 512,
    n_bins: int = 3,
    distances_json: str = "[0,32,64]",
    conditions_json: str = '["intact","evict_item","evict_matched"]',
    pre_filler_tokens: int = 300,
    item_repeats: int = 1,
    tag: str = "granite",
) -> str:
    """E13: does the relay finding hold in a second family, on text?

    Everything the paper claims so far comes from one model family and one
    modality. This runs the same logic on a different lineage, a different
    recurrent-to-attention ratio (36 Mamba2 layers to 4 attention layers, where
    Zamba2-VL-7B has 81 to 13), and an item that is a handful of tokens rather
    than a 345-token image.

    That last difference is the test rather than a compromise. On Zamba2 the
    intervention removed several hundred key-value positions. Here it removes
    the few positions holding a code word. If the readout of the recurrent
    state still collapses when those go, and does not collapse when an equal
    number of neighbouring text positions go, then the relay account is a
    property of the architecture rather than of large visual token blocks.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lab import dynamics as dyn
    from lab import hybrid
    from lab import textmem
    from lab import vtelemetry as vt

    started = time.time()
    distances = [int(d) for d in json.loads(distances_json)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    geom = hybrid.geometry_for(model)
    mamba_ids = hybrid.mamba_layer_ids(model, geom)
    attn_ids = geom.attention_layers
    logger.info(
        "%s: %d layers, %d with a mixer, attention at %s",
        model_id, geom.n_layers, len(mamba_ids), list(attn_ids),
    )

    # Decay parameters, read from the mixers this family actually has.
    layers = getattr(getattr(model, "model", model), "layers")
    abs_a, dt_bias = [], []
    for index in mamba_ids:
        mixer = hybrid.mamba_mixer(layers[index])
        abs_a.append(torch.exp(mixer.A_log.detach().float().cpu()))
        dt_bias.append(mixer.dt_bias.detach().float().cpu())
    params = dyn.DecayParameters(
        abs_a=torch.stack(abs_a),
        dt_bias=torch.stack(dt_bias),
        time_step_min=float(getattr(model.config, "time_step_min", 0.001)),
    )

    with open("/results/corpus/filler.txt") as fh:
        words = fh.read().split()

    def filler(n_tokens: int, rng) -> str:
        if n_tokens <= 0:
            return ""
        count = max(1, int(n_tokens * 0.75))
        start = rng.randrange(0, max(1, len(words) - count))
        return " ".join(words[start:start + count])

    trials = textmem.build_blocks(n_blocks=n_blocks, seed=17)

    # A first pass measures the gate so half-lives can be computed for this
    # model rather than borrowed from Zamba2.
    sample = trials[0]
    text = (
        f"{filler(pre_filler_tokens, random.Random(1))}\n{sample.carrier()}\n"
        f"{filler(64, random.Random(2))}{textmem.PROBE_SUFFIX}"
    )
    sink: dict = {}
    handles = []
    _, _, _, c_end = geom.offsets()
    for position, index in enumerate(mamba_ids):
        mixer = hybrid.mamba_mixer(layers[index])

        def make(slot):
            def hook(_m, _i, output):
                sink.setdefault(slot, []).append(
                    output[0, :, c_end:].detach().float().cpu()
                )
            return hook

        handles.append(mixer.in_proj.register_forward_hook(make(position)))
    try:
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    dt_raw = torch.stack([torch.cat(sink[p], dim=0) for p in range(len(mamba_ids))])
    dt_eff = dyn.effective_dt(dt_raw, params)
    mean_dt = dt_eff[:, -min(32, dt_eff.shape[1]):, :].mean(dim=1)
    half_lives = dyn.half_life_tokens(params, mean_dt)
    masks = (dyn.head_tercile_masks(half_lives) if n_bins == 3
             else dyn.head_quantile_masks(half_lives, n_bins))
    strata = sorted(masks)

    report: dict = {
        "model_id": model_id,
        "n_layers": geom.n_layers,
        "n_mamba_layers": len(mamba_ids),
        "attention_layers": list(attn_ids),
        "n_heads": geom.n_heads,
        "state_dim": geom.state_dim,
        "n_groups": geom.n_groups,
        "half_life": dyn.summarise(half_lives),
        "strata": {s: dyn.stratum_summary(half_lives, masks[s]) for s in strata},
        "distances": distances,
        "item_repeats": item_repeats,
    }
    logger.info("half-life summary\n%s", json.dumps(report["half_life"], indent=2))

    generator = torch.Generator(device="cpu").manual_seed(20260812)
    matrix = (torch.randn(geom.state_numel, n_features, generator=generator)
              / (n_features ** 0.5)).to(model.device)

    records: list[dict] = []
    features = {s: [] for s in strata}
    conditions = tuple(json.loads(conditions_json))

    for count, trial in enumerate(trials):
        pre = filler(pre_filler_tokens, random.Random(f"pre-{trial.trial_id}"))
        for distance in distances:
            post = filler(distance, random.Random(f"post-{trial.trial_id}-{distance}"))
            prefix = f"{pre}\n"
            # Repeating the carrier lengthens the item without changing what is
            # to be remembered. If probe detectability scales with how many
            # tokens an item occupies, which is the natural reading of a
            # 345-token image being readable while a one-token code word is
            # not, then a longer item should become readable at the same
            # sample size.
            carrier = " ".join([trial.carrier()] * item_repeats)
            tail = f"\n{post}{textmem.PROBE_SUFFIX}"

            prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
            carrier_ids = tokenizer(carrier, return_tensors="pt",
                                    add_special_tokens=False).input_ids
            tail_ids = tokenizer(tail, return_tensors="pt",
                                 add_special_tokens=False).input_ids
            full = torch.cat([prefix_ids, carrier_ids, tail_ids], dim=1).to(model.device)
            item_start = prefix_ids.shape[1]
            item_end = item_start + carrier_ids.shape[1] - 1
            n_item = item_end - item_start + 1
            if item_start < n_item:
                raise RuntimeError("prefix too short for a matched control")
            split = item_end + 1

            for condition in conditions:
                with torch.no_grad():
                    out = model(input_ids=full[:, :split], use_cache=True)
                cache = out.past_key_values

                if condition != "intact":
                    drop = (list(range(item_start, item_end + 1))
                            if condition == "evict_item"
                            else list(range(item_start - n_item, item_start)))
                    positions = torch.tensor(drop, device=model.device)
                    for i in attn_ids:
                        keys, values = hybrid.attention_kv(cache, i)
                        keys = keys.clone()
                        values = values.clone()
                        keys[:, :, positions, :] = 0
                        values[:, :, positions, :] = 0
                        hybrid.set_attention_kv(cache, i, keys, values)

                with torch.no_grad():
                    for position in range(split, full.shape[1]):
                        model(
                            input_ids=full[:, position : position + 1],
                            past_key_values=cache,
                            use_cache=True,
                        )

                for stratum in strata:
                    projected = hybrid.project_state_masked(
                        cache, matrix, masks[stratum], mamba_ids, geom
                    )
                    features[stratum].append(
                        projected.float().reshape(-1).numpy().astype(np.float16)
                    )
                records.append({
                    "trial_id": trial.trial_id,
                    "block_id": trial.block_id,
                    "distance": distance,
                    "condition": condition,
                    "item_repeats": item_repeats,
                    "n_item_tokens": int(n_item),
                    "answer_index": trial.answer_index,
                })
        if count % 8 == 0:
            logger.info("trial %d/%d, %.1f min", count, len(trials),
                        (time.time() - started) / 60)

    for stratum in strata:
        base = f"/results/e13_{tag}_{stratum}"
        with open(f"{base}.jsonl", "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        np.savez_compressed(f"{base}_features.npz",
                            features=np.stack(features[stratum]))

    report["n_records"] = len(records)
    report["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/e13_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E13 REPORT\n%s", json.dumps(
        {k: v for k, v in report.items() if k != "strata"}, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image_hybrid, volumes=VOLUMES, gpu="H100", timeout=7200)
def e13_behaviour(
    model_id: str = "ibm-granite/granite-4.0-h-tiny",
    n_blocks: int = 4,
    distances_json: str = "[0,32,64]",
    pre_filler_tokens: int = 300,
) -> str:
    """Can this model do the code-word task at all?

    The first E13 collection returned a probe baseline at chance, which makes
    the eviction contrast meaningless: there is no signal for the intervention
    to remove. Before blaming the probe, establish whether the model itself
    retains the code word. If behaviour is also at chance the task is wrong for
    this model and no amount of probe tuning will rescue it. If behaviour is
    high while the probe is at chance, the probe is underpowered and the fix is
    more trials and narrower features.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lab import textmem

    started = time.time()
    distances = [int(d) for d in json.loads(distances_json)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    with open("/results/corpus/filler.txt") as fh:
        words = fh.read().split()

    def filler(n_tokens: int, rng) -> str:
        if n_tokens <= 0:
            return ""
        count = max(1, int(n_tokens * 0.75))
        start = rng.randrange(0, max(1, len(words) - count))
        return " ".join(words[start:start + count])

    trials = textmem.build_blocks(n_blocks=n_blocks, seed=17)
    tally: dict = {}
    samples: list[dict] = []

    for trial in trials:
        pre = filler(pre_filler_tokens, random.Random(f"pre-{trial.trial_id}"))
        for distance in distances:
            post = filler(distance, random.Random(f"post-{trial.trial_id}-{distance}"))
            prompt = (
                f"{pre}\n{trial.carrier()}\n{post}\n{trial.question()}\nAnswer:"
            )
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **ids, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
            )
            scored = textmem.score_answer(text, trial)
            cell = tally.setdefault(str(distance), {"n": 0, "hits": 0, "unparsed": 0})
            cell["n"] += 1
            cell["hits"] += int(scored["correct"])
            cell["unparsed"] += int(scored["unparsed"])
            if len(samples) < 8:
                samples.append({
                    "distance": distance, "answer": trial.answer,
                    "raw": text.strip()[:80], "chosen": scored["chosen"],
                })

    for cell in tally.values():
        cell["accuracy"] = round(cell["hits"] / cell["n"], 4) if cell["n"] else None

    report = {
        "model_id": model_id,
        "n_trials": len(trials),
        "chance": 0.125,
        "by_distance": tally,
        "samples": samples,
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open("/results/e13_behaviour.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()
    logger.info("E13 BEHAVIOUR\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


DISTANCES = (0, 32, 64, 128, 256, 512, 1024, 2048)


def _image_token_span(input_ids, model, processor) -> tuple[int, int]:
    """Locate the contiguous run of image placeholder tokens in a prompt.

    Tries the config's declared image token id first, and falls back to the
    longest constant run, which for a vision-language prompt is the image. The
    caller asserts the recovered length against the expected token count, so a
    wrong span fails loudly rather than shifting every distance silently.
    """
    ids = input_ids[0].tolist()

    for source, name in ((model.config, "image_token_id"), (model.config, "image_token_index")):
        token_id = getattr(source, name, None)
        if token_id is None:
            continue
        positions = [i for i, t in enumerate(ids) if t == token_id]
        if positions:
            return positions[0], positions[-1]

    best = (0, 0, 0)  # (length, start, end)
    run_start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[run_start]:
            length = i - run_start
            if length > best[0]:
                best = (length, run_start, i - 1)
            run_start = i
    return best[1], best[2]


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=3600)
def e5(model_id: str = MODEL_ID, filler_tokens: int = 3000, tag: str = "7b") -> str:
    """E5: predict the forgetting horizon from the model's own parameters.

    Mamba2 multiplies whatever is already in the state by exp(dt A) at every
    token, so after k further tokens the retention factor is
    exp(A * sum of dt). A is a stored weight and dt is a function of the
    in_proj output, which means the horizon can be computed without fitting
    anything to the decay curve we measured.

    This function computes that prediction. It does not touch the probe. If the
    predicted half-lives cluster near the measured 32-token collapse, decay
    explains the result. If they are far longer, the state is not decaying but
    being overwritten by later writes, and E5b takes the interference branch.
    """
    import random
    import time

    import numpy as np
    import torch

    from lab import dynamics as dyn
    from lab import materials
    from lab import vtelemetry as vt
    from lab.decay import fetch_image
    from lab.materials import PROBE_SUFFIX

    started = time.time()
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    params = dyn.decay_parameters(model, geom)
    logger.info("loaded %s, geometry %s", model_id, geom)

    report: dict = {
        "model_id": model_id,
        "geometry": {
            "n_layers": geom.n_layers,
            "n_heads": geom.n_heads,
            "head_dim": geom.head_dim,
            "state_dim": geom.state_dim,
            "n_groups": geom.n_groups,
            "n_attention_layers": len(geom.attention_layers),
        },
        "time_step_min": params.time_step_min,
    }

    # The decay rate |A| is a stored weight, so this part needs no forward pass
    # at all and is the same for every input the model will ever see.
    abs_a = params.abs_a
    report["abs_A"] = {
        "median": float(abs_a.median()),
        "min": float(abs_a.min()),
        "max": float(abs_a.max()),
        "per_layer_median": [float(row.median()) for row in abs_a],
    }

    # Now the gate. It is input dependent, so it is measured on exactly the
    # prompt shape the decay experiment used: one image, a long filler, and the
    # same neutral readout suffix.
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()
    rng = random.Random(20260811)
    filler = materials.filler_text(filler_words, filler_tokens, rng)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=1, seed=17)

    # dt is input dependent, so it is measured over several images rather than
    # one, and the between-image spread is reported. The image is fetched at
    # its native geometry, matching the decay collection whose cliff this is
    # meant to explain; letterboxing was introduced for the splice, where cache
    # lengths must match, and is not used here.
    near_per_image, whole_per_image, retention_per_image = [], [], []
    spans, floors, extremes = [], [], []
    usable = DISTANCES

    for trial in trials:
        picture = fetch_image(trial.file_name, "/results/images")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": filler}],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[prompt + PROBE_SUFFIX], images=[picture], return_tensors="pt"
        ).to(model.device)

        sink: dict = {}
        with vt.capture_dt(model, sink, geom):
            with torch.no_grad():
                model(**inputs, use_cache=True)
        dt_raw = vt.stacked_dt(sink, geom)  # [L, seq, H]

        img_start, img_end = _image_token_span(inputs["input_ids"], model, processor)
        n_image_tokens = img_end - img_start + 1
        if n_image_tokens < 64:
            raise RuntimeError(
                f"image span of {n_image_tokens} tokens is implausible; "
                "the span detector picked the wrong run"
            )

        post_image = dt_raw[:, img_end + 1 :, :]
        spans.append((int(img_start), int(img_end), int(post_image.shape[1])))
        usable = tuple(d for d in usable if d <= post_image.shape[1])

        dt_eff = dyn.effective_dt(post_image, params)  # [L, seq, H]
        floors.append(
            float((dt_eff <= params.time_step_min + 1e-9).float().mean())
        )
        extremes.append((float(dt_eff.min()), float(dt_eff.median()), float(dt_eff.max())))

        near_per_image.append(dt_eff[:, : min(32, dt_eff.shape[1]), :].mean(dim=1))
        whole_per_image.append(dt_eff.mean(dim=1))
        retention_per_image.append(dyn.retention_curve(params, dt_eff, usable))
        logger.info("image %s done, %d post-image tokens", trial.file_name, post_image.shape[1])

    if len(usable) != len(DISTANCES):
        logger.warning("filler too short for %s", set(DISTANCES) - set(usable))
    report["distances"] = list(usable)
    report["n_images"] = len(trials)
    report["image_token_spans"] = spans
    report["effective_dt"] = {
        "median_over_images": float(torch.tensor([e[1] for e in extremes]).median()),
        "min": min(e[0] for e in extremes),
        "max": max(e[2] for e in extremes),
        "fraction_at_floor": float(torch.tensor(floors).mean()),
    }

    # Two half-life estimates: one from the gate over the first 32 tokens after
    # the image, which is the window where the measured collapse happens, and
    # one over the whole filler.
    near = torch.stack(near_per_image).mean(dim=0)
    whole = torch.stack(whole_per_image).mean(dim=0)
    retention = torch.stack([r[: len(usable)] for r in retention_per_image]).mean(dim=0)
    half_near = dyn.half_life_tokens(params, near)
    half_whole = dyn.half_life_tokens(params, whole)
    report["between_image_half_life_spread"] = {
        "min_median": float(
            min(dyn.half_life_tokens(params, n).median() for n in near_per_image)
        ),
        "max_median": float(
            max(dyn.half_life_tokens(params, n).median() for n in near_per_image)
        ),
    }

    report["half_life_first32"] = dyn.summarise(half_near)
    report["half_life_whole_filler"] = dyn.summarise(half_whole)
    report["half_life_per_layer_median"] = [float(row.median()) for row in half_near]

    report["mean_retention"] = {
        str(d): float(retention[i].mean()) for i, d in enumerate(usable)
    }
    report["median_retention"] = {
        str(d): float(retention[i].median()) for i, d in enumerate(usable)
    }
    report["fraction_heads_retaining_above_0.5"] = {
        str(d): float((retention[i] > 0.5).float().mean()) for i, d in enumerate(usable)
    }

    np.savez_compressed(
        f"/results/e5_dynamics_{tag}.npz",
        abs_a=abs_a.numpy(),
        dt_bias=params.dt_bias.numpy(),
        mean_dt_first32=near.numpy(),
        mean_dt_whole=whole.numpy(),
        half_life_first32=half_near.numpy(),
        half_life_whole=half_whole.numpy(),
        retention=retention.numpy(),
        distances=np.array(usable),
    )
    report["elapsed_minutes"] = round((time.time() - started) / 60, 2)

    with open(f"/results/e5_report_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=2)
    results_vol.commit()

    logger.info("E5 REPORT\n%s", json.dumps(report, indent=2))
    return json.dumps(report, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=86400)
def collect_decay(
    n_blocks: int = 15,
    families_json: str = "",
    tag: str = "main",
    distances_json: str = "",
    model_id: str = MODEL_ID,
) -> str:
    """The main sweep: every family at every distance, both instruments.

    Writes one JSONL of per-trial records and one NPZ of projected state
    features per family, so the decoding analysis can be redone offline
    without touching a GPU again.

    `distances_json` overrides the default grid. E5 predicts a median
    half-life of about 7 tokens, which is far inside the default grid's
    first step of 32, so resolving the predicted curve needs a finer
    sweep than the one the original collection used.
    """
    import sys
    import time

    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from lab import decay, materials, vtelemetry as vt

    families = json.loads(families_json) if families_json else [
        "identity", "count", "position"
    ]
    model, processor = vt.load_model(model_id)
    geom = vt.geometry(model)
    matrix = vt.projection_matrix(N_PROJECT_FEATURES, geom)

    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, families, n_blocks=n_blocks, seed=17)
    by_family: dict = {}
    for t in trials:
        by_family.setdefault(t.family, []).append(t)

    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    codebooks = {
        family: decay.build_codebook(model, processor, group[0].candidates, geom)
        for family, group in by_family.items()
    }

    started = time.time()
    distances = (
        tuple(json.loads(distances_json)) if distances_json else DISTANCES
    )
    manifest: dict = {"tag": tag, "n_blocks": n_blocks, "distances": list(distances)}

    for family, group in by_family.items():
        records: list[dict] = []
        features: list = []
        t0 = time.time()

        for distance in distances:
            for trial in group:
                record, scores, feats = decay.run_trial(
                    model, processor, trial, distance, filler_words, codebooks,
                    matrix, "/results/images", seed=17, geom=geom,
                )
                record["zero_shot_index"] = int(
                    torch.argmax(scores.float().mean(dim=(0, 1)))
                )
                records.append(record)
                features.append(feats.float().reshape(-1).numpy().astype(np.float16))
            logger.info(
                "%s d=%d done, %.1f min elapsed", family, distance,
                (time.time() - started) / 60,
            )

        # Pre-image control, once per trial, no distance.
        for trial in group:
            record, scores, feats = decay.run_pre_image_control(
                model, processor, trial, filler_words, codebooks, matrix, seed=17, geom=geom
            )
            record["zero_shot_index"] = int(
                torch.argmax(scores.float().mean(dim=(0, 1)))
            )
            records.append(record)
            features.append(feats.float().reshape(-1).numpy().astype(np.float16))

        base = f"/results/decay_{tag}_{family}"
        with open(f"{base}.jsonl", "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        np.savez_compressed(f"{base}_features.npz", features=np.stack(features))
        results_vol.commit()

        manifest[family] = {
            "n_records": len(records),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        logger.info("%s written: %s", family, json.dumps(manifest[family]))

    manifest["total_minutes"] = round((time.time() - started) / 60, 1)
    manifest["approx_usd"] = round(manifest["total_minutes"] / 60 * 3.95, 2)
    with open(f"/results/decay_{tag}_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    results_vol.commit()
    logger.info("MANIFEST\n%s", json.dumps(manifest, indent=2))
    return json.dumps(manifest, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=43200)
def collect_splice_v2(
    n_pairs: int = 30, distance: int = 0, families_json: str = "", tag: str = "main"
) -> str:
    """Which memory channel the answer is read from.

    The intervention has to land before the question is processed, or the
    model answers from its own memory and the swap is inert. So each run is
    prefilled with image and filler only; one channel is then replaced with
    the other run's; and only then are the question tokens fed through.

    Three conditions per pair: no swap (the control, which must follow the
    host image), recurrent swapped, attention swapped.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, vtelemetry as vt

    families = json.loads(families_json) if families_json else [
        "identity", "count", "position"
    ]
    model, processor = vt.load_model()
    geom = vt.geometry(model)
    attn_ids = vt.attention_layer_ids(model)
    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    def prefill_context(trial, filler):
        """Image plus filler, with no question yet."""
        img = decay.fetch_image(trial.file_name, "/results/images")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": filler}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        return out.past_key_values

    def ask_through(cache, trial, max_new_tokens: int = 10) -> str:
        """Feed the question into an existing cache and decode greedily.

        The image already lives in the cache, so the continuation goes through
        the language model directly. Calling the vision-language wrapper with
        text alone fails, since it expects pixel inputs on every call.
        """
        question = trial.question()
        ids = processor.tokenizer(
            question, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        language_model = model.language_model

        # Zamba2's naive path only accepts one token at a time once the cache
        # holds a previous state: torch_forward does input_states.squeeze(1)
        # under has_previous_state, so a multi-token feed makes the gate 4D.
        with torch.no_grad():
            out = None
            for position in range(ids.shape[1]):
                out = language_model(
                    input_ids=ids[:, position:position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
            generated = []
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(max_new_tokens):
                generated.append(int(token))
                out = language_model(
                    input_ids=token, past_key_values=cache, use_cache=True
                )
                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    rng = random.Random(23)
    records: list[dict] = []
    started = time.time()

    for family in families:
        trials = materials.build_trials(coco, [family], n_blocks=20, seed=23)
        by_answer: dict = {}
        for t in trials:
            by_answer.setdefault(t.answer, []).append(t)
        answers = sorted(by_answer)

        for index in range(n_pairs):
            a, b = rng.sample(answers, 2)
            host, donor = rng.choice(by_answer[a]), rng.choice(by_answer[b])
            filler = decay.filler_text(filler_words, distance, random.Random(index))

            donor_cache = prefill_context(donor, filler)
            row = {
                "family": family,
                "pair": index,
                "host_answer": host.answer,
                "donor_answer": donor.answer,
                "distance": distance,
            }

            for condition in ("none", "recurrent", "attention"):
                cache = prefill_context(host, filler)
                if condition == "none":
                    text = ask_through(cache, host)
                else:
                    with vt.channel_swapped(cache, donor_cache, condition, attn_ids, geom):
                        text = ask_through(cache, host)
                scored = decay.score_answer(text, host)
                row[condition] = {
                    "raw": text,
                    "chosen": scored["chosen"],
                    "follows_host": scored["chosen"] == host.answer,
                    "follows_donor": scored["chosen"] == donor.answer,
                }
            records.append(row)

            if index % 10 == 0:
                logger.info(
                    "%s pair %d/%d, %.1f min", family, index, n_pairs,
                    (time.time() - started) / 60,
                )

    path = f"/results/splice_{tag}.jsonl"
    with open(path, "w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")

    summary: dict = {"n_pairs_per_family": n_pairs, "distance": distance}
    for family in families:
        rows = [r for r in records if r["family"] == family]
        summary[family] = {
            condition: {
                "follows_host": sum(r[condition]["follows_host"] for r in rows),
                "follows_donor": sum(r[condition]["follows_donor"] for r in rows),
                "n": len(rows),
            }
            for condition in ("none", "recurrent", "attention")
        }
    summary["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/splice_{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    results_vol.commit()
    logger.info("SPLICE SUMMARY\n%s", json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=43200)
def collect_splice(n_pairs: int = 40, distance: int = 0, tag: str = "main") -> str:
    """Which channel carries the picture.

    Prefill two runs that differ only in the image, then give one run the
    other's recurrent state (all 81 layers) while leaving its 13 attention
    caches intact, and the reverse. Whichever image the answer follows is the
    channel the picture was actually read from.
    """
    import random
    import sys
    import time

    sys.path.insert(0, "/root")
    import torch

    from lab import decay, materials, vtelemetry as vt

    model, processor = vt.load_model()
    geom = vt.geometry(model)
    coco = materials.load_coco("/results/corpus/instances_val2017.json")
    trials = materials.build_trials(coco, ["identity"], n_blocks=20, seed=23)
    with open("/results/corpus/filler.txt") as fh:
        filler_words = fh.read().split()

    rng = random.Random(23)
    attn_ids = vt.attention_layer_ids(model)
    pairs = []
    by_answer: dict = {}
    for t in trials:
        by_answer.setdefault(t.answer, []).append(t)
    answers = sorted(by_answer)
    while len(pairs) < n_pairs:
        a, b = rng.sample(answers, 2)
        ta, tb = rng.choice(by_answer[a]), rng.choice(by_answer[b])
        pairs.append((ta, tb))

    def prefill(trial, filler):
        image = decay.fetch_image(trial.file_name, "/results/images")
        prompt = decay._question_prompt(
            processor, filler,
            "Which one of these is in the picture: "
            + ", ".join(trial.candidates)
            + "? Answer with one option.",
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        return inputs, out.past_key_values

    def answer_from(cache, inputs):
        with torch.no_grad():
            gen = model.generate(
                **inputs, past_key_values=cache, max_new_tokens=10, do_sample=False
            )
        return processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    records = []
    started = time.time()
    for index, (ta, tb) in enumerate(pairs):
        filler = decay.filler_text(filler_words, distance, random.Random(index))
        inputs_a, cache_a = prefill(ta, filler)
        _inputs_b, cache_b = prefill(tb, filler)

        row = {
            "pair": index,
            "host_answer": ta.answer,
            "donor_answer": tb.answer,
            "candidates": list(ta.candidates),
            "distance": distance,
        }
        for channel in ("recurrent", "attention"):
            snapshot_a = vt.snapshot_recurrent(cache_a, geom)
            with vt.channel_swapped(cache_a, cache_b, channel, attn_ids, geom):
                text = answer_from(cache_a, inputs_a)
            vt.restore_recurrent(cache_a, snapshot_a)
            lowered = text.lower()
            row[channel] = {
                "raw": text,
                "follows_host": ta.answer.lower() in lowered,
                "follows_donor": tb.answer.lower() in lowered,
            }
        records.append(row)
        if index % 10 == 0:
            logger.info("splice pair %d/%d", index, len(pairs))

    path = f"/results/splice_{tag}.jsonl"
    with open(path, "w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")
    results_vol.commit()

    summary = {"n_pairs": len(records), "distance": distance}
    for channel in ("recurrent", "attention"):
        summary[channel] = {
            "follows_host": sum(r[channel]["follows_host"] for r in records),
            "follows_donor": sum(r[channel]["follows_donor"] for r in records),
        }
    summary["minutes"] = round((time.time() - started) / 60, 1)
    with open(f"/results/splice_{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    results_vol.commit()
    logger.info("SPLICE SUMMARY\n%s", json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)
