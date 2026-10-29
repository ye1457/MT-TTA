"""Portable, audited full-stream evaluation of the selected MT-TTA methods."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .protocol import (canonical_hash, file_sha256, load_config, prepare_data,
                       read_json, require, require_new_output, save_json)
from .routing import route_private_experts_for_task, routing_mode


def prompt_hash(session):
    digest = hashlib.sha256()
    for name in ("shared_bank", "private_bank", "visual_scaffold", "visual_shared_bank", "visual_private_bank"):
        module = getattr(session, name)
        if module is not None:
            for parameter_name, parameter in module.named_parameters():
                digest.update((name + "." + parameter_name).encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def session_kwargs(config, student, preprocess, teacher, teacher_preprocess, reference_spec, branch):
    method = config["baseline_method"]
    text, visual, adaptation = method["text_prompt"], method["visual_prompt"], method["adaptation"]
    visual_on = branch == "primary"
    return dict(
        model=student, preprocess=preprocess, reference_domain_spec=reference_spec,
        num_workers=0 if config["benchmark"] == "pacs" else 4, device="cuda:0",
        num_context=text["num_context"], prompt_init=text["initialization"],
        num_shared_experts=text["shared_experts"], num_private_experts=text["private_experts"],
        expert_init_std=text["expert_init_std"], seed=config["evaluation"]["seed"],
        num_visual_prompts=visual["num_tokens"] if visual_on else 0,
        visual_prompt_init_std=visual["position_extra_noise_std"], visual_prompt_scale_init=visual["scale_init"],
        num_visual_shared_experts=visual["shared_experts"] if visual_on else 0,
        num_visual_private_experts=visual["private_experts"] if visual_on else 0,
        visual_router_source=visual["router_source"], teacher_model=teacher,
        teacher_preprocess=teacher_preprocess, teacher_device="cuda:0",
        teacher_template=adaptation["teacher_template"], teacher_temperature=adaptation["teacher_temperature"],
    )


def evaluation_kwargs(config, spec):
    method, data = config["baseline_method"], config["data"]
    text, visual, adaptation = method["text_prompt"], method["visual_prompt"], method["adaptation"]
    return dict(
        domain_spec=spec, tta_lr=data["domain_learning_rate_overrides"].get(spec.target_domain, data["learning_rate"]),
        pseudo_topk=adaptation["student_pseudo_topk"], update_steps=adaptation["update_rounds_per_selected_image"],
        entropy_weight=adaptation["entropy_weight"], confidence_threshold=data["student_confidence_threshold"],
        shared_mix_weight=text["shared_mix_weight"], topk_private_experts=text["private_topk"],
        routing_buffer_size=text["preview_size"], routing_score=routing_mode(text["router_score"]),
        router_temperature=text["router_temperature"], shared_anchor_weight=adaptation["shared_anchor_weight"],
        visual_topk_private_experts=visual["private_topk"], visual_shared_mix_weight=visual["shared_mix_weight"],
        visual_router_temperature=visual["router_temperature"], teacher_supervision=adaptation["teacher_supervision"],
        teacher_ce_weight=adaptation["teacher_ce_weight"], distill_kl_weight=0.0, weight_decay=adaptation["weight_decay"],
    )


def make_session_class(core, config, records, progress_callback=None):
    """Add observation and selected text routing without changing adaptation math."""
    method = config["baseline_method"]
    steps = method["adaptation"]["update_rounds_per_selected_image"]
    threshold = config["data"]["student_confidence_threshold"]

    class AuditedSession(core.PersistentMoEPromptTTASession):
        def _route_private_experts_for_task(self, domain_spec, **kwargs):
            before = prompt_hash(self)
            require(self.scored == self.optimizer_rounds == 0, "Preview must precede scoring and updates")
            mode = routing_mode(method["text_prompt"]["router_score"])
            require(kwargs["routing_score"] == mode, "Runtime routing differs from the configuration")
            route = (super()._route_private_experts_for_task(domain_spec=domain_spec, **kwargs) if mode == "confidence"
                     else route_private_experts_for_task(self, domain_spec=domain_spec, **kwargs))
            require(prompt_hash(self) == before, "Preview changed prompt parameters")
            n = min(kwargs["routing_buffer_size"], len(domain_spec.examples))
            self.route_audit = {
                "preview_count": n, "before_prompt_hash": before, "after_prompt_hash": prompt_hash(self),
                "preview_relative_paths": [r["relative_path"] for r in records[self.active_domain][:n]],
                "selected_ids": list(route.selected_ids), "selected_weights": route.selected_weights.detach().cpu().tolist(),
                "routing_scores": list(route.routing_scores), "score_mode": mode, "routing_score_dtype": str(self.model.dtype),
            }
            return route

        def _prepare_student_forward(self, *args, **kwargs):
            result = super()._prepare_student_forward(*args, **kwargs)
            probabilities = result[0].detach().softmax(dim=-1)
            prediction = int(probabilities.argmax(dim=-1).item())
            confidence = float(probabilities.max(dim=-1).values.item())
            record = records[self.active_domain][self.scored]
            require(self.optimizer_rounds == self.selected * steps, "Prediction must precede the current image's updates")
            self.trace.write(json.dumps({
                "branch": self.branch, "domain": self.active_domain, "index": self.scored,
                "relative_path": record["relative_path"], "image_sha256": record["sha256"], "label": record["label_index"],
                "student_logits": result[0].detach().cpu()[0].tolist(), "student_probabilities": probabilities.cpu()[0].tolist(),
                "prediction": prediction, "confidence": confidence, "selected": confidence >= threshold,
                "optimizer_rounds_before_prediction": self.optimizer_rounds,
                "prediction_source": "student_only", "scoring": "pre_update",
            }, allow_nan=False) + "\n")
            self.scored += 1
            self.correct += int(prediction == record["label_index"])
            self.selected += int(confidence >= threshold)
            if self.scored % 100 == 0 or self.scored == len(records[self.active_domain]):
                self.trace.flush()
                if progress_callback is not None:
                    progress_callback(self)
            return result

    return AuditedSession


class OptimizerMonitor:
    """Temporarily observe the core's per-domain optimizer constructors."""
    def __init__(self, core, config, metadata):
        self.core, self.config, self.metadata = core, config, metadata
        self.active_session = None

    def capture(self, optimizer, kind):
        session = self.active_session
        require(session is not None and not optimizer.state, "Optimizer must be new with empty state each domain")
        data = self.config["data"]
        lr = data["domain_learning_rate_overrides"].get(session.active_domain, data["learning_rate"])
        for group in optimizer.param_groups:
            require(group["lr"] == lr and group["weight_decay"] == 0.0 and
                    group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8, "Wrong AdamW settings")
        self.metadata["optimizer_creations"].append({"branch": session.branch, "domain": session.active_domain,
            "kind": kind, "initial_state_entries": len(optimizer.state), "lr": lr,
            "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8})
        if kind == "main":
            def after_step(*unused):
                session.optimizer_rounds += 1
            optimizer.register_step_post_hook(after_step)
        return optimizer

    def __enter__(self):
        self.original_main = self.core.build_optimizer_for_active_experts
        self.original_private = self.core.build_private_visual_optimizers
        def main(*args, **kwargs):
            return self.capture(self.original_main(*args, **kwargs), "main")
        def private(*args, **kwargs):
            optimizers = self.original_private(*args, **kwargs)
            return {index: self.capture(optimizer, f"visual_private_{index}") for index, optimizer in optimizers.items()}
        self.core.build_optimizer_for_active_experts = main
        self.core.build_private_visual_optimizers = private
        return self

    def __exit__(self, *unused):
        self.core.build_optimizer_for_active_experts = self.original_main
        self.core.build_private_visual_optimizers = self.original_private


def execute_schedule(config, specs, records, sessions, monitor, output, metadata):
    """Carry each existing branch session across the declared domain schedule."""
    data, method = config["data"], config["baseline_method"]
    previous = dict(metadata["initial_prompt_hashes"])
    results = []
    for index, (branch, domain) in enumerate(data["branch_domain_schedule"]):
        session = sessions[branch]
        monitor.active_session = session
        session.active_domain, session.schedule_index = domain, index
        session.scored = session.correct = session.selected = session.optimizer_rounds = 0
        session.route_audit = None
        initial = prompt_hash(session)
        require(initial == previous[branch], "Prompt parameters were reset between domains")
        require(all(prompt_hash(s) == previous[b] for b, s in sessions.items()), "One branch modified another branch's prompts")
        trace_name = f"{branch}.{domain}.predictions.jsonl"
        started = time.time()
        print("DOMAIN_START " + json.dumps({"branch": branch, "domain": domain}), flush=True)
        with (output / trace_name).open("x") as trace:
            session.trace = trace
            result = session.evaluate_domain(**evaluation_kwargs(config, specs[domain]))
        require(session.scored == result.num_images == len(records[domain]), "Incomplete domain scoring")
        require(session.correct / session.scored == result.top1_accuracy and session.selected == result.num_selected_samples,
                "Core score/gate differs from pre-update student records")
        require(session.optimizer_rounds == result.num_updates == session.selected * method["adaptation"]["update_rounds_per_selected_image"], "Wrong update count")
        require(result.routing_buffer_size == min(method["text_prompt"]["preview_size"], len(records[domain])) and
                result.teacher_supervision_mode == "aux_pseudolabel" and session.route_audit is not None, "Wrong preview or teacher role")
        creations = [o for o in metadata["optimizer_creations"] if o["branch"] == branch and o["domain"] == domain]
        expected = {"main"} | ({f"visual_private_{i}" for i in range(method["visual_prompt"]["private_experts"])} if branch == "primary" else set())
        require(len(creations) == len(expected) and {o["kind"] for o in creations} == expected, "Optimizers were not recreated for the domain")
        previous[branch] = prompt_hash(session)
        reported = data["reported_branch"][domain] == branch
        metadata["domain_states"].append({"branch": branch, "domain": domain, "is_reported": reported,
            "initial_prompt_hash": initial, "final_prompt_hash": previous[branch], "num_scored": session.scored,
            "correct": session.correct, "selected": session.selected, "update_rounds": session.optimizer_rounds,
            "trace_file": trace_name, "trace_sha256": file_sha256(output / trace_name), "route_audit": session.route_audit,
            "elapsed_seconds": time.time() - started})
        results.append({"branch": branch, "domain": domain, "is_reported": reported, "result": result.to_dict()})
        save_json(output / "metadata.json", metadata)
        save_json(output / "domain_results.json", results)
        print("DOMAIN_DONE " + json.dumps({"branch": branch, "domain": domain, "reported": reported,
                                            "student_top1_percent": result.top1_accuracy * 100}), flush=True)
    return results


def recount_trace(path, records, branch, domain, num_classes, threshold, update_steps):
    """Model-free recount of every aligned pre-update student prediction."""
    selected = correct = count = 0
    with Path(path).open() as handle:
        for index, (line, record) in enumerate(zip(handle, records, strict=True)):
            row = json.loads(line)
            require(row["branch"] == branch and row["domain"] == domain and row["index"] == index and
                    row["relative_path"] == record["relative_path"] and row["image_sha256"] == record["sha256"] and
                    row["label"] == record["label_index"], "Trace stream identity mismatch")
            require(row["prediction_source"] == "student_only" and row["scoring"] == "pre_update", "Invalid prediction source/timing")
            probs, logits = row["student_probabilities"], row["student_logits"]
            require(len(probs) == len(logits) == num_classes and
                    all(math.isfinite(v) for v in probs + logits) and all(0 <= v <= 1 for v in probs), "Invalid student output")
            maximum = max(logits)
            exponentials = [math.exp(v - maximum) for v in logits]
            denominator = sum(exponentials)
            require(max(abs(p - e / denominator) for p, e in zip(probs, exponentials)) <= 0.001,
                    "Probabilities do not match the recorded student logits")
            prediction = max(range(num_classes), key=probs.__getitem__)
            require(row["prediction"] == prediction and row["confidence"] == max(probs), "Student argmax/confidence mismatch")
            gate = max(probs) >= threshold
            require(row["selected"] is gate and row["optimizer_rounds_before_prediction"] == selected * update_steps,
                    "Update gate or scoring timing mismatch")
            count += 1
            correct += prediction == record["label_index"]
            selected += gate
    return {"num_images": count, "correct": correct, "selected": selected, "update_rounds": selected * update_steps}


def summarize(output, config, metadata, records):
    data, method = config["data"], config["baseline_method"]
    rows, previews = [], 0
    for state in metadata["domain_states"]:
        path = output / state["trace_file"]
        require(file_sha256(path) == state["trace_sha256"], "Trace changed before completion")
        stats = recount_trace(path, records[state["domain"]], state["branch"], state["domain"], data["num_classes"],
                              data["student_confidence_threshold"], method["adaptation"]["update_rounds_per_selected_image"])
        require(stats["num_images"] == state["num_scored"] and all(stats[k] == state[k] for k in ("correct", "selected", "update_rounds")), "Trace/core counts disagree")
        rows.append({"branch": state["branch"], "domain": state["domain"], "is_reported": state["is_reported"],
                     **stats, "top1_percent": stats["correct"] / stats["num_images"] * 100})
        previews += state["route_audit"]["preview_count"]
    reported = [r for r in rows if r["is_reported"]]
    require([r["domain"] for r in reported] == data["domain_order"] and
            sum(r["num_images"] for r in reported) == data["num_scored_images"], "Incomplete formal benchmark")
    require(sum(r["num_images"] for r in rows) == data["formal_branch_image_visits_excluding_prerouting"] and
            previews == data["preview_image_visits"], "Wrong extra branch/preview compute")
    return {"status": "PASS", "benchmark": config["benchmark"], "seed": 0,
            "prediction_rule": "pre_update_student_only", "teacher_output_fusion": False,
            "num_scored_images": data["num_scored_images"], "domains": reported, "branch_results": rows,
            "macro_top1_percent": sum(r["top1_percent"] for r in reported) / len(reported),
            "micro_top1_percent": sum(r["correct"] for r in reported) / data["num_scored_images"] * 100,
            "preview_image_visits": previews, "formal_branch_image_visits_excluding_prerouting": sum(r["num_images"] for r in rows),
            "branch_selected_samples": sum(r["selected"] for r in rows), "branch_update_rounds": sum(r["update_rounds"] for r in rows),
            "optimizer_instances": len(metadata["optimizer_creations"]), "elapsed_seconds": metadata["elapsed_seconds"],
            "prompt_state_persistent": True, "fresh_optimizers_each_domain": True,
            "all_student_predictions_recounted": True, "preview_parameter_updates": 0,
            **({"independent_benchmark_initialization": config["evaluation"]["independent_benchmark_initialization"],
                "benchmark_state_transfer": config["evaluation"]["benchmark_state_transfer"]}
               if "independent_benchmark_initialization" in config["evaluation"] else {}),
            "fresh_unseen_test_claim": False}


def run(config, data_root, model_cache, gpu, output_dir):
    require_new_output(output_dir)
    specs, records, inventory = prepare_data(config, data_root, hash_images=True)
    return run_prepared(config, specs, records, inventory, model_cache, gpu, output_dir)


def run_prepared(config, specs, records, inventory, model_cache, gpu, output_dir,
                 session_factory=make_session_class):
    """Run validated dataset adapters through the common audited method."""
    output = require_new_output(output_dir)
    require(str(gpu).isdigit(), "--gpu must identify a single CUDA GPU")
    model_cache = Path(model_cache).expanduser().resolve()
    method = config["baseline_method"]
    model_hashes, model_paths = {}, {}
    for role in ("student", "teacher"):
        path = model_cache / method[role + "_weights"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached {role} weights: {path}. Download the official OpenAI CLIP weights to --model-cache first.")
        digest = file_sha256(path)
        require(digest == method[role + "_weights_sha256"], "Model weights do not match the selected configuration")
        model_hashes[path.name] = digest
        model_paths[role] = path
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    import torchvision
    from . import moe_prompt_tta as core
    from .clip_utils import import_clip, load_clip_model
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Full evaluation requires one visible CUDA GPU; CPU fallback is disabled")
    torch.set_num_threads(1)
    clip_dir = Path(import_clip().__file__).resolve().parent
    sources = {"clip_officehome/" + p.name: p for p in Path(__file__).resolve().parent.glob("*.py")}
    sources.update({"clip/" + p.name: p for p in clip_dir.iterdir() if p.suffix in {".py", ".gz"}})
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    save_json(output / "input_manifest.json", {"inventory": inventory, "records": records})
    started = time.time()
    metadata = {"status": "RUNNING", "benchmark": config["benchmark"], "config": config,
        "config_sha256": canonical_hash(config), "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "data": inventory, "model_hashes": model_hashes,
        "model_paths": {role: str(path) for role, path in model_paths.items()}, "source_hashes": source_hashes,
        "gpu": str(gpu), "gpu_name": torch.cuda.get_device_name(0),
        "versions": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__},
        "teacher_output_fusion": False, "teacher_output_calibration": False,
        "output_rule": "student_probability_argmax_only", "initial_prompt_hashes": {}, "optimizer_creations": [], "domain_states": []}
    save_json(output / "metadata.json", metadata)
    try:
        # Load precisely the files whose bytes were checked above. Passing model
        # registry names here could otherwise resolve a different cached file.
        student, preprocess = load_clip_model(str(model_paths["student"]), "cuda:0", model_cache)
        teacher, teacher_preprocess = load_clip_model(str(model_paths["teacher"]), "cuda:0", model_cache)
        require(all(file_sha256(path) == method[role + "_weights_sha256"] for role, path in model_paths.items()),
                "Model weight files changed while loading")
        def progress(session):
            value = {"branch": session.branch, "domain": session.active_domain, "scored": session.scored,
                     "total": len(records[session.active_domain]), "elapsed_seconds": time.time() - started,
                     "completed_branch_image_visits": sum(s["num_scored"] for s in metadata["domain_states"])}
            save_json(output / "progress.json", value)
            print("PROGRESS " + json.dumps(value), flush=True)
        session_class = session_factory(core, config, records, progress)
        branch_order = ["primary"] + (["text_shadow"] if config["data"]["shadow_text_domains"] else [])
        sessions = {}
        for branch in branch_order:
            session = session_class(**session_kwargs(config, student, preprocess, teacher, teacher_preprocess,
                                                    specs[config["data"]["domain_order"][0]], branch))
            session.branch = branch
            sessions[branch] = session
            metadata["initial_prompt_hashes"][branch] = prompt_hash(session)
        require(all(not p.requires_grad for model in (student, teacher) for p in model.parameters()), "Backbones must remain frozen")
        require(student.visual.input_resolution == teacher.visual.input_resolution == 224, "Unexpected model resolution")
        metadata.update(branch_initialization_order=branch_order,
                        preprocess={"student": str(preprocess), "teacher": str(teacher_preprocess)})
        with OptimizerMonitor(core, config, metadata) as monitor:
            execute_schedule(config, specs, records, sessions, monitor, output, metadata)
        require(all(file_sha256(path) == source_hashes[name] for name, path in sources.items()), "Source files changed during evaluation")
        metadata.update(elapsed_seconds=time.time() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        summary = summarize(output, config, metadata, records)
        metadata["status"] = "COMPLETE"
        save_json(output / "metadata.json", metadata)
        save_json(output / "summary.json", summary)
        save_json(output / "completion.json", {"status": "PASS", "config_sha256": canonical_hash(config),
            "summary_sha256": file_sha256(output / "summary.json"), "metadata_sha256": file_sha256(output / "metadata.json"),
            "domain_results_sha256": file_sha256(output / "domain_results.json"),
            "input_manifest_sha256": file_sha256(output / "input_manifest.json")})
        return summary
    except BaseException as error:
        metadata.update(status="FAILED", error=repr(error), elapsed_seconds=time.time() - started)
        save_json(output / "metadata.json", metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("pacs", "officehome"), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing exactly the four full domain directories")
    parser.add_argument("--config", type=Path, help="Explicit configuration; default is the installed selected benchmark config")
    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache/clip", help="Directory containing official cached ViT-B-16.pt and ViT-L-14.pt")
    parser.add_argument("--gpu", default="0", help="One physical CUDA GPU index")
    parser.add_argument("--output-dir", type=Path, help="New result directory, required for model evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate full membership, classes and order without models, GPU or output writes")
    parser.add_argument("--hash-images", action="store_true", help="With --dry-run, also hash every image and report ordered content digests")
    args = parser.parse_args(argv)
    if args.hash_images and not args.dry_run:
        parser.error("--hash-images is only valid with --dry-run; formal evaluation already hashes images")
    config = load_config(args.benchmark, args.config)
    if args.dry_run:
        _, _, inventory = prepare_data(config, args.data_root, hash_images=args.hash_images)
        result = {"status": "DRY_RUN_PASS_NO_MODEL", "config_sha256": canonical_hash(config), "config": config, "data": inventory}
    else:
        require(args.output_dir is not None, "--output-dir is required for a model evaluation")
        result = run(config, args.data_root, args.model_cache, args.gpu, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
