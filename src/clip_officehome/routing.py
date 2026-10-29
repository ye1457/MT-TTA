"""Student-only fixed text routing for explicit PACS method experiments.

Confidence uses the original core implementation. Alternative scores affect
expert selection and weights: the preview images, text context composition, frozen image
features, Top-K selection and fixed softmax route retain their original form.
Alternative scores use float32 arithmetic (including zero-safe entropy logs).
Each score is then quantized to model dtype before recording and before Top-K
and softmax, matching the original router's precision at those operations.
"""
import math


SCORE_MODES = {
    "mean_max_class_probability_across_preview_images": "confidence",
    "mean_normalized_neg_entropy": "neg_entropy",
    "mean_probability_margin": "margin",
}


def routing_mode(config_score):
    """Translate the explicit method-config score name to its runtime name."""
    try:
        return SCORE_MODES[config_score]
    except KeyError as error:
        raise ValueError(f"Unsupported text router score: {config_score!r}") from error


def score_probabilities(probabilities, mode):
    """Return one scalar score from an [images, classes] probability tensor.

    This helper uses no labels, teacher outputs or mutable prompt parameters.
    The returned tensor is useful for CPU unit tests; the router converts it to
    a Python float before constructing its model-dtype expert-score tensor.
    """
    import torch

    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("Router probabilities require nonempty [images, classes>=2] shape")
    if mode == "confidence":
        return probabilities.max(dim=-1).values.mean()
    if mode == "margin":
        top_two = probabilities.float().topk(k=2, dim=-1, largest=True, sorted=True).values
        return (top_two[:, 0] - top_two[:, 1]).mean()
    if mode == "neg_entropy":
        values = probabilities.float()
        log_values = values.clamp_min(torch.finfo(values.dtype).tiny).log()
        entropy = -(values * log_values).sum(dim=-1)
        return (1.0 - entropy / math.log(probabilities.shape[-1])).mean()
    raise ValueError(f"Unsupported router runtime mode: {mode!r}")


def route_private_experts_for_task(
    session, domain_spec, topk_private_experts, routing_buffer_size,
    routing_score, router_temperature, shared_mix_weight,
):
    """Route once from the first preview images, without updates or labels."""
    import torch
    from clip_officehome import moe_prompt_tta as core

    if routing_score not in {"neg_entropy", "margin"}:
        raise ValueError("Confidence routing must delegate directly to the frozen core")
    if len(session.private_bank) == 0:
        return core.RoutedPrivateExperts(
            selected_ids=[], selected_weights=torch.empty(0, device=session.device, dtype=session.model.dtype),
            routing_scores=[],
        )
    warmup_examples = domain_spec.examples[:min(routing_buffer_size, len(domain_spec.examples))]
    if not warmup_examples:
        raise ValueError("Text route preview must contain at least one image")
    # Keep image collection identical to the baseline. Images are encoded by the
    # frozen student without visual prompts; returned features are detached.
    image_features = core.collect_image_features(
        model=session.model, preprocess=session.preprocess, examples=warmup_examples,
        num_workers=session.num_workers, device=session.device, batch_size=min(64, len(warmup_examples)),
    )
    with torch.no_grad():
        shared_context = core.compose_mean_context(session.shared_bank.get_contexts())
        scores = []
        for expert_index in range(len(session.private_bank)):
            private_context = session.private_bank.get_context(expert_index)
            effective_context = core.compose_effective_context(
                shared_context, private_context, shared_mix_weight=shared_mix_weight,
            )
            text_features = session.scaffold.encode_text_features(session.model, effective_context)
            logits = core.CLIP_LOGIT_SCALE * image_features @ text_features.t()
            probabilities = logits.softmax(dim=-1)
            score = score_probabilities(probabilities, routing_score).to(dtype=session.model.dtype)
            scores.append(float(score.item()))
        score_tensor = torch.tensor(scores, device=image_features.device, dtype=session.model.dtype)
        actual_topk = min(max(int(topk_private_experts), 1), len(session.private_bank))
        selected_scores, selected_indices = torch.topk(score_tensor, k=actual_topk, largest=True, sorted=True)
        selected_weights = torch.softmax(selected_scores / router_temperature, dim=0)
    return core.RoutedPrivateExperts(
        selected_ids=selected_indices.tolist(), selected_weights=selected_weights, routing_scores=scores,
    )
