"""
Dataset Generation Script for MLX Fine-Tuning & Evaluation.

Generates realistic cloud infrastructure operational requests and their ground-truth
structured tool calls formatted in ChatML / OpenAI JSONL format for MLX:
{"messages": [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_request},
    {"role": "assistant", "content": json_output}
]}

Outputs:
- data/train.jsonl (for LoRA fine-tuning)
- data/valid.jsonl (for validation during training)
- data/test.jsonl  (untouched holdout set for comprehensive evaluations)
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import SYSTEM_PROMPT

SERVICES = ["auth-api", "billing-worker", "payment-service", "user-mgmt", "notification-hub", "search-indexer", "recommendation-v2", "cart-service", "gateway-proxy", "inventory-db"]
ENVIRONMENTS = [
    ("production", ["prod", "production", "live cluster", "primary prod"]),
    ("staging", ["staging", "stage", "pre-prod", "qa-staging"]),
    ("development", ["dev", "development", "sandbox", "test env"]),
]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"]
CHANNELS = ["#deployments", "#infra-alerts", "#core-eng", "#ops-room", "#release-tracker"]
POD_NAMES = ["postgres-primary-0", "redis-cluster-cache-3", "kafka-broker-1", "celery-worker-7f89b", "ingress-controller-4a11", "clickhouse-replica-2"]
INSTANCE_TYPES = ["m6i.large", "m6i.xlarge", "c6i.2xlarge", "r6i.4xlarge", "t4g.xlarge"]


def generate_deploy_example() -> Dict[str, Any]:
    svc = random.choice(SERVICES)
    major = random.randint(1, 4)
    minor = random.randint(0, 15)
    patch = random.randint(0, 9)
    version = f"v{major}.{minor}.{patch}"
    env_target, env_phrases = random.choice(ENVIRONMENTS)
    env_phrase = random.choice(env_phrases)
    replicas = random.choice([1, 2, 3, 5, 8, 10])
    has_channels = random.random() > 0.3
    channels = random.sample(CHANNELS, k=random.randint(1, 2)) if has_channels else []

    templates = [
        f"Please deploy {svc} version {version} to {env_phrase} with {replicas} replicas.",
        f"Ship {version} of {svc} into {env_phrase}. Spin up {replicas} instances and alert {', '.join(channels)}." if channels else f"Ship {version} of {svc} into {env_phrase} with {replicas} instances.",
        f"Roll out release {version} for {svc} on {env_phrase} (replicas: {replicas}).",
        f"Trigger a deployment of {svc} ({version}) in {env_phrase}. Make sure {replicas} replicas are running." + (f" Notify {channels[0]}." if channels else ""),
        f"Deploy {svc} {version} to {env_phrase} immediately. Set replica count to {replicas}."
    ]
    prompt = random.choice(templates)
    ground_truth = {
        "tool": "deploy_service",
        "parameters": {
            "service": svc,
            "version": version,
            "environment": env_target,
            "replicas": replicas,
            "notify_channels": channels,
        }
    }
    return {"prompt": prompt, "completion": json.dumps(ground_truth, separators=(',', ':'))}


def generate_restart_example() -> Dict[str, Any]:
    pod = random.choice(POD_NAMES)
    region = random.choice(REGIONS)
    force = random.choice([True, False])
    reasons = [
        ("OOMKilled memory leak", ["out of memory", "high RAM consumption", "OOM spike"]),
        ("Unresponsive healthcheck timeout", ["failing liveness probes", "timeout errors", "unresponsive"]),
        ("Connection pool exhaustion", ["db pool starved", "too many open connections", "max socket limit"]),
        ("Configuration reload", ["apply updated secret", "flush local config cache"]),
    ]
    reason_clean, reason_phrases = random.choice(reasons)
    reason_phrase = random.choice(reason_phrases)

    templates = [
        f"Pod {pod} in {region} is {reason_phrase}. {'Force restart it now' if force else 'Gracefully reboot it'}.",
        f"Please restart {pod} located in {region}. Reason: {reason_clean}. {'Immediate force kill' if force else 'Allow graceful shutdown'}.",
        f"We have an issue in {region}: {pod} is experiencing {reason_phrase}. Issue a {'forced' if force else 'standard'} restart.",
        f"Restart pod {pod} ({region}) due to {reason_phrase}." + (" Force restart requested." if force else ""),
    ]
    prompt = random.choice(templates)
    ground_truth = {
        "tool": "restart_pod",
        "parameters": {
            "pod_name": pod,
            "region": region,
            "force": force,
            "reason": reason_clean,
        }
    }
    return {"prompt": prompt, "completion": json.dumps(ground_truth, separators=(',', ':'))}


def generate_rollback_example() -> Dict[str, Any]:
    dep_id = f"dep-{random.randint(1000, 9999)}"
    target_tag = f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 5)}"
    drain = random.choice([True, False])

    templates = [
        f"Roll back deployment {dep_id} to tag {target_tag}. {'Drain active traffic first.' if drain else 'Skip traffic draining and swap immediately.'}",
        f"Revert {dep_id} back to {target_tag}. {'Ensure connections are drained' if drain else 'Do not wait for drain, hard cutover'}.",
        f"Execute rollback on {dep_id} restoring version {target_tag} ({'drain_traffic=true' if drain else 'no traffic draining'}).",
        f"Rollback {dep_id} to previous good state {target_tag}. {'Gracefully terminate after draining.' if drain else 'Immediate cutover without draining.'}",
    ]
    prompt = random.choice(templates)
    ground_truth = {
        "tool": "rollback_deployment",
        "parameters": {
            "deployment_id": dep_id,
            "target_tag": target_tag,
            "drain_traffic": drain,
        }
    }
    return {"prompt": prompt, "completion": json.dumps(ground_truth, separators=(',', ':'))}


def generate_scale_example() -> Dict[str, Any]:
    cluster = f"k8s-{random.choice(['prod', 'staging', 'compute'])}-cluster-{random.choice(['alpha', 'beta', 'west', 'east'])}"
    nodes = random.choice([5, 10, 20, 50, 80, 120])
    autoscale = random.choice([True, False])
    inst = random.choice(INSTANCE_TYPES)

    templates = [
        f"Scale cluster {cluster} to {nodes} worker nodes using {inst}. {'Keep autoscaling active.' if autoscale else 'Disable autoscaler.'}",
        f"Adjust capacity for {cluster}: target {nodes} nodes of type {inst}. {'Enable cluster autoscaler' if autoscale else 'Turn off autoscaling'}.",
        f"Resize {cluster} to {nodes} nodes ({inst}). {'Autoscale enabled.' if autoscale else 'Fixed size without autoscaling.'}",
        f"We need more compute in {cluster}. Scale to {nodes} nodes ({inst}). {'Ensure autoscaling is on.' if autoscale else 'Autoscaler should be disabled.'}",
    ]
    prompt = random.choice(templates)
    ground_truth = {
        "tool": "scale_cluster",
        "parameters": {
            "cluster_name": cluster,
            "node_count": nodes,
            "auto_scale": autoscale,
            "instance_type": inst,
        }
    }
    return {"prompt": prompt, "completion": json.dumps(ground_truth, separators=(',', ':'))}


def build_samples(n: int) -> List[Dict[str, Any]]:
    generators = [generate_deploy_example, generate_restart_example, generate_rollback_example, generate_scale_example]
    samples = []
    for _ in range(n):
        gen = random.choice(generators)
        samples.append(gen())
    return samples


def to_chat_format(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for s in samples:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["completion"]},
            ]
        })
    return formatted


def save_jsonl(records: List[Dict[str, Any]], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    random.seed(42)
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic infrastructure dispatch dataset...")
    # Training split: 200 samples
    train_samples = build_samples(200)
    # Validation split: 40 samples
    valid_samples = build_samples(40)
    # Holdout test split: 60 samples
    test_samples = build_samples(60)

    save_jsonl(to_chat_format(train_samples), output_dir / "train.jsonl")
    save_jsonl(to_chat_format(valid_samples), output_dir / "valid.jsonl")
    save_jsonl(to_chat_format(test_samples), output_dir / "test.jsonl")

    # Also save raw test samples for quick evaluation inspection
    with open(output_dir / "raw_test_samples.json", "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)

    print(f"Dataset successfully created in '{output_dir}':")
    print(f"  - train.jsonl: {len(train_samples)} samples")
    print(f"  - valid.jsonl: {len(valid_samples)} samples")
    print(f"  - test.jsonl:  {len(test_samples)} samples")
    print(f"  - raw_test_samples.json: {len(test_samples)} samples")


if __name__ == "__main__":
    main()
