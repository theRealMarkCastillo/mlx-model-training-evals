"""Generate the synthetic tool-calling dataset: standard splits plus challenge sets.

Every record is a chat triple (system, user, assistant) that MLX-LM trains on
directly, plus a `meta` field the evaluator uses to slice results:

    {"messages": [...], "meta": {"tool": ..., "family": 2, "omitted": ["replicas"], ...}}

Two generalization axes are controlled separately:

- Template family: wording. Train uses families 0-1, valid 2, test 3.
- Entity pool: names and values. Standard splits share one pool; the
  `challenge_entities` set uses names never seen in training.

Challenge sets isolate one difficulty each:

- challenge_entities: training wording, unseen services/pods/regions/etc.
- challenge_defaults: test wording, every optional parameter omitted.
- challenge_abstain:  kinds of unsupported requests never seen in training.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.runs import REPO_ROOT
from src.schema import PARAM_MODEL_MAP, SYSTEM_PROMPT, ToolCall

DATA_DIR = REPO_ROOT / "data"
SEED = 42
TRAIN_FAMILIES, VALID_FAMILIES, TEST_FAMILIES = (0, 1), (2,), (3,)
OMIT_PROBABILITY = 0.3


@dataclass(frozen=True)
class EntityPool:
    services: tuple
    pods: tuple
    regions: tuple
    channels: tuple
    instance_types: tuple
    cluster_names: tuple
    major_versions: tuple
    reasons: tuple
    deployment_id: str  # format string with {n} and {svc}


SEEN = EntityPool(
    services=("auth-api", "billing-worker", "payment-service", "user-mgmt", "notification-hub",
              "search-indexer", "recommendation-v2", "cart-service", "gateway-proxy", "inventory-db"),
    pods=("postgres-primary-0", "redis-cluster-cache-3", "kafka-broker-1", "celery-worker-7f89b",
          "ingress-controller-4a11", "clickhouse-replica-2"),
    regions=("us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"),
    channels=("#deployments", "#infra-alerts", "#core-eng", "#ops-room", "#release-tracker"),
    instance_types=("m6i.large", "m6i.xlarge", "c6i.2xlarge", "r6i.4xlarge", "t4g.xlarge"),
    cluster_names=tuple(f"k8s-{a}-cluster-{b}" for a in ("prod", "staging", "compute")
                        for b in ("alpha", "beta", "west", "east")),
    major_versions=(1, 2, 3, 4),
    reasons=("out of memory", "high RAM consumption", "OOM spike", "failing liveness probes",
             "timeout errors", "unresponsive", "db pool starved", "too many open connections",
             "max socket limit", "apply updated secret", "flush local config cache"),
    deployment_id="dep-{n}",
)

UNSEEN = EntityPool(
    services=("ledger-sync", "media-transcoder", "fraud-scorer", "geo-router", "session-cache", "email-dispatcher"),
    pods=("etcd-member-2", "nginx-edge-5c7d", "mongo-shard-1", "vault-agent-0"),
    regions=("sa-east-1", "ca-central-1", "ap-south-1"),
    channels=("#sre-oncall", "#platform-updates", "#incident-room"),
    instance_types=("m7g.large", "c7i.4xlarge", "r7g.2xlarge"),
    cluster_names=tuple(f"eks-{a}-{b}" for a in ("batch", "edge") for b in ("gamma", "north", "blue")),
    major_versions=(5, 6, 7, 8, 9),
    reasons=("disk pressure", "stuck in CrashLoopBackOff", "certificate expired", "zombie processes"),
    deployment_id="{svc}-r{n}",
)

ENVIRONMENTS = (
    ("production", ("prod", "production", "live cluster", "primary prod")),
    ("staging", ("staging", "stage", "pre-prod", "qa-staging")),
    ("development", ("dev", "development", "sandbox", "test env")),
)

# Unsupported request kinds. Training sees the first group; challenge_abstain the second.
SEEN_UNSUPPORTED = (
    "delete the {svc} database in {env}",
    "rotate the TLS certificates for {svc}",
    "show me the last 100 log lines from {pod}",
    "create a read-only IAM user for the {svc} team",
    "open a PagerDuty incident for {svc}",
)
UNSEEN_UNSUPPORTED = (
    "take a snapshot of the {svc} volume in {region}",
    "add a DNS record pointing {svc}.internal at the new load balancer",
    "report the current error rate of {svc}",
    "increase the memory limit of {pod} to 4Gi",
)
UNSUPPORTED_FRAMES = ("Please {a}.", "Can you {a}?", "I need you to {a}.", "{A}.")


def _version(rng, pool):
    return f"v{rng.choice(pool.major_versions)}.{rng.randint(0, 15)}.{rng.randint(0, 9)}"


def _choose_state(rng, omit_p):
    """True, False, or None (omitted, so the schema default applies)."""
    return None if rng.random() < omit_p else rng.choice([True, False])


def deploy(rng, pool, family, omit_p):
    svc, version = rng.choice(pool.services), _version(rng, pool)
    env, env_phrases = rng.choice(ENVIRONMENTS)
    env_phrase = rng.choice(env_phrases)
    replicas = None if rng.random() < omit_p else rng.choice([1, 2, 3, 5, 8, 10])
    channels = [] if rng.random() < max(omit_p, 0.3) else rng.sample(pool.channels, k=rng.randint(1, 2))

    def rep(text):
        return text.format(n=replicas) if replicas is not None else ""

    prompt = (
        f"Please deploy {svc} version {version} to {env_phrase}{rep(' with {n} replicas')}.",
        f"Ship {version} of {svc} into {env_phrase}{rep(' with {n} instances')}.",
        f"Roll out release {version} for {svc} on {env_phrase}{rep(' (replicas: {n})')}.",
        f"Trigger a deployment of {svc} ({version}) in {env_phrase}.{rep(' Make sure {n} replicas are running.')}",
    )[family]
    if channels:
        prompt += f" Notify {', '.join(channels)}."
    omitted = [name for name, absent in (("replicas", replicas is None), ("notify_channels", not channels)) if absent]
    params = {"service": svc, "version": version, "environment": env,
              "replicas": 2 if replicas is None else replicas, "notify_channels": channels}
    return prompt, params, omitted


def restart(rng, pool, family, omit_p):
    pod, region, reason = rng.choice(pool.pods), rng.choice(pool.regions), rng.choice(pool.reasons)
    force = _choose_state(rng, omit_p)
    base, phrases = (
        (f"Pod {pod} in {region} has a problem: {reason}.", (" Force restart it now.", " Gracefully reboot it.", " Restart it.")),
        (f"Please restart {pod} located in {region}. Reason: {reason}.", (" Immediate force kill.", " Allow graceful shutdown.", "")),
        (f"We have an issue in {region}: {pod} needs a restart ({reason}).", (" Issue a forced restart.", " Issue a standard restart.", " Please restart it.")),
        (f"Restart pod {pod} ({region}) due to {reason}.", (" Force restart requested.", " Use a graceful restart.", "")),
    )[family]
    prompt = base + phrases[{True: 0, False: 1, None: 2}[force]]
    params = {"pod_name": pod, "region": region, "force": bool(force), "reason": reason}
    return prompt, params, ["force"] if force is None else []


def rollback(rng, pool, family, omit_p):
    svc = rng.choice(pool.services)
    dep_id = pool.deployment_id.format(n=rng.randint(1000, 9999), svc=svc)
    tag = _version(rng, pool)
    drain = _choose_state(rng, omit_p)
    base, phrases = (
        (f"Roll back deployment {dep_id} to tag {tag}.", (" Drain active traffic first.", " Skip traffic draining and swap immediately.", "")),
        (f"Revert {dep_id} back to {tag}.", (" Ensure connections are drained.", " Do not wait for drain, hard cutover.", "")),
        (f"Execute rollback on {dep_id} restoring version {tag}.", (" Drain traffic before switching.", " No traffic draining.", "")),
        (f"Rollback {dep_id} to previous good state {tag}.", (" Gracefully terminate after draining.", " Immediate cutover without draining.", "")),
    )[family]
    prompt = base + phrases[{True: 0, False: 1, None: 2}[drain]]
    params = {"deployment_id": dep_id, "target_tag": tag, "drain_traffic": drain is not False}
    return prompt, params, ["drain_traffic"] if drain is None else []


def scale(rng, pool, family, omit_p):
    cluster, nodes = rng.choice(pool.cluster_names), rng.choice([5, 10, 20, 50, 80, 120])
    auto = _choose_state(rng, omit_p)
    inst = None if rng.random() < omit_p else rng.choice(pool.instance_types)

    def typ(text):
        return text.format(i=inst) if inst else ""

    base, phrases = (
        (f"Scale cluster {cluster} to {nodes} worker nodes{typ(' using {i}')}.", (" Keep autoscaling active.", " Disable autoscaler.", "")),
        (f"Adjust capacity for {cluster}: target {nodes} nodes{typ(' of type {i}')}.", (" Enable cluster autoscaler.", " Turn off autoscaling.", "")),
        (f"Resize {cluster} to {nodes} nodes{typ(' ({i})')}.", (" Autoscale enabled.", " Fixed size without autoscaling.", "")),
        (f"We need more compute in {cluster}. Scale to {nodes} nodes{typ(' ({i})')}.", (" Ensure autoscaling is on.", " Autoscaler should be disabled.", "")),
    )[family]
    prompt = base + phrases[{True: 0, False: 1, None: 2}[auto]]
    omitted = [name for name, absent in (("auto_scale", auto is None), ("instance_type", inst is None)) if absent]
    params = {"cluster_name": cluster, "node_count": nodes, "auto_scale": auto is not False,
              "instance_type": inst or "m6i.xlarge"}
    return prompt, params, omitted


def _unsupported(rng, pool, family, kinds):
    env = rng.choice(rng.choice(ENVIRONMENTS)[1])
    action = rng.choice(kinds).format(svc=rng.choice(pool.services), env=env,
                                      pod=rng.choice(pool.pods), region=rng.choice(pool.regions))
    return UNSUPPORTED_FRAMES[family].format(a=action, A=action[0].upper() + action[1:])


def _missing_required(rng, pool, family):
    """A request for a real tool that leaves out a required parameter."""
    svc, pod, cluster = rng.choice(pool.services), rng.choice(pool.pods), rng.choice(pool.cluster_names)
    env = rng.choice(rng.choice(ENVIRONMENTS)[1])
    reason = rng.choice(pool.reasons)
    dep_id = pool.deployment_id.format(n=rng.randint(1000, 9999), svc=svc)
    options = (
        (  # deploy_service without version
            f"Please deploy {svc} to {env}.", f"Ship {svc} into {env}.",
            f"Roll out {svc} on {env}.", f"Trigger a deployment of {svc} in {env}.",
        ),
        (  # restart_pod without region
            f"Pod {pod} has a problem: {reason}. Restart it.", f"Please restart {pod}. Reason: {reason}.",
            f"We have an issue: {pod} needs a restart ({reason}).", f"Restart pod {pod} due to {reason}.",
        ),
        (  # rollback_deployment without target_tag
            f"Roll back deployment {dep_id}.", f"Revert {dep_id}.",
            f"Execute rollback on {dep_id}.", f"Rollback {dep_id} to the last version.",
        ),
        (  # scale_cluster without node_count
            f"Scale cluster {cluster} up.", f"Adjust capacity for {cluster}.",
            f"Resize {cluster}.", f"We need more compute in {cluster}.",
        ),
    )
    return rng.choice(options)[family]


def no_action(rng, pool, family, omit_p, *, kinds=SEEN_UNSUPPORTED, missing=True):
    if missing and rng.random() < 0.5:
        return _missing_required(rng, pool, family), {"reason": "missing_required_parameter"}, []
    return _unsupported(rng, pool, family, kinds), {"reason": "unsupported_request"}, []


def unseen_unsupported(rng, pool, family, omit_p):
    return no_action(rng, pool, family, omit_p, kinds=UNSEEN_UNSUPPORTED, missing=False)


TOOL_GENERATORS = {
    "deploy_service": deploy, "restart_pod": restart, "rollback_deployment": rollback,
    "scale_cluster": scale, "no_action": no_action,
}
ACTION_TOOLS = ("deploy_service", "restart_pod", "rollback_deployment", "scale_cluster")


def build_samples(rng, n, seen, *, families, pool=SEEN, omit_p=OMIT_PROBABILITY,
                  tools=tuple(TOOL_GENERATORS), generators=None):
    """Balanced across tools, prompts unique across every set built with `seen`."""
    generators = generators or {tool: TOOL_GENERATORS[tool] for tool in tools}
    names = list(generators)
    samples = []
    for i in range(n):
        tool = names[i % len(names)]
        for _ in range(10000):
            family = rng.choice(families)
            prompt, params, omitted = generators[tool](rng, pool, family, omit_p)
            if prompt not in seen:
                break
        else:
            raise ValueError(f"Unable to generate enough unique {tool} prompts")
        seen.add(prompt)
        target = {"tool": tool if tool in PARAM_MODEL_MAP else "no_action", "parameters": params}
        ToolCall.model_validate(target)
        samples.append({
            "prompt": prompt,
            "completion": json.dumps(target, separators=(",", ":")),
            "meta": {"tool": target["tool"], "family": family, "omitted": omitted,
                     "entities": "unseen" if pool is UNSEEN else "seen"},
        })
    rng.shuffle(samples)
    return samples


def build_splits(seed=SEED):
    rng = random.Random(seed)
    seen = set()
    return {
        "train": build_samples(rng, 250, seen, families=TRAIN_FAMILIES),
        "valid": build_samples(rng, 50, seen, families=VALID_FAMILIES),
        "test": build_samples(rng, 75, seen, families=TEST_FAMILIES),
        "challenge_entities": build_samples(rng, 40, seen, families=TRAIN_FAMILIES, pool=UNSEEN, tools=ACTION_TOOLS),
        "challenge_defaults": build_samples(rng, 40, seen, families=TEST_FAMILIES, omit_p=1.0, tools=ACTION_TOOLS),
        "challenge_abstain": build_samples(rng, 40, seen, families=(0, 1, 2, 3),
                                           generators={"unseen_unsupported": unseen_unsupported}),
    }


def to_chat_format(samples):
    return [{
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["prompt"]},
            {"role": "assistant", "content": s["completion"]},
        ],
        "meta": s["meta"],
    } for s in samples]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# General-capability set: the forgetting check's yardstick.
#
# These records are deliberately NOT tool calls and use a plain assistant system prompt,
# because the question is whether the adapter that learned the ops contract still handles
# ordinary requests. They are scored by assistant loss only (there is no single correct
# string to match), so short, unambiguous reference answers are enough.
# --------------------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = "You are a helpful assistant."

GENERAL_PROMPTS = (
    ("What is 17 times 24?", "17 times 24 is 408."),
    ("Add 2, 5, and 11.", "2 + 5 + 11 = 18."),
    ("Is 91 a prime number?", "No, 91 is not prime: it equals 7 times 13."),
    ("What is 15% of 240?", "15% of 240 is 36."),
    ("What comes next: 2, 6, 12, 20, 30, ...?", "The next number is 42; the differences are 4, 6, 8, 10, 12."),
    ("Capital of Australia?", "The capital of Australia is Canberra."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen."),
    ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
    ("Name the largest ocean on Earth.", "The largest ocean on Earth is the Pacific Ocean."),
    ("Summarize in one sentence: the team shipped late, and the customer churned.",
     "The team shipped late and the customer churned as a result."),
    ("Rewrite politely: send me the file now.", "Could you please send me the file when you have a moment?"),
    ("Translate to Spanish: good morning.", "Buenos dias."),
    ("Translate to French: thank you very much.", "Merci beaucoup."),
    ("Classify the sentiment: the review was dull and overpriced.", "Negative."),
    ("Classify the sentiment: fast delivery and the product works well.", "Positive."),
    ("Write a Python one-liner that reverses a string s.", "The expression s[::-1] returns the string reversed."),
    ("What does the SQL keyword JOIN do?", "JOIN combines rows from two tables using a related column."),
    ("Explain what an HTTP 404 status means.", "HTTP 404 means the server could not find the requested resource."),
    ("What is a deadlock in computing?", "A deadlock is when two or more processes each wait for a resource the other holds, so none can proceed."),
    ("Give one advantage of unit tests.", "Unit tests catch regressions early and document expected behaviour."),
    ("What is the difference between RAM and a hard disk?",
     "RAM is fast volatile working memory; a hard disk is slower persistent storage."),
    ("How many minutes are in two and a half hours?", "Two and a half hours is 150 minutes."),
    ("Sort these numbers ascending: 9, 2, 14, 7.", "Ascending order: 2, 7, 9, 14."),
)


def build_general_set():
    """Chat records with free-form answers, used only by the forgetting check."""
    return [{
        "messages": [
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"tool": None, "family": None, "omitted": [], "entities": "general"},
    } for prompt, answer in GENERAL_PROMPTS]


def main(output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    splits = build_splits()
    for name, samples in splits.items():
        save_jsonl(to_chat_format(samples), output_dir / f"{name}.jsonl")
        print(f"  {name + '.jsonl':26} {len(samples):4} records")
    save_jsonl(build_general_set(), output_dir / "general.jsonl")
    print(f"  {'general.jsonl':26} {len(GENERAL_PROMPTS):4} records (for the forgetting check)")
    print(f"Dataset written to {output_dir}")
    return splits


if __name__ == "__main__":
    main()
