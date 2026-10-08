"""
Schema definitions and validation utilities for tool calling.
Uses Pydantic v2 to enforce strict typing, validation, and serialization.
"""

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class DeployServiceParams(StrictModel):
    service: str = Field(description="Name of the microservice (e.g. auth-service, payment-api)")
    version: str = Field(description="Release tag or semantic version (e.g. v2.4.1, commit hash)")
    environment: Literal["production", "staging", "development"] = Field(
        description="Target deployment environment"
    )
    replicas: int = Field(default=2, ge=1, le=100, description="Number of replica instances")
    notify_channels: list[str] = Field(
        default_factory=list, description="Slack or webhook notification channels (e.g. ['#deployments'])"
    )


class RestartPodParams(StrictModel):
    pod_name: str = Field(description="Identifier or exact name of the kubernetes pod")
    region: str = Field(description="Cloud region (e.g. us-east-1, eu-west-1, ap-northeast-1)")
    force: bool = Field(default=False, description="Whether to kill gracefully or forcefully immediately")
    reason: str = Field(description="Operational rationale for restart (e.g. high memory leak, unresponsive)")


class RollbackDeploymentParams(StrictModel):
    deployment_id: str = Field(description="Deployment identifier (e.g. dep-9821, auth-v2)")
    target_tag: str = Field(description="Target version tag to restore (e.g. v1.8.4)")
    drain_traffic: bool = Field(
        default=True, description="Whether to drain live traffic before terminating current instances"
    )


class ScaleClusterParams(StrictModel):
    cluster_name: str = Field(description="Name of the Kubernetes or compute cluster (e.g. eks-core-prod)")
    node_count: int = Field(ge=1, le=500, description="Desired total number of worker nodes")
    auto_scale: bool = Field(default=True, description="Enable cluster autoscaler daemon")
    instance_type: str = Field(default="m6i.xlarge", description="Cloud instance size or VM type")


class NoActionParams(StrictModel):
    reason: Literal["unsupported_request", "missing_required_parameter"] = Field(
        description="Why no tool call is made: no tool performs the request, or a required parameter is absent"
    )


class DeployServiceCall(StrictModel):
    tool: Literal["deploy_service"]
    parameters: DeployServiceParams


class RestartPodCall(StrictModel):
    tool: Literal["restart_pod"]
    parameters: RestartPodParams


class RollbackDeploymentCall(StrictModel):
    tool: Literal["rollback_deployment"]
    parameters: RollbackDeploymentParams


class ScaleClusterCall(StrictModel):
    tool: Literal["scale_cluster"]
    parameters: ScaleClusterParams


class NoActionCall(StrictModel):
    tool: Literal["no_action"]
    parameters: NoActionParams


class ToolCall(RootModel[Annotated[
    DeployServiceCall | RestartPodCall | RollbackDeploymentCall | ScaleClusterCall | NoActionCall,
    Field(discriminator="tool"),
]]):
    """Bind each tool name to exactly its own parameter schema."""


PARAM_MODEL_MAP = {
    "deploy_service": DeployServiceParams,
    "restart_pod": RestartPodParams,
    "rollback_deployment": RollbackDeploymentParams,
    "scale_cluster": ScaleClusterParams,
    "no_action": NoActionParams,
}


def build_system_prompt():
    lines = [
        "You are an automated Cloud Infrastructure Action Dispatcher.",
        "Convert operational requests into strictly structured tool calls.",
        "Available tools and parameters:",
    ]
    for name, model in PARAM_MODEL_MAP.items():
        schema = model.model_json_schema()
        lines.append(name + ":")
        for field, info in schema["properties"].items():
            constraints = {k: v for k, v in info.items() if k not in ("title", "description")}
            if field == "notify_channels":
                constraints["default"] = []
            lines.append(f"- {field}: {json.dumps(constraints)}. {info.get('description', '')}")
    lines.extend([
        "Respond ONLY with one JSON object containing tool and parameters.",
        "No markdown fences, preamble, explanations, or extra fields. Use exact parameter types.",
        "Include all parameters, using documented defaults when omitted in the request.",
        "For restart_pod.reason, copy the reason phrase exactly as written in the request, without surrounding sentence punctuation.",
        "Only notify channels explicitly requested; otherwise use an empty list.",
        "If no tool performs the request, or a required parameter (one without a default) is missing, "
        "respond with no_action instead of guessing a value.",
    ])
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite floating-point range")
    return number


def _invalid_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _first_embedded_object(decoder, text):
    """Return the first decodable JSON object in surrounding chatter.

    Braces that do not start valid JSON (e.g. "use {tool}") are skipped. Duplicate
    keys and nonstandard constants still fail, since they are not JSONDecodeErrors.
    """
    start = text.find("{")
    while start >= 0:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found")


def parse_and_validate(raw_text: str) -> dict[str, Any]:
    """Validate without modifying decoded data; normalized defaults are separate.

    A recoverable object inside chatter counts as valid JSON but not pure JSON.
    "Valid JSON" means a JSON *object* was recovered: bare scalars and arrays
    are treated the same as no JSON at all, since the task contract is one object.
    Duplicate keys and nonstandard numeric constants are rejected.
    """
    result = {
        "raw_text": raw_text, "is_pure_json": False, "is_valid_json": False,
        "is_schema_valid": False, "parsed_data": None, "normalized_data": None,
        "error": None,
    }
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant, parse_float=_finite_float)
    text = raw_text.strip()
    try:
        try:
            parsed = decoder.decode(text)
            is_pure = isinstance(parsed, dict)
        except json.JSONDecodeError:
            parsed = _first_embedded_object(decoder, text)
            is_pure = False
        result["parsed_data"] = parsed
        result["is_pure_json"] = is_pure
        result["is_valid_json"] = isinstance(parsed, dict)
        validated = ToolCall.model_validate(parsed)
        result["normalized_data"] = validated.model_dump()
        result["is_schema_valid"] = True
    except (ValueError, RecursionError) as exc:
        result["error"] = str(exc)
    return result
