"""
Schema definitions and validation utilities for tool calling.
Uses Pydantic v2 to enforce strict typing, validation, and serialization.
"""

from typing import Literal, Union, List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import json
import re


class DeployServiceParams(BaseModel):
    service: str = Field(description="Name of the microservice (e.g. auth-service, payment-api)")
    version: str = Field(description="Release tag or semantic version (e.g. v2.4.1, commit hash)")
    environment: Literal["production", "staging", "development"] = Field(
        description="Target deployment environment"
    )
    replicas: int = Field(default=2, ge=1, le=100, description="Number of replica instances")
    notify_channels: List[str] = Field(
        default_factory=list, description="Slack or webhook notification channels (e.g. ['#deployments'])"
    )


class RestartPodParams(BaseModel):
    pod_name: str = Field(description="Identifier or exact name of the kubernetes pod")
    region: str = Field(description="Cloud region (e.g. us-east-1, eu-west-1, ap-northeast-1)")
    force: bool = Field(default=False, description="Whether to kill gracefully or forcefully immediately")
    reason: str = Field(description="Operational rationale for restart (e.g. high memory leak, unresponsive)")


class RollbackDeploymentParams(BaseModel):
    deployment_id: str = Field(description="Deployment identifier (e.g. dep-9821, auth-v2)")
    target_tag: str = Field(description="Target version tag to restore (e.g. v1.8.4)")
    drain_traffic: bool = Field(
        default=True, description="Whether to drain live traffic before terminating current instances"
    )


class ScaleClusterParams(BaseModel):
    cluster_name: str = Field(description="Name of the Kubernetes or compute cluster (e.g. eks-core-prod)")
    node_count: int = Field(ge=1, le=500, description="Desired total number of worker nodes")
    auto_scale: bool = Field(default=True, description="Enable cluster autoscaler daemon")
    instance_type: str = Field(default="m6i.xlarge", description="Cloud instance size or VM type")


class ToolCall(BaseModel):
    tool: Literal["deploy_service", "restart_pod", "rollback_deployment", "scale_cluster"]
    parameters: Union[DeployServiceParams, RestartPodParams, RollbackDeploymentParams, ScaleClusterParams]


SYSTEM_PROMPT = """You are an automated Cloud Infrastructure Action Dispatcher.
Your task is to convert operational requests into strictly structured tool calls.

Available Tools:
1. deploy_service:
   Parameters:
   - service (str): Microservice name
   - version (str): Target version/tag
   - environment (str: "production" | "staging" | "development")
   - replicas (int: 1-100, default 2)
   - notify_channels (list of str, default [])

2. restart_pod:
   Parameters:
   - pod_name (str): Exact pod name
   - region (str): Cloud region (e.g. us-east-1, us-west-2, eu-west-1)
   - force (bool: default false)
   - reason (str): Operational rationale

3. rollback_deployment:
   Parameters:
   - deployment_id (str): Deployment name or ID
   - target_tag (str): Target version to restore
   - drain_traffic (bool: default true)

4. scale_cluster:
   Parameters:
   - cluster_name (str): Cluster identifier
   - node_count (int: 1-500)
   - auto_scale (bool: default true)
   - instance_type (str: default "m6i.xlarge")

CRITICAL INSTRUCTIONS:
- You must respond ONLY with a single valid JSON object containing "tool" and "parameters".
- Do NOT wrap your response in markdown code blocks (no ```json or ```).
- Do NOT add any preamble, conversational greeting, explanation, or follow-up text.
- Strictly adhere to parameter names and types."""


PARAM_MODEL_MAP = {
    "deploy_service": DeployServiceParams,
    "restart_pod": RestartPodParams,
    "rollback_deployment": RollbackDeploymentParams,
    "scale_cluster": ScaleClusterParams,
}


def parse_and_validate(raw_text: str) -> Dict[str, Any]:
    """
    Evaluates raw output from the model.
    Returns:
        {
            "raw_text": str,
            "is_pure_json": bool (True if output was clean JSON without markdown fences/chatter),
            "is_valid_json": bool (True if parseable JSON anywhere),
            "is_schema_valid": bool (True if adheres to Pydantic ToolCall),
            "parsed_data": dict | None,
            "error": str | None
        }
    """
    clean_text = raw_text.strip()
    is_pure_json = False
    is_valid_json = False
    is_schema_valid = False
    parsed_data = None
    error_msg = None

    # Check 1: Is it pure JSON (no markdown backticks, starts with { and ends with })?
    if clean_text.startswith("{") and clean_text.endswith("}") and "```" not in clean_text:
        try:
            parsed_data = json.loads(clean_text)
            is_pure_json = True
            is_valid_json = True
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"

    # Check 2: If not pure JSON, can we extract JSON from markdown fences?
    if not is_valid_json:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            try:
                parsed_data = json.loads(json_match.group(1))
                is_valid_json = True
            except json.JSONDecodeError as e:
                error_msg = f"Extracted JSON decode error: {str(e)}"
        else:
            # Try searching for any outermost { ... }
            bracket_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if bracket_match:
                try:
                    parsed_data = json.loads(bracket_match.group(1))
                    is_valid_json = True
                except json.JSONDecodeError as e:
                    error_msg = f"Loose JSON parse error: {str(e)}"

    # Check 3: Schema validation against Pydantic model
    if is_valid_json and isinstance(parsed_data, dict):
        tool_name = parsed_data.get("tool")
        params = parsed_data.get("parameters")
        if tool_name in PARAM_MODEL_MAP and isinstance(params, dict):
            try:
                param_model = PARAM_MODEL_MAP[tool_name]
                validated_params = param_model(**params)
                is_schema_valid = True
                parsed_data["parameters"] = validated_params.model_dump()
            except ValidationError as ve:
                error_msg = f"Schema validation error: {ve.errors()[0]['msg']} on {ve.errors()[0]['loc']}"
        else:
            error_msg = f"Invalid or missing tool name: '{tool_name}'"

    return {
        "raw_text": raw_text,
        "is_pure_json": is_pure_json,
        "is_valid_json": is_valid_json,
        "is_schema_valid": is_schema_valid,
        "parsed_data": parsed_data,
        "error": error_msg,
    }
