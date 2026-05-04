from __future__ import annotations

import json
from typing import Any


def empty_object_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": True}


def schema_identity(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)


def merge_object_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    properties_by_name: dict[str, list[dict[str, Any]]] = {}
    required_sets: list[set[str]] = []
    additional_properties = False

    for schema in schemas:
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, value in properties.items():
                if isinstance(value, dict):
                    properties_by_name.setdefault(name, []).append(value)
        required = schema.get("required")
        if isinstance(required, list):
            required_sets.append({item for item in required if isinstance(item, str)})
        else:
            required_sets.append(set())
        if schema.get("additionalProperties", True) is not False:
            additional_properties = True

    merged: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: merge_schemas(property_schemas)
            for name, property_schemas in sorted(properties_by_name.items())
        },
        "additionalProperties": additional_properties,
    }
    if required_sets:
        required = sorted(set.intersection(*required_sets))
        if required:
            merged["required"] = required
    return merged


def merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in schemas:
        if not schema:
            continue
        identity = schema_identity(schema)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(schema)

    if not unique:
        return {}
    if len(unique) == 1:
        return unique[0]

    schema_types = {schema.get("type") for schema in unique if isinstance(schema.get("type"), str)}
    if schema_types == {"object"}:
        return merge_object_schemas(unique)
    if schema_types == {"array"}:
        item_schemas = [schema.get("items") for schema in unique if isinstance(schema.get("items"), dict)]
        merged: dict[str, Any] = {"type": "array"}
        if item_schemas:
            merged["items"] = merge_schemas(item_schemas)
        return merged
    return {"anyOf": unique}


def infer_schema_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schemas = [infer_schema_from_value(item) for item in value]
        schema: dict[str, Any] = {"type": "array"}
        if item_schemas:
            schema["items"] = merge_schemas(item_schemas)
        return schema
    if isinstance(value, dict):
        return infer_tool_parameters_schema([value])
    return {}


def infer_tool_parameters_schema(argument_samples: list[Any]) -> dict[str, Any]:
    dict_samples = [sample for sample in argument_samples if isinstance(sample, dict)]
    if not dict_samples:
        return empty_object_schema()

    properties: dict[str, dict[str, Any]] = {}
    all_keys = sorted({key for sample in dict_samples for key in sample})
    for key in all_keys:
        observed = [infer_schema_from_value(sample[key]) for sample in dict_samples if key in sample]
        properties[key] = merge_schemas(observed)

    required = sorted(set.intersection(*(set(sample.keys()) for sample in dict_samples))) if dict_samples else []
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return schema
