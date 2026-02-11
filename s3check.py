import json
import logging
import time
import uuid

import boto3
import streamlit as st
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# DEFAULT CONFIG
# -------------------------------------------------
DEFAULT_PROFILE = "qms-assumed-role"
DEFAULT_REGION = "ap-south-1"
DEFAULT_BUCKET = "edme-apps-data-dev"
DEFAULT_PREFIX = "policygpt/"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
CHAT_MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"
MAX_CITATIONS_PER_FIELD = 3
MAX_PROMPT_CHARS = 100000
MAX_EVIDENCE_CHARS = 20000
MAX_EVIDENCE_SNIPPET = 400
MAX_FIELD_EVIDENCE_ITEMS = 8

# -------------------------------------------------
# AWS CLIENTS
# -------------------------------------------------
@st.cache_resource
def get_clients(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3"), session.client("textract")

@st.cache_resource
def get_embeddings(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    bedrock_runtime = session.client("bedrock-runtime")
    return BedrockEmbeddings(
        model_id=EMBED_MODEL_ID,
        client=bedrock_runtime,
        model_kwargs={
            "dimensions": EMBED_DIM,
            "normalize": True,
        },
    )

@st.cache_resource
def get_llm(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    bedrock_runtime = session.client("bedrock-runtime")
    return ChatBedrock(
        model_id=CHAT_MODEL_ID,
        client=bedrock_runtime,
        model_kwargs={"max_tokens": 4096},
    )

# -------------------------------------------------
# SCHEMA HELPERS
# -------------------------------------------------
STRUCTURAL_KEYS = {
    "type",
    "description",
    "required",
    "properties",
    "items",
    "enum",
    "oneOf",
    "anyOf",
    "allOf",
    "title",
    "default",
    "examples",
}

def load_uploaded_json(uploaded_file) -> dict:
    raw = uploaded_file.getvalue().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("NULL", "null").replace("Null", "null").replace("None", "null")
        return json.loads(cleaned)

def collect_leaf_fields(spec: dict, path: str = "") -> list[dict]:
    fields = []
    if not isinstance(spec, dict):
        return fields

    spec_type = spec.get("type")
    description = spec.get("description", "")

    if spec_type == "object":
        for key, prop in spec.get("properties", {}).items():
            next_path = f"{path}.{key}" if path else key
            fields.extend(collect_leaf_fields(prop, next_path))
        return fields

    if spec_type == "array":
        items = spec.get("items", {})
        if isinstance(items, dict) and items.get("type") == "object":
            for key, prop in items.get("properties", {}).items():
                next_path = f"{path}.{key}" if path else key
                fields.extend(collect_leaf_fields(prop, next_path))
            return fields
        fields.append(
            {
                "path": path,
                "description": description,
                "type": spec_type,
            }
        )
        return fields

    fields.append(
        {
            "path": path,
            "description": description,
            "type": spec_type,
        }
    )
    return fields

def build_output_skeleton_from_spec(spec: dict):
    if not isinstance(spec, dict):
        return {"value": None, "citations": []}

    spec_type = spec.get("type")
    extras = {k: v for k, v in spec.items() if k not in STRUCTURAL_KEYS}

    if spec_type == "object":
        output = {}
        for key, prop in spec.get("properties", {}).items():
            output[key] = build_output_skeleton_from_spec(prop)
        for key, val in extras.items():
            if key not in output:
                output[key] = val
        return output

    if spec_type == "array":
        items = spec.get("items", {})
        if isinstance(items, dict) and items.get("type") == "object":
            name_prop = items.get("properties", {}).get("name", {})
            enum = name_prop.get("enum") if isinstance(name_prop, dict) else None
            if enum:
                arr = []
                for name in enum:
                    item_obj = build_output_skeleton_from_spec(items)
                    if isinstance(item_obj, dict) and "name" in item_obj:
                        if isinstance(item_obj["name"], dict):
                            item_obj["name"]["value"] = name
                            item_obj["name"]["citations"] = []
                    # Provide defaults for coverage items
                    if isinstance(item_obj, dict):
                        is_cov = item_obj.get("is_covered")
                        if isinstance(is_cov, dict) and "value" in is_cov:
                            is_cov["value"] = False
                            is_cov["citations"] = []
                        terms = item_obj.get("terms_conditions")
                        if isinstance(terms, dict) and "value" in terms:
                            terms["value"] = ""
                            terms["citations"] = []
                    arr.append(item_obj)
                return arr
        return []

    leaf = {"value": None, "citations": []}
    for key, val in extras.items():
        if key not in leaf:
            leaf[key] = val
    return leaf

def is_schema_spec(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("type") in {"object", "array", "string", "number", "integer", "boolean"}:
        return True
    if "properties" in obj or "items" in obj:
        return True
    return False

def build_template_skeleton(obj):
    if isinstance(obj, dict):
        if "value" in obj and "citations" in obj:
            leaf = dict(obj)
            leaf["value"] = None
            leaf["citations"] = []
            return leaf
        return {k: build_template_skeleton(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            return [build_template_skeleton(item) for item in obj]
        return []
    return obj

def collect_template_paths(obj, path: str = "") -> list[str]:
    paths = []
    if isinstance(obj, dict):
        if "value" in obj and "citations" in obj:
            if path:
                paths.append(path)
            return paths
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            paths.extend(collect_template_paths(value, next_path))
        return paths
    if isinstance(obj, list):
        if path:
            paths.append(path)
        return paths
    return paths

def filter_template_by_paths(obj, paths: set[str], path: str = ""):
    if isinstance(obj, dict):
        if "value" in obj and "citations" in obj:
            return dict(obj) if path in paths else None
        filtered = {}
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            kept = filter_template_by_paths(value, paths, next_path)
            if kept is not None:
                filtered[key] = kept
        return filtered if filtered else None
    if isinstance(obj, list):
        return list(obj) if path in paths else None
    return None

def filter_spec_by_paths(spec: dict, paths: set[str], path: str = "") -> dict | None:
    if not isinstance(spec, dict):
        return None

    spec_type = spec.get("type")
    if spec_type == "object":
        props = spec.get("properties", {})
        kept_props = {}
        for key, prop in props.items():
            next_path = f"{path}.{key}" if path else key
            kept = filter_spec_by_paths(prop, paths, next_path)
            if kept is not None:
                kept_props[key] = kept
        if not kept_props:
            return None
        filtered = dict(spec)
        filtered["properties"] = kept_props
        if "required" in filtered:
            filtered["required"] = [k for k in filtered["required"] if k in kept_props]
        return filtered

    if spec_type == "array":
        target_prefix = path
        if any(p == target_prefix or p.startswith(f"{target_prefix}.") for p in paths):
            return dict(spec)
        return None

    # Leaf
    if path in paths:
        return dict(spec)
    return None

def normalize_output_to_spec(output_json, spec: dict):
    if not isinstance(spec, dict):
        return output_json

    spec_type = spec.get("type")
    if spec_type == "object":
        result = {}
        props = spec.get("properties", {})
        if not isinstance(output_json, dict):
            output_json = {}
        for key, prop in props.items():
            result[key] = normalize_output_to_spec(output_json.get(key), prop)
        return result

    if spec_type == "array":
        items = spec.get("items", {})
        items_type = items.get("type") if isinstance(items, dict) else None
        if isinstance(output_json, dict) and isinstance(output_json.get("value"), list):
            data_list = output_json.get("value", [])
        elif isinstance(output_json, list):
            data_list = output_json
        else:
            data_list = []
        if items_type == "object" and isinstance(items, dict):
            return [normalize_output_to_spec(item, items) for item in data_list]
        return data_list

    if isinstance(output_json, dict) and "value" in output_json and "citations" in output_json:
        return output_json
    return {"value": output_json, "citations": []}

def extract_json_object(text: str):
    if not text:
        return None
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(cleaned) if ch in "{["]
    for idx in reversed(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None

def merge_output_skeleton(skeleton, model_output):
    if model_output is None:
        return skeleton

    if isinstance(skeleton, dict) and isinstance(model_output, dict):
        if "value" in skeleton and "citations" in skeleton:
            merged = dict(skeleton)
            if "value" in model_output:
                merged["value"] = model_output.get("value")
            if "citations" in model_output:
                merged["citations"] = model_output.get("citations", [])
            return merged
        if set(model_output.keys()).issubset({"value", "citations"}):
            return skeleton
        merged = dict(skeleton)
        for key, value in model_output.items():
            if key in merged:
                merged[key] = merge_output_skeleton(merged[key], value)
        return merged

    if isinstance(skeleton, list) and isinstance(model_output, list):
        if not skeleton:
            return model_output
        def get_name(item):
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, dict):
                    return name.get("value")
                if isinstance(name, str):
                    return name
            return None

        if all(get_name(item) for item in skeleton):
            model_map = {
                get_name(item): item
                for item in model_output
                if get_name(item)
            }
            merged_list = []
            for item in skeleton:
                name = get_name(item)
                if name in model_map:
                    merged_list.append(merge_output_skeleton(item, model_map[name]))
                else:
                    merged_list.append(item)
            return merged_list

        merged_list = []
        for idx, item in enumerate(skeleton):
            if idx < len(model_output):
                merged_list.append(merge_output_skeleton(item, model_output[idx]))
            else:
                merged_list.append(item)
        return merged_list

    if isinstance(skeleton, dict) and "value" in skeleton and "citations" in skeleton:
        merged = dict(skeleton)
        merged["value"] = model_output
        return merged

    return model_output

def build_field_requirements_summary(spec: dict, max_fields: int = 120) -> str:
    fields = collect_leaf_fields(spec)
    if not fields:
        return "[]"
    lines = []
    for field in fields[:max_fields]:
        path = field.get("path", "")
        ftype = field.get("type", "")
        desc = (field.get("description") or "").strip()
        lines.append(f"- {path} | type={ftype} | description={desc}")
    if len(fields) > max_fields:
        lines.append(f"- ... truncated ({len(fields) - max_fields} more fields)")
    return "\n".join(lines)

def truncate_text(text: str, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def build_evidence_context(
    spec: dict,
    vectorstore,
    items: list[dict],
    k: int = 2,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    payload = {"fields": []}
    for field in collect_leaf_fields(spec):
        query = field["path"]
        if field.get("description"):
            query = f"{query} {field['description']}"
        docs = []
        if vectorstore is not None:
            try:
                docs = vectorstore.similarity_search(query, k=k)
            except Exception:
                docs = []
        evidence_items = []
        for doc in docs:
            item_idx = doc.metadata.get("item_index")
            citations = []
            if isinstance(item_idx, int) and 0 <= item_idx < len(items):
                citations = items[item_idx].get("citations", [])
            evidence_items.append(
                {
                    "text": truncate_text(doc.page_content, MAX_EVIDENCE_SNIPPET),
                    "citations": citations,
                }
            )
        entry = (
            {
                "path": field["path"],
                "type": field.get("type"),
                "description": field.get("description", ""),
                "evidence": evidence_items,
            }
        )
        # Enforce max_chars to avoid model input overflow.
        tentative = {"fields": payload["fields"] + [entry]}
        if len(json.dumps(tentative, ensure_ascii=False)) > max_chars:
            break
        payload["fields"].append(entry)
    return json.dumps(payload, ensure_ascii=False)

def build_evidence_context_for_paths(
    paths: list[str],
    vectorstore,
    items: list[dict],
    k: int = 2,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    payload = {"fields": []}
    for path in paths:
        docs = []
        if vectorstore is not None:
            try:
                docs = vectorstore.similarity_search(path, k=k)
            except Exception:
                docs = []
        evidence_items = []
        for doc in docs:
            item_idx = doc.metadata.get("item_index")
            citations = []
            if isinstance(item_idx, int) and 0 <= item_idx < len(items):
                citations = items[item_idx].get("citations", [])
            evidence_items.append(
                {
                    "text": truncate_text(doc.page_content, MAX_EVIDENCE_SNIPPET),
                    "citations": citations,
                }
            )
        entry = {"path": path, "evidence": evidence_items}
        tentative = {"fields": payload["fields"] + [entry]}
        if len(json.dumps(tentative, ensure_ascii=False)) > max_chars:
            break
        payload["fields"].append(entry)
    return json.dumps(payload, ensure_ascii=False)

def build_system_prompt(spec: dict, evidence_context: str) -> str:
    field_requirements = build_field_requirements_summary(spec)
    return (
        "You are extracting fields from the provided PDF evidence.\n"
        "Follow the input JSON schema exactly.\n"
        "Return a JSON object where each required leaf field maps to an object: "
        "{\"value\": <typed value>, \"citations\": []}.\n"
        "Rules:\n"
        "- Read each field description and type before extracting.\n"
        "- Type correctness is mandatory (date YYYY-MM-DD, number numeric, boolean true/false).\n"
        "- Do not guess; only use the evidence context.\n"
        "- If a scalar field is not found, set value=null and citations=[].\n"
        "- If an array field is not found, set value=[] and citations=[].\n"
        "- JSON only; no explanations.\n"
        "\nSchema:\n"
        f"{json.dumps(spec, ensure_ascii=False)}\n"
        "\nField Requirements:\n"
        f"{field_requirements}\n"
        "\nEvidence Context:\n"
        f"{evidence_context}\n"
        "\nReturn JSON only."
    )

def build_template_prompt(template: dict, evidence_context: str, paths: list[str]) -> str:
    return (
        "You are extracting fields from the provided PDF evidence.\n"
        "Follow the template JSON structure exactly.\n"
        "Fill only the `value` fields; keep `citations` as empty arrays.\n"
        "If a value is not found, set it to null.\n"
        "For list fields, return [] if no evidence is found.\n"
        "JSON only; no explanations.\n"
        "\nTemplate JSON:\n"
        f"{json.dumps(template, ensure_ascii=False)}\n"
        "\nField Paths:\n"
        f"{json.dumps(paths, ensure_ascii=False)}\n"
        "\nEvidence Context:\n"
        f"{evidence_context}\n"
        "\nReturn JSON only."
    )

def build_field_queries(path: str, description: str | None = None) -> list[str]:
    label, context = path_label_segments(path)
    queries = []
    if label:
        queries.append(label)
    if context:
        queries.append(" ".join([*context, label]).strip())
    if description:
        queries.append(f"{label} {description}".strip())
        if context:
            queries.append(f"{' '.join(context)} {description}".strip())
    for year in extract_year_context(path):
        if label:
            queries.append(f"{label} {year}".strip())
    for alias in field_aliases(label):
        queries.append(alias)
        if context:
            queries.append(f"{' '.join(context)} {alias}".strip())
    # unique preserve order
    seen = set()
    unique = []
    for q in queries:
        qn = q.strip()
        if not qn or qn in seen:
            continue
        seen.add(qn)
        unique.append(qn)
    return unique

def find_lexical_items(items: list[dict], label: str, max_hits: int = 3) -> list[int]:
    hits = []
    norm_label = normalize_label_text(label)
    if not norm_label:
        return hits
    tokens = [t for t in norm_label.split() if len(t) >= 3]
    for idx, item in enumerate(items):
        text = normalize_label_text(item.get("text", ""))
        if not text:
            continue
        if all(t in text for t in tokens):
            hits.append(idx)
            if len(hits) >= max_hits:
                break
    return hits

def build_evidence_items_for_field(
    path: str,
    description: str | None,
    vectorstore,
    items: list[dict],
    k: int = 3,
) -> list[dict]:
    label, context = path_label_segments(path)
    queries = build_field_queries(path, description)
    evidence = []
    seen_items = set()

    if vectorstore is not None:
        for q in queries[:5]:
            try:
                docs = vectorstore.similarity_search(q, k=k)
            except Exception:
                docs = []
            for doc in docs:
                item_idx = doc.metadata.get("item_index")
                if isinstance(item_idx, int) and 0 <= item_idx < len(items):
                    if item_idx in seen_items:
                        continue
                    seen_items.add(item_idx)
                    citations = items[item_idx].get("citations", [])[:MAX_CITATIONS_PER_FIELD]
                    evidence.append(
                        {
                            "id": f"i{item_idx}",
                            "text": truncate_text(items[item_idx].get("text", ""), MAX_EVIDENCE_SNIPPET),
                            "citations": citations,
                        }
                    )
                    # add neighbors for table context
                    for neighbor in (item_idx - 1, item_idx + 1):
                        if 0 <= neighbor < len(items) and neighbor not in seen_items:
                            seen_items.add(neighbor)
                            citations_n = items[neighbor].get("citations", [])[:MAX_CITATIONS_PER_FIELD]
                            evidence.append(
                                {
                                    "id": f"i{neighbor}",
                                    "text": truncate_text(
                                        items[neighbor].get("text", ""),
                                        MAX_EVIDENCE_SNIPPET,
                                    ),
                                    "citations": citations_n,
                                }
                            )
                if len(evidence) >= MAX_FIELD_EVIDENCE_ITEMS:
                    break
            if len(evidence) >= MAX_FIELD_EVIDENCE_ITEMS:
                break

    lexical_terms = [label] + context + field_aliases(label)
    for term in lexical_terms:
        if not term:
            continue
        for item_idx in find_lexical_items(items, term, max_hits=3):
            if item_idx in seen_items:
                continue
            seen_items.add(item_idx)
            citations = items[item_idx].get("citations", [])[:MAX_CITATIONS_PER_FIELD]
            evidence.append(
                {
                    "id": f"i{item_idx}",
                    "text": truncate_text(items[item_idx].get("text", ""), MAX_EVIDENCE_SNIPPET),
                    "citations": citations,
                }
            )
            if len(evidence) >= MAX_FIELD_EVIDENCE_ITEMS:
                break
        if len(evidence) >= MAX_FIELD_EVIDENCE_ITEMS:
            break

    return evidence[:MAX_FIELD_EVIDENCE_ITEMS]

def build_field_prompt(
    path: str,
    field_type: str | None,
    description: str | None,
    evidence_items: list[dict],
) -> str:
    label, context = path_label_segments(path)
    years = extract_year_context(path)
    return (
        "You extract a single field from evidence.\n"
        "Return JSON: {\"value\": <typed or null>, \"evidence_ids\": []}.\n"
        "Use only the evidence; if not found, value=null and evidence_ids=[].\n"
        "If the evidence uses a close synonym (e.g., outstanding = unpaid), you may use it.\n"
        "Choose evidence_ids from the provided evidence list.\n"
        f"Field path: {path}\n"
        f"Field label: {label}\n"
        f"Field context: {' | '.join(context) if context else 'none'}\n"
        f"Field year context: {', '.join(years) if years else 'none'}\n"
        f"Field type: {field_type or 'unknown'}\n"
        f"Field description: {description or ''}\n"
        f"Evidence: {json.dumps(evidence_items, ensure_ascii=False)}\n"
        "Return JSON only."
    )

def build_batch_prompt(fields: list[dict]) -> str:
    return (
        "You are extracting multiple fields from evidence.\n"
        "Return a JSON object mapping each field path to {\"value\": <typed or null>, \"evidence_ids\": []}.\n"
        "Use only evidence for each field; if not found, set value=null and evidence_ids=[].\n"
        "If evidence uses a close synonym, you may use it.\n"
        f"Fields: {json.dumps(fields, ensure_ascii=False)}\n"
        "Return JSON only."
    )

def map_evidence_ids_to_citations(evidence_items: list[dict], evidence_ids: list) -> list[dict]:
    id_set = set()
    for eid in evidence_ids or []:
        if isinstance(eid, str):
            id_set.add(eid)
    citations = []
    for item in evidence_items:
        if item.get("id") in id_set:
            for c in item.get("citations", []):
                citations.append(c)
                if len(citations) >= MAX_CITATIONS_PER_FIELD:
                    return citations
    return citations

def set_value_by_path(output: dict, path: str, value, citations: list[dict]):
    if not path:
        return
    parts = path.split(".")
    node = output
    for idx, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            return
        if idx == len(parts) - 1:
            target = node[part]
            if isinstance(target, dict) and "value" in target and "citations" in target:
                target["value"] = value
                target["citations"] = citations
            return
        node = node[part]

def collect_empty_leaf_paths(obj, path: str = "") -> list[str]:
    paths = []
    if isinstance(obj, dict):
        if "value" in obj and "citations" in obj:
            value = obj.get("value")
            if value is None or value == [] or value == "":
                if path:
                    paths.append(path)
            return paths
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            paths.extend(collect_empty_leaf_paths(value, next_path))
        return paths
    if isinstance(obj, list):
        for item in obj:
            paths.extend(collect_empty_leaf_paths(item, path))
        return paths
    return paths

# -------------------------------------------------
# CITATION MATCHING
# -------------------------------------------------
def normalize_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip().lower()

def normalize_digits(value: str) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")

def normalize_label_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())

def extract_year_context(path: str) -> list[str]:
    if not path:
        return []
    parts = path.split(".")
    years = []
    for part in parts:
        if part.isdigit() and len(part) == 6 and part.startswith("20"):
            y1 = part[:4]
            y2 = part[4:]
            years.extend(
                [
                    f"{y1}-{y2}",
                    f"{y1}/{y2}",
                    f"{y1} {y2}",
                    part,
                ]
            )
    return years

def path_label_segments(path: str) -> tuple[str, list[str]]:
    if not path:
        return "", []
    parts = path.split(".")
    label = parts[-1].replace("_", " ")
    context = [p.replace("_", " ") for p in parts[:-1]]
    return label, context

def field_aliases(label: str) -> list[str]:
    aliases_map = {
        "Outstanding Amount": [
            "outstanding",
            "outstanding amount",
            "outstanding claim",
            "os amount",
        ],
        "Paid Claim Amount": [
            "paid claim",
            "claims paid",
            "paid amount",
        ],
        "Premium Paid": [
            "premium paid",
            "premium received",
        ],
        "Total Claim Amount": [
            "total claim",
            "total claims",
        ],
        "Total Number of Claims": [
            "number of claims",
            "no of claims",
            "claims count",
        ],
        "Claim Ratio": [
            "claim ratio",
            "loss ratio",
            "claims ratio",
        ],
        "Deductible Amount": [
            "deductible",
            "excess",
        ],
        "Deductible Basis": [
            "deductible basis",
            "basis",
            "excess basis",
        ],
        "Total Premium": [
            "total premium",
            "gross premium",
        ],
        "Net Premium": [
            "net premium",
            "premium net",
        ],
        "GST 18": [
            "gst",
            "gst 18",
            "18% gst",
            "tax",
        ],
        "Capacity": [
            "capacity",
            "share",
            "participation",
        ],
        "Brokerage": [
            "brokerage",
            "commission",
        ],
        "Terrorism Premium": [
            "terrorism premium",
            "terror premium",
        ],
        "Total Premium incl GST": [
            "total premium incl gst",
            "premium including gst",
            "total incl gst",
        ],
    }
    return aliases_map.get(label, [])

def find_citations_for_text(components: list[dict], text: str) -> list[dict]:
    if text is None:
        return []

    if isinstance(text, bool):
        target = "true" if text else "false"
        matches = []
        for comp in components:
            line_text = comp.get("text", "")
            if not line_text:
                continue
            if target in normalize_text(line_text):
                matches.append(comp)
        return [
            component_to_citation(c, "contains")
            for c in matches[:MAX_CITATIONS_PER_FIELD]
        ]

    if isinstance(text, (int, float)):
        target = normalize_digits(text)
        if not target:
            return []
        matches = []
        for comp in components:
            line_text = comp.get("text", "")
            if not line_text:
                continue
            if target in normalize_digits(line_text):
                matches.append(comp)
        return [
            component_to_citation(c, "contains")
            for c in matches[:MAX_CITATIONS_PER_FIELD]
        ]

    value = str(text).strip()
    if not value:
        return []

    norm_value = normalize_text(value)
    exact_matches = []
    contains_matches = []
    line_in_value_matches = []

    for comp in components:
        line_text = comp.get("text", "")
        if not line_text:
            continue
        norm_line = normalize_text(line_text)
        if not norm_line:
            continue
        if norm_line == norm_value:
            exact_matches.append(comp)
        elif norm_value in norm_line:
            contains_matches.append(comp)
        elif norm_line in norm_value:
            line_in_value_matches.append(comp)

    if exact_matches:
        return [
            component_to_citation(c, "exact")
            for c in exact_matches[:MAX_CITATIONS_PER_FIELD]
        ]
    if contains_matches:
        return [
            component_to_citation(c, "contains")
            for c in contains_matches[:MAX_CITATIONS_PER_FIELD]
        ]
    if line_in_value_matches:
        return [
            component_to_citation(c, "contained")
            for c in line_in_value_matches[:MAX_CITATIONS_PER_FIELD]
        ]

    return []

def backfill_value_citations(obj, components: list[dict]):
    if isinstance(obj, dict):
        if "value" in obj and "citations" in obj:
            value = obj.get("value")
            if value is None or value == "" or value == []:
                obj["citations"] = []
                return obj
            if not obj.get("citations"):
                obj["citations"] = find_citations_for_text(components, value)
            return obj
        for key, value in obj.items():
            obj[key] = backfill_value_citations(value, components)
        return obj
    if isinstance(obj, list):
        return [backfill_value_citations(item, components) for item in obj]
    return obj
# -------------------------------------------------
# TEXTRACT HELPERS
# -------------------------------------------------
def upload_pdf_to_s3(s3, bucket: str, key: str, data: bytes) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/pdf",
    )

def start_textract_job(textract, bucket: str, key: str) -> str:
    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return response["JobId"]

def start_textract_analysis_job(
    textract,
    bucket: str,
    key: str,
    feature_types: list[str],
    queries: list[str] | None = None,
) -> str:
    params = {
        "DocumentLocation": {"S3Object": {"Bucket": bucket, "Name": key}},
        "FeatureTypes": feature_types,
    }
    if queries:
        params["QueriesConfig"] = {
            "Queries": [
                {"Text": q, "Alias": f"q{idx+1}"}
                for idx, q in enumerate(queries)
                if q.strip()
            ]
        }
    response = textract.start_document_analysis(**params)
    return response["JobId"]

def wait_for_textract(textract, job_id: str, max_wait: int, poll: int) -> str:
    waited = 0
    while waited < max_wait:
        response = textract.get_document_text_detection(JobId=job_id, MaxResults=1)
        status = response["JobStatus"]
        if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            return status
        time.sleep(poll)
        waited += poll
    raise TimeoutError("Textract job timed out")

def wait_for_textract_analysis(textract, job_id: str, max_wait: int, poll: int) -> str:
    waited = 0
    while waited < max_wait:
        response = textract.get_document_analysis(JobId=job_id, MaxResults=1)
        status = response["JobStatus"]
        if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            return status
        time.sleep(poll)
        waited += poll
    raise TimeoutError("Textract analysis job timed out")

def collect_textract_lines(textract, job_id: str) -> list[dict]:
    blocks = []
    next_token = None

    while True:
        args = {"JobId": job_id}
        if next_token:
            args["NextToken"] = next_token
        response = textract.get_document_text_detection(**args)
        blocks.extend(response.get("Blocks", []))
        next_token = response.get("NextToken")
        if not next_token:
            break

    lines = []
    for b in blocks:
        if b.get("BlockType") != "LINE" or "Text" not in b:
            continue
        bbox = b.get("Geometry", {}).get("BoundingBox", {})
        lines.append(
            {
                "text": b["Text"],
                "page": b.get("Page"),
                "block_id": b.get("Id"),
                "confidence": b.get("Confidence"),
                "bounding_box": {
                    "left": bbox.get("Left", 0.0),
                    "top": bbox.get("Top", 0.0),
                    "width": bbox.get("Width", 0.0),
                    "height": bbox.get("Height", 0.0),
                },
            }
        )
    return lines

def collect_textract_analysis_blocks(textract, job_id: str) -> list[dict]:
    blocks = []
    next_token = None
    while True:
        args = {"JobId": job_id}
        if next_token:
            args["NextToken"] = next_token
        response = textract.get_document_analysis(**args)
        blocks.extend(response.get("Blocks", []))
        next_token = response.get("NextToken")
        if not next_token:
            break
    return blocks

def build_lines_from_blocks(blocks: list[dict]) -> list[dict]:
    lines = []
    for b in blocks:
        if b.get("BlockType") != "LINE" or "Text" not in b:
            continue
        bbox = b.get("Geometry", {}).get("BoundingBox", {})
        lines.append(
            {
                "text": b["Text"],
                "page": b.get("Page"),
                "block_id": b.get("Id"),
                "confidence": b.get("Confidence"),
                "bounding_box": {
                    "left": bbox.get("Left", 0.0),
                    "top": bbox.get("Top", 0.0),
                    "width": bbox.get("Width", 0.0),
                    "height": bbox.get("Height", 0.0),
                },
            }
        )
    return lines

def build_items_from_tables(tables: list[dict]) -> list[dict]:
    items = []
    for idx, table in enumerate(tables):
        rows = table.get("rows") or []
        lines = [" | ".join([str(cell) for cell in row]) for row in rows if row]
        text = "\n".join(lines).strip()
        if not text:
            continue
        pseudo = {
            "text": text,
            "page": table.get("page"),
            "confidence": table.get("confidence"),
            "bounding_box": table.get("bbox") or {},
        }
        items.append(
            {
                "type": "table",
                "text": text,
                "page": table.get("page"),
                "citations": [component_to_citation(pseudo, "exact")],
            }
        )
    return items

def build_items_from_forms(forms: list[dict]) -> list[dict]:
    items = []
    for kv in forms:
        key = kv.get("key", "")
        value = kv.get("value", "")
        text = f"{key}: {value}".strip(": ").strip()
        if not text:
            continue
        pseudo = {
            "text": text,
            "page": kv.get("page"),
            "confidence": kv.get("confidence"),
            "bounding_box": kv.get("bbox") or {},
        }
        items.append(
            {
                "type": "form",
                "text": text,
                "page": kv.get("page"),
                "citations": [component_to_citation(pseudo, "exact")],
            }
        )
    return items

def build_merged_items(
    paragraphs: list[dict],
    tables: list[dict],
    forms: list[dict],
) -> list[dict]:
    merged = []
    for p in paragraphs:
        merged.append(
            {
                "type": "paragraph",
                "text": p.get("text", ""),
                "page": p.get("page"),
                "citations": p.get("citations", []),
            }
        )
    merged.extend(build_items_from_tables(tables))
    merged.extend(build_items_from_forms(forms))
    return merged
def build_block_map(blocks: list[dict]) -> dict[str, dict]:
    return {b.get("Id"): b for b in blocks if b.get("Id")}

def text_from_block(block: dict, block_map: dict[str, dict]) -> str:
    if not block:
        return ""
    if "Text" in block and block["Text"]:
        return block["Text"]
    parts = []
    for rel in block.get("Relationships", []):
        if rel.get("Type") != "CHILD":
            continue
        for cid in rel.get("Ids", []):
            child = block_map.get(cid)
            if not child:
                continue
            if child.get("BlockType") == "WORD":
                parts.append(child.get("Text", ""))
            elif child.get("BlockType") == "SELECTION_ELEMENT":
                if child.get("SelectionStatus") == "SELECTED":
                    parts.append("X")
    return " ".join([p for p in parts if p])

def extract_tables(blocks: list[dict]) -> list[dict]:
    block_map = build_block_map(blocks)
    tables = []
    for table in blocks:
        if table.get("BlockType") != "TABLE":
            continue
        cells = []
        for rel in table.get("Relationships", []):
            if rel.get("Type") != "CHILD":
                continue
            for cid in rel.get("Ids", []):
                cell = block_map.get(cid)
                if not cell or cell.get("BlockType") != "CELL":
                    continue
                text = text_from_block(cell, block_map)
                cells.append(
                    {
                        "row": cell.get("RowIndex"),
                        "col": cell.get("ColumnIndex"),
                        "text": text,
                    }
                )
        max_row = max([c["row"] for c in cells], default=0)
        max_col = max([c["col"] for c in cells], default=0)
        rows = []
        for r in range(1, max_row + 1):
            row = []
            for c in range(1, max_col + 1):
                match = next((x for x in cells if x["row"] == r and x["col"] == c), None)
                row.append(match["text"] if match else "")
            rows.append(row)
        tables.append(
            {
                "page": table.get("Page"),
                "rows": rows,
                "confidence": table.get("Confidence"),
                "bbox": table.get("Geometry", {}).get("BoundingBox", {}),
            }
        )
    return tables

def extract_forms(blocks: list[dict]) -> list[dict]:
    block_map = build_block_map(blocks)
    key_blocks = [
        b for b in blocks
        if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in (b.get("EntityTypes") or [])
    ]
    kvs = []
    for key in key_blocks:
        key_text = text_from_block(key, block_map)
        value_text = ""
        for rel in key.get("Relationships", []):
            if rel.get("Type") != "VALUE":
                continue
            for vid in rel.get("Ids", []):
                val_block = block_map.get(vid)
                value_text = text_from_block(val_block, block_map)
        if key_text or value_text:
            kvs.append(
                {
                    "page": key.get("Page"),
                    "key": key_text,
                    "value": value_text,
                    "confidence": key.get("Confidence"),
                    "bbox": key.get("Geometry", {}).get("BoundingBox", {}),
                }
            )
    return kvs

def extract_layout(blocks: list[dict]) -> list[dict]:
    block_map = build_block_map(blocks)
    layouts = []
    for b in blocks:
        bt = b.get("BlockType") or ""
        if not bt.startswith("LAYOUT"):
            continue
        layouts.append(
            {
                "page": b.get("Page"),
                "type": bt,
                "text": text_from_block(b, block_map),
                "confidence": b.get("Confidence"),
                "bbox": b.get("Geometry", {}).get("BoundingBox", {}),
            }
        )
    return layouts

# -------------------------------------------------
# CITATION HELPERS
# -------------------------------------------------
def confidence_label(score):
    if score is None:
        return "unknown"
    try:
        val = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if val <= 1.0:
        val = val * 100.0
    if val >= 90:
        return "high"
    if val >= 75:
        return "medium"
    return "low"

def normalize_confidence(score):
    if score is None:
        return None
    try:
        val = float(score)
    except (TypeError, ValueError):
        return None
    if val > 1.0:
        return val / 100.0
    return val

def match_weight(match_type: str) -> float:
    if match_type == "exact":
        return 1.0
    if match_type == "contains":
        return 0.85
    if match_type == "contained":
        return 0.75
    return 0.6

def component_to_citation(component: dict, match_type: str = "exact") -> dict:
    bbox = component.get("bounding_box", {})
    page = component.get("page")
    conf = component.get("confidence")
    norm_conf = normalize_confidence(conf)
    weight = match_weight(match_type)
    combined = None if norm_conf is None else max(0.0, min(1.0, norm_conf * weight))
    return {
        "bbox": {
            "top": bbox.get("top"),
            "left": bbox.get("left"),
            "page": page,
            "width": bbox.get("width"),
            "height": bbox.get("height"),
            "original_page": page,
        },
        "type": "Text",
        "extra": None,
        "content": component.get("text", ""),
        "confidence": confidence_label(combined),
        "granular_confidence": {
            "parse_confidence": combined,
            "extract_confidence": None,
        },
    }

def build_page_citations(lines: list[dict]) -> list[dict]:
    pages: dict[int, list[dict]] = {}
    for line in lines:
        page = line.get("page") or 0
        pages.setdefault(page, []).append(line)

    output = []
    for page in sorted(pages.keys()):
        page_lines = pages[page]
        page_text = "\n".join([l.get("text", "") for l in page_lines if l.get("text")])
        page_citations = [component_to_citation(l, "exact") for l in page_lines]
        output.append(
            {
                "page": page,
                "text": page_text,
                "citations": page_citations,
            }
        )
    return output

def build_paragraphs(lines: list[dict], gap_threshold: float = 0.02) -> list[dict]:
    sorted_lines = sorted(
        lines,
        key=lambda l: (
            l.get("page") or 0,
            l.get("bounding_box", {}).get("top", 0.0),
            l.get("bounding_box", {}).get("left", 0.0),
        ),
    )

    paragraphs = []
    current = None
    prev_top = None
    prev_page = None

    for line in sorted_lines:
        text = line.get("text", "")
        if not text:
            continue
        page = line.get("page") or 0
        top = line.get("bounding_box", {}).get("top", 0.0)

        new_para = False
        if current is None:
            new_para = True
        elif page != prev_page:
            new_para = True
        elif prev_top is not None and abs(top - prev_top) > gap_threshold:
            new_para = True

        if new_para:
            if current:
                paragraphs.append(current)
            current = {
                "page": page,
                "text": text,
                "citations": [component_to_citation(line, "exact")],
            }
        else:
            current["text"] = f"{current['text']} {text}"
            current["citations"].append(component_to_citation(line, "exact"))

        prev_top = top
        prev_page = page

    if current:
        paragraphs.append(current)

    return paragraphs

def build_chunks_from_items(
    items: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = []
    metas = []
    for idx, item in enumerate(items):
        text = item.get("text", "")
        if not text:
            continue
        page = item.get("page")
        chunks = splitter.split_text(text)
        for c_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append(
                {
                    "page": page,
                    "item_index": idx,
                    "item_type": item.get("type", "paragraph"),
                    "chunk_index": c_idx,
                }
            )
    return texts, metas

def get_vectorstore():
    return st.session_state.get("vectorstore")

def set_vectorstore(vs):
    st.session_state.vectorstore = vs

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("S3 Textract Check")

# Hidden parameters (use defaults)
profile = DEFAULT_PROFILE
region = DEFAULT_REGION
bucket = DEFAULT_BUCKET
prefix = DEFAULT_PREFIX
poll_seconds = 5
max_wait_seconds = 600
feature_types = ["TABLES", "FORMS", "LAYOUT"]
gap_threshold = 0.02
chunk_size = 1000
chunk_overlap = 200

s3, textract = get_clients(profile, region)
llm = get_llm(profile, region)

pdf_ready = st.session_state.get("pdf_ready", False)
stored_paragraphs = st.session_state.get("paragraphs")
stored_lines = st.session_state.get("lines")

uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
process_btn = st.button("Upload to S3 and Extract", key="process_pdf")

if uploaded and process_btn:
    try:
        pdf_bytes = uploaded.read()
        safe_name = uploaded.name.replace(" ", "_")
        s3_key = f"{prefix}{uuid.uuid4().hex}_{safe_name}"

        with st.spinner("Uploading to S3..."):
            upload_pdf_to_s3(s3, bucket, s3_key, pdf_bytes)
        st.caption(f"S3 key: {s3_key}")

        use_analysis = bool(feature_types)

        with st.spinner("Starting Textract job..."):
            if use_analysis:
                job_id = start_textract_analysis_job(
                    textract,
                    bucket,
                    s3_key,
                    feature_types,
                )
            else:
                job_id = start_textract_job(textract, bucket, s3_key)

        with st.spinner("Waiting for Textract..."):
            if use_analysis:
                status = wait_for_textract_analysis(
                    textract,
                    job_id,
                    int(max_wait_seconds),
                    int(poll_seconds),
                )
            else:
                status = wait_for_textract(
                    textract,
                    job_id,
                    int(max_wait_seconds),
                    int(poll_seconds),
                )

        if status not in ("SUCCEEDED", "PARTIAL_SUCCESS"):
            st.error(f"Textract failed with status: {status}")
            st.stop()

        with st.spinner("Collecting extracted text..."):
            if use_analysis:
                blocks = collect_textract_analysis_blocks(textract, job_id)
                lines = build_lines_from_blocks(blocks)
            else:
                blocks = []
                lines = collect_textract_lines(textract, job_id)

        if not lines:
            st.warning("No text found in this PDF.")
            st.stop()

        paragraphs = build_paragraphs(lines, float(gap_threshold))
        tables = extract_tables(blocks) if use_analysis else []
        forms = extract_forms(blocks) if use_analysis else []
        merged_items = build_merged_items(paragraphs, tables, forms)

        embeddings = get_embeddings(profile, region)
        chunk_texts, chunk_metas = build_chunks_from_items(
            merged_items,
            int(chunk_size),
            int(chunk_overlap),
        )
        if chunk_texts:
            with st.spinner("Chunking + embedding with Bedrock Titan..."):
                # Reset vectorstore for a new PDF to avoid mixing documents.
                set_vectorstore(None)
                vectorstore = FAISS.from_texts(
                    texts=chunk_texts,
                    embedding=embeddings,
                    metadatas=chunk_metas,
                )
                set_vectorstore(vectorstore)

        st.session_state.lines = lines
        st.session_state.paragraphs = paragraphs
        st.session_state.pdf_ready = True
        st.session_state.s3_key = s3_key
        st.session_state.feature_types = feature_types
        st.session_state.tables = tables
        st.session_state.forms = forms
        st.session_state.layouts = extract_layout(blocks) if use_analysis else []
        st.session_state.evidence_items = merged_items
        st.session_state.pdf_ready_message = "PDF processed. You can upload requests.json now."
        st.rerun()
    except Exception as exc:
        logger.exception("S3/Textract flow failed")
        st.error(f"Failed: {exc}")

if st.session_state.get("pdf_ready_message"):
    st.success(st.session_state.pop("pdf_ready_message"))

if pdf_ready and stored_paragraphs:
    s3_key = st.session_state.get("s3_key")
    if s3_key:
        st.caption(f"S3 key: {s3_key}")
    ft = st.session_state.get("feature_types") or []
    if ft:
        st.caption(f"FeatureTypes: {', '.join(ft)}")

    st.subheader("Merged Paragraphs (Text + Citations)")
    tables = st.session_state.get("tables", [])
    forms = st.session_state.get("forms", [])
    merged_items = build_merged_items(stored_paragraphs, tables, forms)
    st.session_state.evidence_items = merged_items
    paragraphs_jsonl = "\n".join(
        json.dumps(p, ensure_ascii=False) for p in merged_items
    )
    st.text_area(
        "Paragraphs (JSONL)",
        paragraphs_jsonl,
        height=360,
    )

    vectorstore = get_vectorstore()
    if vectorstore is None:
        embeddings = get_embeddings(profile, region)
        chunk_texts, chunk_metas = build_chunks_from_items(
            merged_items,
            int(chunk_size),
            int(chunk_overlap),
        )
        if chunk_texts:
            vectorstore = FAISS.from_texts(
                texts=chunk_texts,
                embedding=embeddings,
                metadatas=chunk_metas,
            )
            set_vectorstore(vectorstore)

    st.divider()
    st.subheader("Map Schema with Opus 4.5")
    uploaded_spec = st.file_uploader(
        "Upload requests.json",
        type=["json"],
        key="spec_uploader",
    )
    generate_btn = st.button("Generate output.json", key="generate_output")

    if uploaded_spec and generate_btn:
        try:
            spec_data = load_uploaded_json(uploaded_spec)
        except Exception as exc:
            st.error(f"Invalid JSON file: {exc}")
            st.stop()

        is_schema = is_schema_spec(spec_data)
        if is_schema:
            skeleton = build_output_skeleton_from_spec(spec_data)
        else:
            skeleton = build_template_skeleton(spec_data)

        evidence_items = st.session_state.get("evidence_items") or merged_items

        def run_llm_for_spec(sub_spec: dict):
            max_evidence = MAX_EVIDENCE_CHARS
            system_prompt = ""
            evidence_context = ""
            for _ in range(3):
                if is_schema:
                    evidence_context = build_evidence_context(
                        sub_spec,
                        vectorstore,
                        evidence_items,
                        k=2,
                        max_chars=max_evidence,
                    )
                    system_prompt = build_system_prompt(sub_spec, evidence_context)
                else:
                    paths = collect_template_paths(sub_spec)
                    evidence_context = build_evidence_context_for_paths(
                        paths,
                        vectorstore,
                        evidence_items,
                        k=2,
                        max_chars=max_evidence,
                    )
                    system_prompt = build_template_prompt(sub_spec, evidence_context, paths)
                if len(system_prompt) <= MAX_PROMPT_CHARS or max_evidence <= 2000:
                    break
                max_evidence = int(max_evidence * 0.6)
            if len(system_prompt) > MAX_PROMPT_CHARS:
                raise ValueError("Prompt too long after trimming evidence.")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate the JSON output now."),
            ]
            response = llm.invoke(messages)
            content = getattr(response, "content", str(response))
            return extract_json_object(content)

        if is_schema:
            leaf_fields = collect_leaf_fields(spec_data)
            field_tasks = [
                (f.get("path"), f.get("description"), f.get("type"))
                for f in leaf_fields
                if f.get("path")
            ]
        else:
            paths = collect_template_paths(spec_data)
            field_tasks = [(p, None, None) for p in paths]

        merged = dict(skeleton)
        total = len(field_tasks)
        with st.spinner(f"Running Opus 4.5 field extraction for {total} fields..."):
            batch_size = 12
            queue = [field_tasks[i:i + batch_size] for i in range(0, len(field_tasks), batch_size)]
            while queue:
                batch = queue.pop(0)
                fields_payload = []
                field_evidence_map = {}
                for path, desc, ftype in batch:
                    ev_items = build_evidence_items_for_field(
                        path,
                        desc,
                        vectorstore,
                        evidence_items,
                        k=20,
                    )
                    if not ev_items:
                        continue
                    field_evidence_map[path] = ev_items
                    fields_payload.append(
                        {
                            "path": path,
                            "type": ftype,
                            "description": desc or "",
                            "evidence": ev_items,
                        }
                    )
                if not fields_payload:
                    continue
                prompt = build_batch_prompt(fields_payload)
                if len(prompt) > MAX_PROMPT_CHARS and len(batch) > 1:
                    mid = len(batch) // 2
                    queue.insert(0, batch[mid:])
                    queue.insert(0, batch[:mid])
                    continue
                messages = [
                    SystemMessage(content="You are a precise information extraction system."),
                    HumanMessage(content=prompt),
                ]
                try:
                    response = llm.invoke(messages)
                except Exception:
                    logger.exception("LLM batch call failed")
                    continue
                content = getattr(response, "content", str(response))
                result = extract_json_object(content)
                if not isinstance(result, dict):
                    continue
                for path, payload in result.items():
                    if not isinstance(payload, dict):
                        continue
                    value = payload.get("value")
                    evidence_ids = payload.get("evidence_ids") or []
                    ev_items = field_evidence_map.get(path, [])
                    citations = map_evidence_ids_to_citations(ev_items, evidence_ids)
                    if not citations:
                        citations = find_citations_for_text(stored_lines or [], value)
                    set_value_by_path(merged, path, value, citations)

            # Second pass: try missing fields individually with expanded evidence
            missing_paths = collect_empty_leaf_paths(merged)
            if missing_paths:
                for path, desc, ftype in field_tasks:
                    if path not in missing_paths:
                        continue
                    ev_items = build_evidence_items_for_field(
                        path,
                        desc,
                        vectorstore,
                        evidence_items,
                        k=40,
                    )
                    if not ev_items:
                        continue
                    prompt = build_field_prompt(path, ftype, desc, ev_items)
                    messages = [
                        SystemMessage(content="You are a precise information extraction system."),
                        HumanMessage(content=prompt),
                    ]
                    try:
                        response = llm.invoke(messages)
                    except Exception:
                        logger.exception("LLM call failed for field %s", path)
                        continue
                    content = getattr(response, "content", str(response))
                    result = extract_json_object(content)
                    if not isinstance(result, dict):
                        continue
                    value = result.get("value")
                    evidence_ids = result.get("evidence_ids") or []
                    citations = map_evidence_ids_to_citations(ev_items, evidence_ids)
                    if not citations:
                        citations = find_citations_for_text(stored_lines or [], value)
                    set_value_by_path(merged, path, value, citations)

        if is_schema:
            merged = normalize_output_to_spec(merged, spec_data)
        merged = backfill_value_citations(merged, stored_lines or [])
        st.subheader("Mapped Output JSON")
        st.json(merged)
