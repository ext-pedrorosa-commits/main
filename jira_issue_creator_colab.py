"""Script para Google Colab que cria issues no Jira a partir de CSV.

Requisitos:
- pandas
- requests (alternativa valida ao atlassian-python-api)

Arquivos esperados:
- /content/dados.csv
- /content/mapeamento.csv
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests

try:
    from google.colab import userdata
except ImportError:  # fora do Colab
    userdata = None


@dataclass
class JiraConfig:
    jira_url: str
    jira_email: str
    jira_api_token: str
    default_project_key: str
    default_issue_type: str = "Task"
    dry_run: bool = False


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        token = value.strip().lower()
        return token in {"", "nan", "none", "null"}
    return False


def _normalize_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.strip().lower().replace("-", "_").replace(" ", "_")


def _is_description_field(jira_field_id: str) -> bool:
    return _normalize_token(jira_field_id) == "description"


def _is_adf_document(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "doc"
        and isinstance(value.get("content"), list)
    )


def _text_to_adf(text: str) -> Dict[str, Any]:
    """Converte texto simples para ADF (Atlassian Document Format)."""
    lines = text.splitlines()
    paragraph_content: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines):
        if idx > 0:
            paragraph_content.append({"type": "hardBreak"})
        if line:
            paragraph_content.append({"type": "text", "text": line})

    paragraph: Dict[str, Any] = {"type": "paragraph"}
    if paragraph_content:
        paragraph["content"] = paragraph_content

    return {"type": "doc", "version": 1, "content": [paragraph]}


def _looks_like_adf_error(error_message: str) -> bool:
    token = (error_message or "").strip().lower()
    indicators = (
        "atlassian document",
        "adf",
        "type: doc",
        "type doc",
        "must be an object",
        "document format",
        "rich text",
    )
    return any(indicator in token for indicator in indicators)


def normalize_description_for_jira(value: Any) -> Any:
    """Normaliza description para o formato aceito pela API v3."""
    if value is None:
        return None
    if _is_adf_document(value):
        return value
    if isinstance(value, str):
        if _is_empty(value):
            return None
        return _text_to_adf(value.strip())
    return value


def _split_multi_value(raw_value: str) -> List[str]:
    if ";" in raw_value:
        parts = raw_value.split(";")
    else:
        parts = raw_value.split(",")
    return [part.strip() for part in parts if part and part.strip()]


def _try_parse_json(raw_value: str) -> Any:
    token = raw_value.strip()
    if not token.startswith("{") and not token.startswith("["):
        return raw_value
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        return raw_value


def _to_jira_date(raw_value: str) -> str:
    """Converte datas comuns para formato Jira datepicker: yyyy-mm-dd."""
    value = raw_value.strip()
    if _is_empty(value):
        return value

    # Formato ja esperado pelo Jira.
    for date_fmt in ("%Y-%m-%d",):
        try:
            return datetime.strptime(value, date_fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Formatos de entrada comuns em CSV.
    for date_fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, date_fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    raise ValueError(
        f"Data invalida '{raw_value}'. Use formatos como dd/mm/yyyy ou yyyy-mm-dd."
    )


def resolve_field_type(field_type: str) -> str:
    token = _normalize_token(field_type)
    aliases = {
        "text": {"texto", "text", "string", "plain_text"},
        "richtext": {
            "richtext",
            "rich_text",
            "textarea",
            "texto_rico",
            "descricao_rica",
            "adf",
        },
        "number": {"numero", "number", "int", "integer", "float", "double"},
        "user": {"usuario", "user", "accountid"},
        "user_list": {"usuarios", "multiuser", "user_list", "lista_usuarios"},
        "select": {"select", "single_select", "lista_selecao", "option"},
        "multiselect": {"multiselect", "multi_select", "lista", "checkbox"},
        "labels": {"labels", "rotulos", "tags"},
        "date": {"date", "data", "datepicker"},
        "datetime": {"datetime", "datahora", "date_time"},
        "project": {"project", "projeto"},
        "issuetype": {"issuetype", "tipo_issue", "tipo_da_issue"},
        "json": {"json", "object", "objeto", "dict"},
    }
    for canonical, values in aliases.items():
        if token in values:
            return canonical
    return "text"


def format_jira_field_value(raw_value: Any, field_type: str) -> Any:
    """Converte o valor do CSV para o formato aceito pelo campo Jira."""
    if _is_empty(raw_value):
        return None

    if isinstance(raw_value, (dict, list)):
        return raw_value

    raw_as_text = str(raw_value).strip()
    parsed_json = _try_parse_json(raw_as_text)
    canonical_type = resolve_field_type(field_type)

    if canonical_type == "text":
        return raw_as_text

    if canonical_type == "richtext":
        if _is_adf_document(parsed_json):
            return parsed_json
        return _text_to_adf(raw_as_text)

    if canonical_type == "number":
        if "." in raw_as_text:
            return float(raw_as_text)
        return int(raw_as_text)

    if canonical_type == "date":
        return _to_jira_date(raw_as_text)

    if canonical_type == "datetime":
        return raw_as_text

    if canonical_type == "project":
        if isinstance(parsed_json, dict):
            return parsed_json
        return {"key": raw_as_text}

    if canonical_type == "issuetype":
        if isinstance(parsed_json, dict):
            return parsed_json
        return {"name": raw_as_text}

    if canonical_type == "user":
        if isinstance(parsed_json, dict):
            return parsed_json
        return {"accountId": raw_as_text}

    if canonical_type == "user_list":
        if isinstance(parsed_json, list):
            return parsed_json
        return [{"accountId": value} for value in _split_multi_value(raw_as_text)]

    if canonical_type == "select":
        if isinstance(parsed_json, dict):
            return parsed_json
        return {"value": raw_as_text}

    if canonical_type == "multiselect":
        if isinstance(parsed_json, list):
            return parsed_json
        return [{"value": value} for value in _split_multi_value(raw_as_text)]

    if canonical_type == "labels":
        if isinstance(parsed_json, list):
            return parsed_json
        return _split_multi_value(raw_as_text)

    if canonical_type == "json":
        return parsed_json

    return raw_as_text


def load_mapping(mapping_csv_path: str) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_csv_path, dtype=str, keep_default_na=False)
    if mapping_df.shape[1] < 3:
        raise ValueError("mapeamento.csv precisa ter no minimo 3 colunas.")

    mapping_df = mapping_df.iloc[:, :3].copy()
    mapping_df.columns = ["source_column", "jira_field_id", "field_type"]

    for column in mapping_df.columns:
        mapping_df[column] = mapping_df[column].astype(str).str.strip()

    mapping_df = mapping_df[
        (mapping_df["source_column"] != "") & (mapping_df["jira_field_id"] != "")
    ].reset_index(drop=True)

    return mapping_df


def load_tasks(dados_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dados_csv_path, dtype=str, keep_default_na=False)


def build_issue_fields(
    row: pd.Series,
    mapping_df: pd.DataFrame,
    default_project_key: str,
    default_issue_type: str,
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "project": {"key": default_project_key},
        "issuetype": {"name": default_issue_type},
    }
    description_lines: List[str] = []
    description_mappings_count = int(
        mapping_df["jira_field_id"].apply(_is_description_field).sum()
    )
    compose_description_from_many = description_mappings_count > 1

    for _, mapping in mapping_df.iterrows():
        source_column = mapping["source_column"]
        jira_field_id = mapping["jira_field_id"]
        field_type = mapping["field_type"]

        if source_column not in row.index:
            continue

        if compose_description_from_many and _is_description_field(jira_field_id):
            raw_value = str(row[source_column]).strip()
            if not _is_empty(raw_value):
                description_lines.append(f"[{source_column}]: {raw_value}")
            continue

        parsed_value = format_jira_field_value(row[source_column], field_type)
        if _is_description_field(jira_field_id):
            parsed_value = normalize_description_for_jira(parsed_value)
        if parsed_value is not None:
            fields[jira_field_id] = parsed_value

    if compose_description_from_many and description_lines:
        fields["description"] = normalize_description_for_jira(
            "\n".join(description_lines)
        )

    if isinstance(fields.get("project"), str):
        fields["project"] = {"key": fields["project"]}
    if isinstance(fields.get("issuetype"), str):
        fields["issuetype"] = {"name": fields["issuetype"]}

    if _is_empty(fields.get("summary")):
        raise ValueError("Campo obrigatorio 'summary' ausente para a linha atual.")

    return fields


def _post_issue_request(config: JiraConfig, fields: Dict[str, Any]) -> requests.Response:
    endpoint = f"{config.jira_url.rstrip('/')}/rest/api/3/issue"
    return requests.post(
        endpoint,
        auth=(config.jira_email, config.jira_api_token),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"fields": fields},
        timeout=30,
    )


def _parse_error_details(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": response.text}


def _apply_adf_retry_if_needed(
    fields: Dict[str, Any], details: Any
) -> tuple[Dict[str, Any], List[str]]:
    if not isinstance(details, dict):
        return fields, []

    field_errors = details.get("errors") or {}
    if not isinstance(field_errors, dict):
        return fields, []

    updated_fields = dict(fields)
    converted_fields: List[str] = []

    for field_name, field_message in field_errors.items():
        if field_name == "summary" or field_name not in updated_fields:
            continue
        current_value = updated_fields[field_name]
        if not isinstance(current_value, str) or _is_empty(current_value):
            continue
        if not _looks_like_adf_error(str(field_message)):
            continue
        updated_fields[field_name] = _text_to_adf(current_value.strip())
        converted_fields.append(field_name)

    return updated_fields, converted_fields


def create_issue_requests(config: JiraConfig, fields: Dict[str, Any]) -> Dict[str, Any]:
    response = _post_issue_request(config, fields)
    retried_fields: List[str] = []

    if response.status_code >= 400:
        details = _parse_error_details(response)

        retry_fields, retried_fields = _apply_adf_retry_if_needed(fields, details)
        if retried_fields:
            response = _post_issue_request(config, retry_fields)
            if response.status_code < 400:
                return response.json()
            details = _parse_error_details(response)

        if isinstance(details, dict):
            error_messages = details.get("errorMessages") or []
            field_errors = details.get("errors") or {}
            parts: List[str] = []
            if error_messages:
                parts.append(
                    "errorMessages: "
                    + "; ".join(str(message) for message in error_messages)
                )
            if field_errors:
                parts.append(
                    "errors: "
                    + "; ".join(
                        f"{field_name} -> {field_message}"
                        for field_name, field_message in field_errors.items()
                    )
                )
            if retried_fields:
                parts.append(
                    "ADF retry attempted for: " + ", ".join(sorted(retried_fields))
                )
            if not parts:
                parts.append(json.dumps(details, ensure_ascii=False))
            raise RuntimeError(f"Erro Jira {response.status_code}: {' | '.join(parts)}")
        raise RuntimeError(f"Erro Jira {response.status_code}: {details}")

    return response.json()


def _get_colab_secret(secret_name: str) -> str:
    if userdata is None:
        raise RuntimeError(
            "google.colab.userdata indisponivel. Execute este script no Google Colab."
        )
    value = userdata.get(secret_name)
    if _is_empty(value):
        raise ValueError(
            f"Segredo '{secret_name}' vazio ou ausente. Configure em Colab > Secrets."
        )
    return str(value).strip()


def load_config_from_colab(
    default_project_key: str,
    default_issue_type: str = "Task",
    dry_run: bool = False,
) -> JiraConfig:
    return JiraConfig(
        jira_url=_get_colab_secret("JIRA_URL"),
        jira_email=_get_colab_secret("JIRA_EMAIL"),
        jira_api_token=_get_colab_secret("JIRA_API_TOKEN"),
        default_project_key=default_project_key,
        default_issue_type=default_issue_type,
        dry_run=dry_run,
    )


def _apply_pandas_display_defaults() -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)


def create_issues_from_csv(
    dados_csv_path: str,
    mapping_csv_path: str,
    config: JiraConfig,
) -> pd.DataFrame:
    tasks_df = load_tasks(dados_csv_path)
    mapping_df = load_mapping(mapping_csv_path)

    missing_columns = sorted(
        set(mapping_df["source_column"].tolist()) - set(tasks_df.columns.tolist())
    )
    if missing_columns:
        print(f"Aviso: colunas nao encontradas no dados.csv: {missing_columns}")

    results: List[Dict[str, Any]] = []
    for row_idx, row in tasks_df.iterrows():
        csv_line = row_idx + 2  # +1 cabecalho, +1 indice zero-based
        try:
            fields = build_issue_fields(
                row=row,
                mapping_df=mapping_df,
                default_project_key=config.default_project_key,
                default_issue_type=config.default_issue_type,
            )

            if config.dry_run:
                issue_key = None
                status = "DRY_RUN"
                error = None
            else:
                created = create_issue_requests(config, fields)
                issue_key = created.get("key")
                status = "CREATED"
                error = None

            results.append(
                {
                    "linha_csv": csv_line,
                    "status": status,
                    "issue_key": issue_key,
                    "error": error,
                }
            )
        except Exception as exc:  # erro por linha
            results.append(
                {
                    "linha_csv": csv_line,
                    "status": "ERROR",
                    "issue_key": None,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Ajuste os caminhos se necessario.
    DADOS_CSV_PATH = "/content/dados.csv"
    MAPEAMENTO_CSV_PATH = "/content/mapeamento.csv"

    # Defaults usados quando nao vierem mapeados no CSV.
    DEFAULT_PROJECT_KEY = "PROJ"
    DEFAULT_ISSUE_TYPE = "Task"

    # Use True para validar payloads sem criar issues no Jira.
    DRY_RUN = True

    jira_config = load_config_from_colab(
        default_project_key=DEFAULT_PROJECT_KEY,
        default_issue_type=DEFAULT_ISSUE_TYPE,
        dry_run=DRY_RUN,
    )

    result_df = create_issues_from_csv(
        dados_csv_path=DADOS_CSV_PATH,
        mapping_csv_path=MAPEAMENTO_CSV_PATH,
        config=jira_config,
    )

    _apply_pandas_display_defaults()
    print(result_df)
