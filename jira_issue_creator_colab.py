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


def resolve_field_type(field_type: str) -> str:
    token = _normalize_token(field_type)
    aliases = {
        "text": {"texto", "text", "string", "plain_text"},
        "number": {"numero", "number", "int", "integer", "float", "double"},
        "user": {"usuario", "user", "accountid"},
        "user_list": {"usuarios", "multiuser", "user_list", "lista_usuarios"},
        "select": {"select", "single_select", "lista_selecao", "option"},
        "multiselect": {"multiselect", "multi_select", "lista", "checkbox"},
        "labels": {"labels", "rotulos", "tags"},
        "date": {"date", "data"},
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

    if canonical_type == "number":
        if "." in raw_as_text:
            return float(raw_as_text)
        return int(raw_as_text)

    if canonical_type == "date":
        return raw_as_text

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

    for _, mapping in mapping_df.iterrows():
        source_column = mapping["source_column"]
        jira_field_id = mapping["jira_field_id"]
        field_type = mapping["field_type"]

        if source_column not in row.index:
            continue

        parsed_value = format_jira_field_value(row[source_column], field_type)
        if parsed_value is not None:
            fields[jira_field_id] = parsed_value

    if isinstance(fields.get("project"), str):
        fields["project"] = {"key": fields["project"]}
    if isinstance(fields.get("issuetype"), str):
        fields["issuetype"] = {"name": fields["issuetype"]}

    if _is_empty(fields.get("summary")):
        raise ValueError("Campo obrigatorio 'summary' ausente para a linha atual.")

    return fields


def create_issue_requests(config: JiraConfig, fields: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = f"{config.jira_url.rstrip('/')}/rest/api/3/issue"
    response = requests.post(
        endpoint,
        auth=(config.jira_email, config.jira_api_token),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"fields": fields},
        timeout=30,
    )

    if response.status_code >= 400:
        try:
            details = response.json()
        except ValueError:
            details = {"error": response.text}
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

    print(result_df)
