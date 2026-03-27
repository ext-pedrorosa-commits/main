"""
Script para Google Colab:
1) Ler um CSV com IDs de custom fields do Jira.
2) Consultar a API do Jira Cloud para cada campo.
3) Adicionar uma coluna ao lado com os valores disponíveis.
4) Adicionar outra coluna com o nome do tipo do campo.
5) Quando o campo nao for de selecao/lista, registrar uma mensagem com o tipo.

Uso rapido no Colab:
    !pip -q install pandas requests
    # Opcional: defina variaveis de ambiente
    # %env JIRA_BASE_URL=https://sua-empresa.atlassian.net
    # %env JIRA_EMAIL=seu-email@empresa.com
    # %env JIRA_API_TOKEN=seu_token
    !python jira_custom_fields_colab.py
"""

from __future__ import annotations

import os
import re
import time
from getpass import getpass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None


SELECT_FIELD_TOKENS = {
    "select",
    "multiselect",
    "radiobuttons",
    "multicheckboxes",
    "cascadingselect",
}


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        timeout_seconds: int = 30,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {"Accept": "application/json"}

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        backoff_seconds = 2
        last_error = "Erro desconhecido."

        for attempt in range(1, self.max_retries + 1):
            response = requests.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=self.headers,
                auth=self.auth,
                params=params,
                timeout=self.timeout_seconds,
            )

            if response.status_code == 429 and attempt < self.max_retries:
                retry_after = int(response.headers.get("Retry-After", "2"))
                time.sleep(max(retry_after, 1))
                continue

            if 500 <= response.status_code <= 599 and attempt < self.max_retries:
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue

            if response.status_code >= 400:
                body = response.text.replace("\n", " ").strip()
                if len(body) > 300:
                    body = f"{body[:300]}..."
                last_error = f"{method} {path} retornou {response.status_code}: {body}"
                break

            if not response.text.strip():
                return {}
            return response.json()

        raise RuntimeError(last_error)

    def get_custom_fields_map(self) -> Dict[str, Dict]:
        fields = self._request("GET", "/rest/api/3/field")
        return {
            field["id"]: field
            for field in fields
            if str(field.get("id", "")).startswith("customfield_")
        }

    def list_field_contexts(self, field_id: str) -> List[Dict]:
        contexts: List[Dict] = []
        start_at = 0
        max_results = 50

        while True:
            payload = self._request(
                "GET",
                f"/rest/api/3/field/{field_id}/context",
                params={"startAt": start_at, "maxResults": max_results},
            )
            values = payload.get("values", [])
            contexts.extend(values)

            is_last = bool(payload.get("isLast", False))
            if is_last or not values:
                break
            start_at += payload.get("maxResults", max_results)

        return contexts

    @staticmethod
    def _flatten_options(values: List[Dict], parent_value: Optional[str] = None) -> List[str]:
        flattened: List[str] = []
        for option in values:
            raw_value = str(option.get("value") or option.get("id") or "").strip()
            if not raw_value:
                continue

            label = f"{parent_value} > {raw_value}" if parent_value else raw_value
            children = option.get("children") or []
            if children:
                flattened.extend(JiraClient._flatten_options(children, raw_value))
            else:
                flattened.append(label)

        return flattened

    def list_context_options(self, field_id: str, context_id: str) -> List[str]:
        options: List[str] = []
        start_at = 0
        max_results = 100

        while True:
            payload = self._request(
                "GET",
                f"/rest/api/3/field/{field_id}/context/{context_id}/option",
                params={"startAt": start_at, "maxResults": max_results},
            )
            values = payload.get("values", [])
            options.extend(self._flatten_options(values))

            is_last = bool(payload.get("isLast", False))
            if is_last or not values:
                break
            start_at += payload.get("maxResults", max_results)

        # Remove duplicados preservando ordem
        return list(dict.fromkeys(options))


def normalize_field_id(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"customfield_\d+", text, flags=re.IGNORECASE):
        return text.lower()

    if re.fullmatch(r"\d+", text):
        return f"customfield_{text}"

    return text


def is_select_like_field(custom_key: str) -> bool:
    normalized = str(custom_key or "").strip().lower()
    if not normalized:
        return False

    return any(
        normalized == token or normalized.endswith(f":{token}")
        for token in SELECT_FIELD_TOKENS
    )


def choose_id_column(df: pd.DataFrame) -> str:
    if df.empty:
        raise ValueError("CSV sem linhas para processar.")
    if len(df.columns) == 0:
        raise ValueError("CSV sem colunas.")

    preferred_names = {
        "custom_field_id",
        "customfield_id",
        "customfield",
        "field_id",
        "campo_id",
        "id",
    }

    for column in df.columns:
        if str(column).strip().lower() in preferred_names:
            return column

    candidate_pattern = re.compile(r"^(customfield_\d+|\d+)$", flags=re.IGNORECASE)
    scored_columns: List[Tuple[str, float]] = []
    for column in df.columns:
        values = df[column].dropna().astype(str).str.strip()
        if values.empty:
            scored_columns.append((column, 0.0))
            continue
        score = float(values.map(lambda x: bool(candidate_pattern.fullmatch(x))).mean())
        scored_columns.append((column, score))

    best_column, best_score = max(scored_columns, key=lambda item: item[1])
    if best_score >= 0.6:
        return best_column

    # Fallback para primeira coluna
    return str(df.columns[0])


def summarize_field_options(
    field_id: str,
    fields_map: Dict[str, Dict],
    jira: JiraClient,
) -> Tuple[str, str]:
    field = fields_map.get(field_id)
    if not field:
        return f"Campo '{field_id}' nao encontrado no Jira.", "nao_encontrado"

    field_name = str(field.get("name") or field_id)
    schema = field.get("schema") or {}
    custom_key = str(schema.get("custom") or "")
    generic_type = str(schema.get("type") or "desconhecido")
    field_type_label = custom_key or generic_type
    field_type_name = (
        custom_key.split(":")[-1].strip()
        if ":" in custom_key
        else (custom_key.strip() or generic_type)
    )
    if not field_type_name:
        field_type_name = "desconhecido"

    if not is_select_like_field(custom_key):
        return (
            f"Campo '{field_name}' do tipo '{field_type_label}' "
            "nao e lista/selecao.",
            field_type_name,
        )

    try:
        contexts = jira.list_field_contexts(field_id)
    except RuntimeError as exc:
        return f"Erro ao listar contextos de '{field_name}': {exc}", field_type_name

    if not contexts:
        return (
            f"Campo '{field_name}' nao possui contextos configurados.",
            field_type_name,
        )

    context_summaries: List[str] = []
    for context in contexts:
        context_id = str(context.get("id"))
        context_name = str(context.get("name") or f"contexto_{context_id}")
        try:
            options = jira.list_context_options(field_id, context_id)
        except RuntimeError as exc:
            context_summaries.append(f"{context_name}: erro ao buscar opcoes ({exc})")
            continue

        if options:
            context_summaries.append(f"{context_name}: {', '.join(options)}")
        else:
            context_summaries.append(f"{context_name}: sem opcoes")

    return " | ".join(context_summaries), field_type_name


def process_csv(
    input_csv_path: str,
    output_csv_path: str,
    jira: JiraClient,
    id_column: Optional[str] = None,
) -> Tuple[str, str, str]:
    df = pd.read_csv(input_csv_path)
    target_column = id_column or choose_id_column(df)

    if target_column not in df.columns:
        raise ValueError(f"Coluna '{target_column}' nao encontrada no CSV.")

    output_column = f"{target_column}_jira_valores"
    output_type_column = f"{target_column}_jira_tipo_campo"
    insertion_index = df.columns.get_loc(target_column) + 1
    if output_column in df.columns:
        df[output_column] = ""
    else:
        df.insert(insertion_index, output_column, "")
    type_column_index = df.columns.get_loc(output_column) + 1
    if output_type_column in df.columns:
        df[output_type_column] = ""
    else:
        df.insert(type_column_index, output_type_column, "")

    fields_map = jira.get_custom_fields_map()
    cache: Dict[str, Tuple[str, str]] = {}

    total_rows = len(df)
    for idx, raw_value in enumerate(df[target_column], start=1):
        normalized_id = normalize_field_id(raw_value)
        if not normalized_id:
            result = "ID vazio ou invalido."
            type_result = "invalido"
        else:
            if normalized_id not in cache:
                cache[normalized_id] = summarize_field_options(
                    field_id=normalized_id,
                    fields_map=fields_map,
                    jira=jira,
                )
            result, type_result = cache[normalized_id]

        df.at[idx - 1, output_column] = result
        df.at[idx - 1, output_type_column] = type_result

        if idx == 1 or idx % 10 == 0 or idx == total_rows:
            print(f"Processando linha {idx}/{total_rows}...")

    df.to_csv(output_csv_path, index=False)
    return target_column, output_column, output_type_column


def prompt_for_credentials() -> Tuple[str, str, str]:
    base_url = os.getenv("JIRA_BASE_URL", "").strip()
    email = os.getenv("JIRA_EMAIL", "").strip()
    api_token = os.getenv("JIRA_API_TOKEN", "").strip()

    if not base_url:
        base_url = input("JIRA_BASE_URL (ex: https://sua-empresa.atlassian.net): ").strip()
    if not email:
        email = input("JIRA_EMAIL: ").strip()
    if not api_token:
        api_token = getpass("JIRA_API_TOKEN: ").strip()

    if not base_url or not email or not api_token:
        raise ValueError("Base URL, email e API token sao obrigatorios.")

    return base_url, email, api_token


def resolve_input_csv_path() -> str:
    env_path = os.getenv("INPUT_CSV_PATH", "").strip()
    if env_path:
        return env_path

    user_path = input(
        "Caminho do CSV (vazio para upload no Colab): "
    ).strip()
    if user_path:
        return user_path

    if colab_files is None:
        raise ValueError(
            "Sem caminho informado e upload do Colab indisponivel. "
            "Defina INPUT_CSV_PATH ou informe o caminho manualmente."
        )

    print("Fazendo upload do CSV...")
    uploaded = colab_files.upload()
    if not uploaded:
        raise ValueError("Nenhum arquivo enviado.")
    uploaded_file_name = next(iter(uploaded.keys()))
    return uploaded_file_name


def main() -> None:
    print("Iniciando processamento de custom fields do Jira...")
    base_url, email, api_token = prompt_for_credentials()
    input_csv_path = resolve_input_csv_path()
    output_csv_path = os.getenv("OUTPUT_CSV_PATH", "custom_fields_com_valores.csv")
    id_column_env = os.getenv("FIELD_ID_COLUMN", "").strip() or None

    jira = JiraClient(base_url=base_url, email=email, api_token=api_token)
    used_id_column, created_column, created_type_column = process_csv(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        jira=jira,
        id_column=id_column_env,
    )

    print(
        f"Concluido. Coluna de IDs: '{used_id_column}'. "
        f"Colunas criadas/atualizadas: '{created_column}' e '{created_type_column}'."
    )
    print(f"Arquivo gerado: {output_csv_path}")

    if colab_files is not None:
        print("Baixando resultado...")
        colab_files.download(output_csv_path)


if __name__ == "__main__":
    main()
