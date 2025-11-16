from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


def _get_blob_client(conn_str: str, container_name: str):
    service = BlobServiceClient.from_connection_string(conn_str)
    return service.get_container_client(container_name)


def read_parquet_from_blob(
    conn_str: str,
    container_name: str,
    blob_path: str,
) -> pd.DataFrame:
    container_client = _get_blob_client(conn_str, container_name)
    try:
        blob_client = container_client.get_blob_client(blob_path)
        stream = io.BytesIO()
        download_stream = blob_client.download_blob()
        stream.write(download_stream.readall())
        stream.seek(0)
        return pd.read_parquet(stream)
    except ResourceNotFoundError as exc:
        raise FileNotFoundError(f"Blob {blob_path} not found in container {container_name}") from exc


def upload_local_file_to_blob(
    conn_str: str,
    container_name: str,
    local_path: str,
    blob_path: str,
    overwrite: bool = True,
) -> None:
    container_client = _get_blob_client(conn_str, container_name)
    blob_client = container_client.get_blob_client(blob_path)
    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=overwrite)


def ensure_local_dir(path: str) -> None:
    """Create the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
