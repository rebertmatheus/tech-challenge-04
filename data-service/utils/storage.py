from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
import logging

def get_storage_client(conn_str: str, container_name: str):
    """Retorna container client configurado"""
    logger = logging.getLogger(__name__)

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)

        try:
            container_client.create_container()
            logger.info(f"Container {container_name} criado")
        except ResourceExistsError:
            logger.info(f"Container {container_name} já existe")

        return container_client
    except Exception as e:
        logger.exception("Falha ao conectar no Azure Blob Storage")
        raise