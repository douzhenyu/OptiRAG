"""Milvus 连接管理 — 精简版（RAG-Anything 托管 vector store）"""

from pymilvus import MilvusClient, connections
from loguru import logger
from app.config import config


class MilvusManager:
    """轻量 Milvus 连接管理器"""

    def __init__(self):
        self._client: MilvusClient | None = None

    def connect(self) -> MilvusClient:
        if self._client is not None:
            return self._client

        uri = f"http://{config.milvus_host}:{config.milvus_port}"
        self._client = MilvusClient(uri=uri)
        logger.info(f"Milvus 连接成功: {uri}")
        return self._client

    def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            connections.list_connections()
            return True
        except Exception:
            return False

    def close(self):
        try:
            connections.disconnect("default")
        except Exception:
            pass
        self._client = None


milvus_manager = MilvusManager()
