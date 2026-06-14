"""配置管理模块 — 光学科研 RAG 助手"""

from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用
    app_name: str = "OpticalRAG"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope LLM
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-max"
    vision_model: str = "qwen-vl-max"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_db_name: str = "optical_rag"

    # RAG
    rag_top_k: int = 5
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # 文档处理
    allowed_extensions: str = "pdf,docx,xlsx,pptx,txt,md,png,jpg,jpeg"
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # MCP 扩展（可选）
    mcp_servers_config: str = "{}"

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


config = Settings()
