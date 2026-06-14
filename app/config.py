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
    milvus_timeout: int = 30000  # 毫秒
    milvus_db_name: str = "optical_rag"

    # RAG
    rag_top_k: int = 5
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # RAG-Anything 引擎
    ra_parser: str = "mineru"        # 解析引擎: "mineru" | "docling" | "paddleocr"
    ra_parse_method: str = "auto"    # 解析方式: "auto" | "ocr" | "txt"
    ra_device: str = "cpu"           # 推理设备: "cpu" | "cuda" | "mps"
    ra_lang: str = ""                # 文档语言: ""(自动) | "ch" | "en"
    ra_query_mode: str = "hybrid"    # 查询模式: "hybrid" | "local" | "global" | "naive"
    ra_enable_images: bool = True    # 图片理解（光路图、光谱图）
    ra_enable_tables: bool = True    # 表格理解（规格参数表）
    ra_enable_formulas: bool = True  # 公式理解（光学公式）

    # 文档处理
    allowed_extensions: str = "pdf,docx,xlsx,pptx,txt,md,png,jpg,jpeg"
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # 会话持久化
    sqlite_db_path: str = "./data/optical_rag.db"

    # MCP 扩展（可选）
    mcp_servers_config: str = "{}"

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


config = Settings()
