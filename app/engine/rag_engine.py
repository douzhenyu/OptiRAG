"""RAG-Anything 引擎封装 — 文档摄入与多模态检索"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Callable, Any
from loguru import logger

from app.config import config
from app.models.document import DocumentInfo, IngestResult


class RAGEngine:
    """RAG-Anything 封装，管理文档全生命周期"""

    def __init__(self):
        self._raganything = None
        self._metadata: dict[str, DocumentInfo] = {}
        self._metadata_path = Path(config.upload_dir) / ".rag_metadata.json"
        self._init_metadata_store()

    def _init_metadata_store(self) -> None:
        """从磁盘加载元数据索引"""
        Path(config.upload_dir).mkdir(parents=True, exist_ok=True)
        if self._metadata_path.exists():
            try:
                raw = json.loads(self._metadata_path.read_text())
                self._metadata = {
                    k: DocumentInfo(**v) for k, v in raw.items()
                }
                logger.info(f"加载 {len(self._metadata)} 条文档元数据")
            except Exception as e:
                logger.warning(f"元数据加载失败: {e}，使用空索引")
                self._metadata = {}

    def _save_metadata(self) -> None:
        """持久化元数据索引"""
        raw = {k: v.model_dump(mode="json") for k, v in self._metadata.items()}
        self._metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))

    async def _ensure_raganything(self):
        """延迟初始化 RAG-Anything（首次调用时才加载）"""
        if self._raganything is not None:
            return
        try:
            from raganything import RAGAnything

            self._raganything = RAGAnything(
                working_dir=config.upload_dir,
                llm_model=config.llm_model,
                llm_api_key=config.dashscope_api_key,
                llm_base_url=config.dashscope_api_base,
                vision_model=config.vision_model,
                vision_api_key=config.dashscope_api_key,
                vision_base_url=config.dashscope_api_base,
                embedding_model=config.embedding_model,
                embedding_api_key=config.dashscope_api_key,
                embedding_base_url=config.dashscope_api_base,
                embedding_dim=config.embedding_dimensions,
                vector_store_type="milvus",
                milvus_host=config.milvus_host,
                milvus_port=config.milvus_port,
            )
            await self._raganything.initialize()
            logger.info("RAG-Anything 引擎初始化完成")
        except ImportError:
            logger.warning(
                "RAG-Anything 未安装，使用纯 Milvus 降级模式。"
                "多模态能力(图片/表格/公式)不可用。"
            )
            self._raganything = None

    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件 SHA256 去重"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _file_hash_exists(self, file_hash: str) -> str | None:
        """检查文件是否已索引，返回已有文档 ID"""
        for doc_id, info in self._metadata.items():
            if info.status == "indexed":
                existing_path = Path(config.upload_dir) / info.filename
                if existing_path.exists():
                    if self._compute_file_hash(existing_path) == file_hash:
                        return doc_id
        return None

    async def ingest(self, file_path: Path) -> IngestResult:
        """摄入单个文件"""
        filename = file_path.name
        file_ext = file_path.suffix.lower().lstrip(".")

        if file_ext not in config.allowed_extensions_list:
            return IngestResult(
                document_id="",
                filename=filename,
                status="failed",
                chunks=0, tables=0, images=0, formulas=0,
                error=f"不支持的文件格式: .{file_ext}"
            )

        if file_path.stat().st_size > config.max_file_size_bytes:
            return IngestResult(
                document_id="",
                filename=filename,
                status="failed",
                chunks=0, tables=0, images=0, formulas=0,
                error=f"文件大小超过限制 ({config.max_file_size_mb}MB)"
            )

        # 去重检查
        file_hash = self._compute_file_hash(file_path)
        existing_id = self._file_hash_exists(file_hash)
        if existing_id:
            logger.info(f"文件已存在: {filename} -> {existing_id}")
            existing = self._metadata[existing_id]
            return IngestResult(
                document_id=existing_id,
                filename=filename,
                status="indexed",
                chunks=existing.chunks,
                tables=existing.tables,
                images=existing.images,
                formulas=existing.formulas,
            )

        # 生成文档 ID
        doc_id = f"doc_{file_hash[:16]}"

        # 标记为索引中
        info = DocumentInfo(
            document_id=doc_id,
            filename=filename,
            file_type=file_ext,
            file_size=file_path.stat().st_size,
            status="indexing",
        )
        self._metadata[doc_id] = info
        self._save_metadata()

        try:
            await self._ensure_raganything()

            if self._raganything is not None:
                result = await self._raganything.ingest_document(str(file_path))
                chunks = getattr(result, "chunks_count", 0)
                tables = getattr(result, "tables_count", 0)
                images = getattr(result, "images_count", 0)
                formulas = getattr(result, "formulas_count", 0)
            else:
                # 降级模式：仅文本摄入
                chunks = 1
                tables = images = formulas = 0
                logger.warning(f"RAG-Anything 不可用，{filename} 使用降级模式")

            info.status = "indexed"
            info.chunks = chunks
            info.tables = tables
            info.images = images
            info.formulas = formulas
            self._save_metadata()

            logger.info(f"文档摄入完成: {filename} (chunks={chunks})")
            return IngestResult(
                document_id=doc_id,
                filename=filename,
                status="indexed",
                chunks=chunks,
                tables=tables,
                images=images,
                formulas=formulas,
            )

        except Exception as e:
            info.status = "failed"
            self._save_metadata()
            logger.error(f"文档摄入失败: {filename}, {e}")
            return IngestResult(
                document_id=doc_id,
                filename=filename,
                status="failed",
                chunks=0, tables=0, images=0, formulas=0,
                error=str(e),
            )

    async def ingest_batch(
        self,
        paths: list[Path],
        on_progress: Callable[[int, int, str], Any] | None = None,
    ) -> list[IngestResult]:
        """批量摄入文档"""
        results: list[IngestResult] = []
        total = len(paths)
        for i, path in enumerate(paths):
            if on_progress:
                on_progress(i + 1, total, path.name)
            result = await self.ingest(path)
            results.append(result)
        return results

    async def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """混合检索"""
        k = top_k or config.rag_top_k
        await self._ensure_raganything()

        if self._raganything is not None:
            try:
                result = await self._raganything.query(question, top_k=k)
                return result.get("results", [])
            except Exception as e:
                logger.error(f"RAG-Anything 查询失败: {e}")

        logger.warning("使用降级查询")
        return []

    def list_documents(
        self, page: int = 1, page_size: int = 20,
        keyword: str | None = None, file_type: str | None = None,
    ) -> tuple[list[DocumentInfo], int]:
        """列出已索引文档（分页+筛选）"""
        docs = list(self._metadata.values())
        if keyword:
            docs = [d for d in docs if keyword.lower() in d.filename.lower()]
        if file_type:
            docs = [d for d in docs if d.file_type == file_type]
        docs.sort(key=lambda d: d.created_at, reverse=True)
        total = len(docs)
        start = (page - 1) * page_size
        return docs[start:start + page_size], total

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        """获取文档详情"""
        return self._metadata.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """删除文档及其索引"""
        info = self._metadata.get(doc_id)
        if info is None:
            return False
        file_path = Path(config.upload_dir) / info.filename
        if file_path.exists():
            file_path.unlink()
        del self._metadata[doc_id]
        self._save_metadata()
        logger.info(f"文档已删除: {info.filename} ({doc_id})")
        return True


# 全局单例
rag_engine = RAGEngine()
