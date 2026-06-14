"""RAG-Anything 引擎封装 — 文档摄入与多模态检索"""

import json
import hashlib
from pathlib import Path
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

    # ── 元数据持久化 ──────────────────────────────────────────

    def _init_metadata_store(self) -> None:
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
        raw = {k: v.model_dump(mode="json") for k, v in self._metadata.items()}
        self._metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))

    # ── LLM / VLM / Embedding 回调函数 ────────────────────────

    @staticmethod
    def _create_llm_func():
        """创建文本 LLM 回调函数（OpenAI 兼容模式 → DashScope）"""
        from openai import OpenAI

        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **kwargs,
        ) -> str:
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            return response.choices[0].message.content or ""

        return llm_func

    @staticmethod
    def _create_vision_func():
        """创建视觉 VLM 回调函数（Qwen-VL → DashScope）"""
        from openai import OpenAI

        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        async def vision_func(
            messages: list[dict] | None = None,
            image_data: str | None = None,
            prompt: str | None = None,
            **kwargs,
        ) -> str:
            # 构建带图片的消息
            content: list[dict] = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data},
                })

            msgs = [{"role": "user", "content": content}] if content else (messages or [])
            response = client.chat.completions.create(
                model=config.vision_model,
                messages=msgs,
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return response.choices[0].message.content or ""

        return vision_func

    @staticmethod
    def _create_embedding_func():
        """创建 Embedding 回调函数（DashScope text-embedding-v4）"""
        from openai import OpenAI
        from raganything import EmbeddingFunc

        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        async def _embed(texts: list[str]) -> list[list[float]]:
            # DashScope embedding API 单次最多 25 条
            batch_size = 20
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = client.embeddings.create(
                    model=config.embedding_model,
                    input=batch,
                    dimensions=config.embedding_dimensions,
                    encoding_format="float",
                )
                all_embeddings.extend([item.embedding for item in resp.data])
            return all_embeddings

        return EmbeddingFunc(
            embedding_dim=config.embedding_dimensions,
            max_token_size=8192,
            func=_embed,
        )

    # ── RAG-Anything 初始化 ───────────────────────────────────

    async def _ensure_raganything(self):
        """延迟初始化 RAG-Anything"""
        if self._raganything is not None:
            return
        try:
            from raganything import RAGAnything, RAGAnythingConfig

            ra_config = RAGAnythingConfig(
                working_dir=config.upload_dir,
                parser=config.ra_parser,
                parse_method=config.ra_parse_method,
                enable_image_processing=config.ra_enable_images,
                enable_table_processing=config.ra_enable_tables,
                enable_equation_processing=config.ra_enable_formulas,
            )

            self._raganything = RAGAnything(
                config=ra_config,
                llm_model_func=self._create_llm_func(),
                vision_model_func=self._create_vision_func(),
                embedding_func=self._create_embedding_func(),
            )
            logger.info(
                "RAG-Anything 引擎初始化完成 "
                f"(parser=mineru, llm={config.llm_model}, "
                f"vision={config.vision_model}, embedding={config.embedding_model})"
            )
        except ImportError:
            logger.warning(
                "RAG-Anything 未安装，使用降级模式。"
                "多模态能力(图片/表格/公式)不可用。"
            )
            self._raganything = None

    # ── 文件去重 ──────────────────────────────────────────────

    def _compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _file_hash_exists(self, file_hash: str) -> str | None:
        for doc_id, info in self._metadata.items():
            if info.status == "indexed":
                existing_path = Path(config.upload_dir) / info.filename
                if existing_path.exists():
                    if self._compute_file_hash(existing_path) == file_hash:
                        return doc_id
        return None

    # ── 文档摄入 ──────────────────────────────────────────────

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
                error=f"不支持的文件格式: .{file_ext}",
            )

        if file_path.stat().st_size > config.max_file_size_bytes:
            return IngestResult(
                document_id="",
                filename=filename,
                status="failed",
                chunks=0, tables=0, images=0, formulas=0,
                error=f"文件大小超过限制 ({config.max_file_size_mb}MB)",
            )

        # 去重
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

        doc_id = f"doc_{file_hash[:16]}"

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
                # 使用 RAG-Anything 的 process_document_complete 处理文档
                result = await self._raganything.process_document_complete(
                    file_path=str(file_path),
                    output_dir=str(Path(config.upload_dir) / ".parsed"),
                    parse_method=config.ra_parse_method,
                    parser=config.ra_parser,
                    device=config.ra_device,
                    lang=config.ra_lang or None,
                    formula=config.ra_enable_formulas,
                    table=config.ra_enable_tables,
                )
                chunks = getattr(result, "chunks_count", 0)
                tables = getattr(result, "tables_count", 0)
                images = getattr(result, "images_count", 0)
                formulas = getattr(result, "formulas_count", 0)
            else:
                chunks = 1
                tables = images = formulas = 0
                logger.warning(f"RAG-Anything 不可用，{filename} 使用降级模式")

            info.status = "indexed"
            info.chunks = chunks
            info.tables = tables
            info.images = images
            info.formulas = formulas
            self._save_metadata()

            logger.info(f"文档摄入完成: {filename} (chunks={chunks}, tables={tables}, images={images}, formulas={formulas})")
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

    # ── 检索 ──────────────────────────────────────────────────

    async def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """混合检索（hybrid 模式：向量 + 知识图谱）"""
        k = top_k or config.rag_top_k
        await self._ensure_raganything()

        if self._raganything is not None:
            try:
                result = await self._raganything.aquery(
                    query=question,
                    mode=config.ra_query_mode,
                )
                # aquery 返回结构: {"response": "...", "references": [...]}
                references = result.get("references", []) if isinstance(result, dict) else []
                # 截取 top_k
                return references[:k] if references else []
            except Exception as e:
                logger.error(f"RAG-Anything aquery 失败: {e}，尝试 naive 模式")
                try:
                    result = await self._raganything.aquery(
                        query=question,
                        mode="naive",
                    )
                    references = result.get("references", []) if isinstance(result, dict) else []
                    return references[:k] if references else []
                except Exception as e2:
                    logger.error(f"RAG-Anything naive 查询也失败: {e2}")

        logger.warning("使用降级查询，返回空结果")
        return []

    # ── 知识库管理 ────────────────────────────────────────────

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
        return self._metadata.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
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
