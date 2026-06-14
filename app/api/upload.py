"""文档上传接口"""

from pathlib import Path
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import config
from app.engine.rag_engine import rag_engine
from app.models.document import IngestResult
from loguru import logger

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并自动索引"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in config.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}，支持: {', '.join(config.allowed_extensions_list)}",
        )

    upload_dir = Path(config.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        if len(content) > config.max_file_size_bytes:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({config.max_file_size_mb}MB)",
            )
        await f.write(content)

    logger.info(f"文件已保存: {file_path} ({len(content)} bytes)")

    result: IngestResult = await rag_engine.ingest(file_path)

    if result.status == "failed":
        return {
            "status": "failed",
            "filename": file.filename,
            "error": result.error,
        }

    return {
        "status": "indexed",
        "filename": file.filename,
        "document_id": result.document_id,
        "chunks": result.chunks,
        "tables": result.tables,
        "images": result.images,
        "formulas": result.formulas,
    }
