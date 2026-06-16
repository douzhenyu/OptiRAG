# OptiRAG 部署指南

## 架构

```
┌─────────────────────────────────────────┐
│                  ECS / 云服务器           │
│  ┌─────────────────────────────────────┐│
│  │  Nginx (可选)                        ││
│  │  :80 → FastAPI:9900                 ││
│  │  /static/ → 静态文件                 ││
│  └──────────────┬──────────────────────┘│
│                 │                        │
│  ┌──────────────▼──────────────────────┐│
│  │  OptiRAG (FastAPI + uvicorn)        ││
│  │  :9900                              ││
│  └──────────────┬──────────────────────┘│
│                 │                        │
│  ┌──────────────▼──────────────────────┐│
│  │  Milvus Standalone :19530           ││
│  │  ├── etcd                           ││
│  │  └── MinIO (对象存储)                ││
│  └─────────────────────────────────────┘│
│                                         │
│  持久化卷:                               │
│  ./data       → SQLite 会话              │
│  ./uploads    → 上传文档                  │
│  ./volumes/   → Milvus 向量数据           │
└─────────────────────────────────────────┘
```

## 快速部署（5 分钟）

### 1. 服务器准备

```bash
# ECS 最低配置：2C4G，20G 硬盘，Ubuntu 22.04
ssh root@<your-server-ip>

# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 安装 Docker Compose
apt install -y docker-compose-plugin
```

### 2. 拉取项目

```bash
git clone https://github.com/douzhenyu/OptiRAG.git
cd OptiRAG
```

### 3. 配置环境变量

```bash
# 复制生产环境配置
cp .env.production .env

# 编辑 .env，填入真实的 API Key
vim .env
# 必改：DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
```

### 4. 启动服务

```bash
# 构建并启动所有容器
docker compose up -d

# 查看日志
docker compose logs -f app

# 等待健康检查通过（约 30 秒）
docker compose ps
```

### 5. 验证

```bash
# 健康检查
curl http://localhost:9900/health

# 上传测试文档
curl -X POST http://localhost:9900/api/upload \
  -F "file=@README.md"

# 问答测试
curl -X POST http://localhost:9900/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"OptiRAG 是什么？"}'
```

## 启用 Nginx（有域名时）

```bash
# 1. 编辑 docker-compose.yml，取消 nginx 部分的注释
# 2. 编辑 nginx.conf，将 server_name _ 改为你的域名
# 3. 重启
docker compose up -d
```

## HTTPS（Let's Encrypt）

```bash
# 安装 certbot
apt install -y certbot

# 获取证书（需要域名已解析到服务器）
certbot certonly --standalone -d your-domain.com

# 修改 nginx.conf 添加 SSL 配置，然后重启
docker compose restart nginx
```

## 常用运维命令

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f app          # OptiRAG
docker compose logs -f milvus       # Milvus

# 重启服务
docker compose restart app

# 进入容器调试
docker exec -it optirag-app bash

# 备份数据
tar -czf backup-$(date +%Y%m%d).tar.gz data/ uploads/ volumes/

# 更新部署
git pull
docker compose build app
docker compose up -d app
```

## 资源配置建议

| 规模 | CPU | 内存 | 磁盘 |
|------|-----|------|------|
| 个人/小团队 | 2C | 4G | 20G |
| 中型知识库 | 4C | 8G | 50G |
| 大型知识库 | 8C | 16G | 100G+ |

---

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | **必填**，阿里云 DashScope API Key | — |
| `LLM_MODEL` | 文本生成模型 | `qwen-max` |
| `VISION_MODEL` | 视觉理解模型 | `qwen-vl-max` |
| `MILVUS_HOST` | Milvus 地址（Docker 内用容器名 `milvus`） | `localhost` |
| `RA_DEVICE` | 推理设备，云服务器用 `cpu` | `cpu` |
| `RA_QUERY_MODE` | 检索模式：`hybrid` / `local` / `global` / `naive` | `hybrid` |
| `MAX_FILE_SIZE_MB` | 上传文件大小上限 | `50` |
| `CHUNK_SIZE` | 文档分块 token 数 | `1200` |
