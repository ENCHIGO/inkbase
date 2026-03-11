# Inkbase: 面向 Markdown 的高性能数据库中间件

## Context

构建一款面向 Markdown 的高性能数据库，作为服务端中间件服务所有 AI Agent。项目从零开始，当前仅有 README.md。

**技术选型决定**：Rust / 全功能（结构化查询+语义搜索+知识图谱）/ MCP+REST 双协议 / 自研存储引擎

---

## 系统架构总览

```
 AI Agents / Clients
         │
 ┌───────┴────────────────────────────────┐
 │ INTERFACE LAYER                        │
 │  MCP Server (rmcp)  │  REST API (axum) │
 └───────┬────────────────────────────────┘
         │
 ┌───────┴────────────────────────────────┐
 │ QUERY ENGINE LAYER                     │
 │  Query Planner & Executor              │
 │  Structural / FullText / Vector / Graph│
 └───────┬────────────────────────────────┘
         │
 ┌───────┴────────────────────────────────┐
 │ DOMAIN LAYER                           │
 │  Markdown Parser(comrak) │ Graph(petgraph) │ Embedding(ort) │
 └───────┬────────────────────────────────┘
         │
 ┌───────┴────────────────────────────────┐
 │ STORAGE ENGINE LAYER                   │
 │  Document/Block/Link/Tag Store         │
 │  Buffer Pool │ WAL │ MVCC │ B+Tree     │
 └───────┬────────────────────────────────┘
      [Filesystem]
```

---

## Rust Workspace 结构

```
inkbase/
  Cargo.toml                    # workspace root
  crates/
    inkbase-core/                # 核心类型、trait、错误类型
    inkbase-parser/              # Markdown 解析 (comrak)、frontmatter、链接提取
    inkbase-storage/             # 存储引擎 (Phase 1 用 sled, Phase 4 自研)
    inkbase-index/               # 搜索索引 (tantivy 全文 + HNSW 向量)
    inkbase-graph/               # 知识图谱 (petgraph)
    inkbase-embedding/           # 向量嵌入管道 (ort/OpenAI/Ollama)
    inkbase-query/               # 查询引擎、MQL DSL
    inkbase-mcp/                 # MCP 服务端 (rmcp)
    inkbase-api/                 # REST API (axum)
    inkbase-server/              # 二进制入口 (clap CLI)
  tests/                        # 集成测试
  benches/                      # 性能基准测试
```

---

## 核心数据模型

5 种存储记录类型：

1. **DocumentRecord** — 一个 Markdown 文件：`doc_id, path, frontmatter, raw_content_hash(BLAKE3), created_at, updated_at, version`
2. **BlockRecord** — 一个结构元素（标题/段落/代码块/列表/表格等）：`block_id, doc_id, block_type, depth, ordinal, parent_block_id, heading_level, language, text_content, raw_markdown`
3. **LinkRecord** — 文档间链接：`link_id, source_doc_id, source_block_id, target, target_doc_id, link_type, anchor_text`
4. **EmbeddingRecord** — 向量嵌入：`embedding_id, doc_id, block_id, vector, model_id`
5. **TagRecord** — Frontmatter 标签：`tag_id, doc_id, key, value`

---

## 存储引擎设计（自研，Phase 4 实现）

- **页式存储**：8KB 页，Slotted Page 布局，溢出页机制
- **索引**：B+Tree（结构索引 + 标签索引）
- **WAL**：写前日志，fsync 保证持久性，64MB 段轮转
- **并发**：Phase 1 用 RwLock 单写多读；Phase 4+ 实现完整 MVCC 快照隔离
- **文件格式**：magic `MDOTADB\0`，含 page_count, free_list_head, root_btree_page, wal_lsn

---

## 四大索引

| 索引类型 | 实现 | 用途 |
|---------|------|------|
| 结构索引 | B+Tree `(doc_id, block_type, ordinal)` | "找所有 H2 标题"、"找所有 Rust 代码块" |
| 全文索引 | Tantivy 嵌入式 | BM25 全文检索 |
| 向量索引 | HNSW 内存 + 磁盘快照 | 语义相似度搜索 |
| 图索引 | petgraph DiGraph 内存 + 邻接表磁盘 | 链接遍历、反向链接、最短路径 |

---

## MCP Server Tools（AI Agent 主接口）

| Tool | 功能 |
|------|------|
| `ingest_document` | 解析并存储 Markdown 文档 |
| `get_document` | 获取文档及其结构 |
| `query_blocks` | 按结构查询块（类型/标题级别/语言） |
| `list_documents` | 列出文档（支持过滤） |
| `search_fulltext` | 全文检索 |
| `search_semantic` | 语义相似度搜索 |
| `get_links` / `get_backlinks` | 知识图谱遍历 |
| `graph_query` | 图算法（最短路径/邻域/中心性） |
| `delete_document` | 删除文档 |

MCP Resources: `inkbase://documents/{path}`, `inkbase://stats`, `inkbase://graph/overview`

---

## REST API 端点

```
POST/GET/PUT/DELETE  /api/v1/documents[/{path}]
GET                  /api/v1/documents/{path}/blocks
GET                  /api/v1/documents/{path}/links|backlinks
POST                 /api/v1/search/fulltext
POST                 /api/v1/search/semantic
GET                  /api/v1/graph/links|backlinks|shortest-path|stats
GET                  /api/v1/stats
POST                 /api/v1/admin/reindex|vacuum
GET                  /api/v1/health
```

---

## 关键依赖

| 用途 | Crate |
|------|-------|
| 异步运行时 | `tokio` |
| HTTP 框架 | `axum` + `tower-http` |
| Markdown 解析 | `comrak`（完整 AST，优于 pulldown-cmark 的流式解析） |
| MCP 协议 | `rmcp` |
| 全文搜索 | `tantivy` |
| 向量索引 | `hnswlib-rs` |
| 图数据结构 | `petgraph` |
| 本地嵌入模型 | `ort`（ONNX Runtime，all-MiniLM-L6-v2） |
| 临时存储(MVP) | `sled` |
| CLI | `clap` |
| 日志 | `tracing` + `tracing-subscriber` |
| 哈希 | `blake3` |
| 序列化 | `serde` + `serde_json` + `serde_yaml` |

---

## 分阶段实施计划

### Phase 1: Walking Skeleton（先做出来能用）

**目标**：Markdown 文档的摄取、存储、结构化查询，通过 MCP 可用。

1. 创建 Cargo workspace + 所有 crate 骨架
2. `inkbase-core` — 类型定义、错误类型、配置
3. `inkbase-parser` — comrak 集成、AST→BlockRecord、frontmatter 解析、链接提取
4. `inkbase-storage` — **用 sled 作为临时存储**（trait 抽象，后续可替换）
5. `inkbase-mcp` — rmcp stdio MCP 服务，实现 `ingest_document`, `get_document`, `query_blocks`, `list_documents`, `delete_document`
6. `inkbase-server` — 二进制入口，启动 MCP 服务

### Phase 2: 搜索 + REST + 图

- `inkbase-index` — Tantivy 全文搜索集成
- `inkbase-graph` — petgraph 内存图，链接/反向链接遍历
- `inkbase-api` — axum REST API
- MCP 新增 `search_fulltext`, `get_links`, `get_backlinks`

### Phase 3: 语义搜索

- `inkbase-embedding` — ort 本地嵌入 + 异步管道
- `inkbase-index` 向量部分 — HNSW 索引
- MCP/REST 新增 `search_semantic`

### Phase 4: 自研存储引擎

- 页式存储、B+Tree、Buffer Pool、WAL
- 替换 sled，迁移工具
- 基准测试验证性能

### Phase 5: 高级功能

- MVCC 快照隔离
- MQL 查询语言 (pest 语法)
- 查询优化器
- 高级图算法 (PageRank, 连通分量)
- OpenAI/Ollama 嵌入提供者
- MCP streamable-http 远程传输

---

## 验证方案

1. **Phase 1 完成后**：启动 MCP 服务 → 用 Claude Code 连接 → ingest 一个 Markdown 文件 → query_blocks 查看结构 → get_document 验证存储
2. **Phase 2 完成后**：全文搜索测试 → REST API curl 测试 → 链接遍历测试
3. **集成测试**：`tests/` 目录下编写端到端测试
4. **性能基准**：`benches/` 目录下用 criterion 做基准测试（摄取吞吐量、查询延迟）

---

## 关键设计决策

- **comrak 而非 pulldown-cmark**：需要完整 AST 支持结构查询和父子关系
- **Phase 1 用 sled，Phase 4 替换为自研引擎**：快速出活，trait 抽象保证替换透明
- **Tantivy 嵌入式**：单二进制部署，无外部依赖
- **HNSW + petgraph 内存方案**：Markdown 知识库规模（<1M 文档）完全可以内存容纳
- **MCP 为主接口**：AI Agent 原生协议，REST 为辅助管理接口
- **trait-based 嵌入提供者**：模型可插拔，默认本地 ONNX 零配置可用
