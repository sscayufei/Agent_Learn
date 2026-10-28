# 知识库配置、上传来源、切片、召回测试、问答与重建响应契约；候选来源与回答门控分别返回。
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeBaseConfigUpdate(BaseModel):
    embedding_provider: str | None = None  # 向量生成提供方，须符合共享集合契约
    embedding_model: str | None = None  # 向量模型名，不能只凭维度相同就替换
    embedding_dimensions: int | None = Field(default=None, ge=64, le=4096)  # 每条向量元素数量，不是切片长度
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None  # 只向量、只关键词或混合召回
    rerank_enabled: bool | None = None  # 是否对候选再次排序
    rerank_provider: str | None = None  # 重排提供方，与 Embedding 提供方不同
    rerank_model: str | None = None  # 重排模型名，不负责生成文档向量
    top_k: int | None = Field(default=None, ge=1, le=20)  # 最终候选目标数量，过滤后可能更少
    score_threshold_enabled: bool | None = None  # 是否启用候选分数门槛
    score_threshold: float | None = Field(default=None, ge=0, le=1)  # 检索分数门槛，不是答案正确率
    chunk_size: int | None = Field(default=None, ge=200, le=4000)  # 单片字符上限，不是 token 数
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)  # 相邻片段目标重叠字符量，应小于片大小


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    full_name: str
    description: str = ""
    provider: str = "github"
    document_count: int = 0
    chunk_count: int = 0
    character_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeDocumentResponse(BaseModel):
    id: UUID  # 原文件处理记录 UUID，不是某个 chunk UUID
    repo_id: UUID  # 知识所属仓库范围
    name: str
    source_type: str
    status: str  # queued/parsing/splitting/embedding/completed/failed
    content_type: str | None = None
    file_size: int = 0  # 原文件字节数
    character_count: int = 0  # 提取后文本字符数
    chunk_count: int = 0  # 原文件拆出的片段数量
    error_message: str | None = None  # 后台失败原因，成功上传不代表此字段永远为空
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class KnowledgeChunkResponse(BaseModel):
    id: UUID  # 具体文档切片 UUID
    position: int  # 面向读者的 1-based 片段序号
    title: str
    content: str  # 实际可查看的切片正文
    character_count: int  # 当前片段字符数
    enabled: bool = True  # 返回的片段可用标记


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)  # 原始自然语言问题
    top_k: int = Field(default=5, ge=1, le=10)  # 本次请求希望使用的来源数量上限
    source_types: list[str] = Field(default_factory=list, max_length=20)  # 检索来源筛选；空列表按服务默认策略
    metadata_filters: dict[str, Any] = Field(default_factory=dict)  # 标签、路径、时间等额外范围条件


class RAGSourceResponse(BaseModel):
    citation: int  # 本次来源列表的引用编号，对应回答 [1] 等
    document_id: str  # 检索命中的切片身份
    source_id: str  # 原始文件/业务来源身份，多片可以相同
    source_type: str  # 原文件、Issue、PR 评论等来源类型
    title: str
    snippet: str  # 实际送给回答或用于展示的短片段
    score: float  # 当前排序分，不是事实正确率
    metadata: dict[str, Any] = Field(default_factory=dict)  # 章节、版本、行页等来源定位
    retrieval: dict[str, Any] = Field(default_factory=dict)  # 召回分支、重排、扩展与门控轨迹


class RAGSearchResponse(BaseModel):
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestRequest(BaseModel):
    query: str = Field(min_length=1, max_length=250)  # 本次只测召回的查询，不请求生成答案
    top_k: int | None = Field(default=None, ge=1, le=20)  # 未指定时沿用知识库值
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None  # None 沿用配置，False 明确关闭
    score_threshold_enabled: bool | None = None  # None 沿用配置，False 明确关闭阈值
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestResponse(BaseModel):
    id: UUID
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RetrievalTestHistoryResponse(BaseModel):
    id: UUID
    query: str
    result_count: int
    took_ms: int
    created_at: datetime | None = None


class ModelCatalogResponse(BaseModel):
    embedding_models: list[dict[str, Any]]
    rerank_models: list[dict[str, Any]]
    deepseek: dict[str, Any]


class RAGAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSourceResponse]  # 与本次答案编号绑定的来源列表
    retrieval_count: int  # 实际保留下来的来源数量
    generation_mode: Literal["llm", "extractive", "no_evidence"]  # 真实模型、证据摘录或无证据返回
    model: str | None = None  # 实际生成模型；非模型路径可以为空
    answer_gate: dict[str, Any] = Field(default_factory=dict)  # 允许回答、资料不足、冲突或待澄清
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 本次真正生效的设置与流水线信息
    took_ms: int  # 整次问答耗时，单位毫秒


class RAGStatusResponse(BaseModel):
    repo_id: UUID
    document_count: int  # 允许 RAG 来源中的切片总数
    source_count: int  # 独立原始来源数，与切片数不同
    source_types: dict[str, int] = Field(default_factory=dict)  # 按来源类型分组的切片计数
    embedding: dict[str, Any] = Field(default_factory=dict)  # 向量契约与提供方信息
    vector_store: dict[str, Any] = Field(default_factory=dict)  # Milvus 实际运行状态
    generation: dict[str, Any] = Field(default_factory=dict)  # 生成模型配置与回退方式
    supported_files: list[str] = Field(default_factory=list)


class RAGReindexResponse(BaseModel):
    repo_id: UUID
    indexed_documents: int
    embedding_provider: str
    embedding_model: str
