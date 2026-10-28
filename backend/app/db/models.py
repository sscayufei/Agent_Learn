# 关系数据库模型总览：仓库关联 Issue/PR/CI；会话关联消息、执行轨迹与记忆；知识库关联原文和切片。
# UUID 是本系统主键，GitHub number/id 是外部标识，二者不能混用；repo_id 是多仓库查询的隔离边界。
# JSON 字段承载工具结果、元数据等可变结构；Document 保存可追溯文本，向量索引由 Milvus 服务管理。
# 阅读时先看 __tablename__、主键/外键，再沿 relationship 查看对象关联，最后看默认值与删除规则。
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 本系统用户 UUID
    name: Mapped[str] = mapped_column(String(200), default="demo-user")
    role: Mapped[str] = mapped_column(String(80), default="owner", index=True)  # 演示角色权限入口，不等于完整登录认证
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)  # 仓库访问令牌密文，非前端展示字段
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 仓库是同步、会话、知识与代码工具的共同根；sync_metadata 记录覆盖，last_sync_at 只表示最近完整成功。
class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 本系统仓库 UUID，关联各业务表
    owner: Mapped[str] = mapped_column(String(200), index=True)  # 远端命名空间或组织
    name: Mapped[str] = mapped_column(String(200), index=True)  # 仓库短名
    full_name: Mapped[str] = mapped_column(String(400), unique=True, index=True)  # owner/name 形式的唯一展示/定位名
    provider: Mapped[str] = mapped_column(String(80), default="github")  # GitHub 或兼容提供方类型
    api_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # 读取结构化数据的 API 根地址
    clone_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # Git 克隆地址，与 REST API 地址不同
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # 当前本地源码目录，可能是用户已有工作树
    checkout_mode: Mapped[str] = mapped_column(String(80), default="managed")  # 托管或用户接入模式，影响清理边界
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)  # 服务端保存的访问令牌密文
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 远端默认分支名
    github_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 远端仓库 ID，不是本系统 UUID
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 最近完整成功/成功为空的同步时间
    sync_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # 各来源覆盖、页数、错误与尝试时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    issues: Mapped[list["Issue"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(back_populates="repo", cascade="all, delete-orphan")


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo_ids: Mapped[list[str]] = mapped_column(JSON, default=list)  # 本工作区所包含仓库 UUID 字符串列表
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 创建这个聚合范围的用户
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProjectIndexState(Base):
    __tablename__ = "project_index_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)  # 每个仓库最多一份项目文档索引状态
    status: Mapped[str] = mapped_column(String(40), default="missing", index=True)  # missing/building/ready/failed 等生命周期
    fingerprint: Mapped[str] = mapped_column(String(200), default="")  # 识别项目资料是否变化的指纹
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 暂不提示建立索引的截止时间
    docs_indexed: Mapped[int] = mapped_column(Integer, default=0)  # 已收录的项目文档计数
    docs_total: Mapped[int] = mapped_column(Integer, default=0)  # 本轮发现候选文档计数
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # 用于项目概览的结构化摘要
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    repo: Mapped[Repository] = relationship()


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库，隔离不同项目
    title: Mapped[str] = mapped_column(String(300), default="默认会话")  # 会话标题，与原始消息正文分开
    status: Mapped[str] = mapped_column(String(80), default="active", index=True)  # 会话生命周期状态
    message_count: Mapped[int] = mapped_column(Integer, default=0)  # 维护的消息计数，不是 token 总量
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)


class Issue(Base):
    __tablename__ = "issues"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 内部 Issue UUID，分析接口用它定位
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 编号所属的仓库范围
    github_issue_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 远端稳定 ID，重复同步用它更新
    number: Mapped[int] = mapped_column(Integer, index=True)  # 仓库内 #编号，不是全局唯一
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)  # 原始描述，是 RAG/分诊的重要输入
    state: Mapped[str] = mapped_column(String(50), default="open")  # open/closed 等实时结构化状态
    labels: Mapped[list[str]] = mapped_column(JSON, default=list)
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    assignees: Mapped[list[str]] = mapped_column(JSON, default=list)  # 远端已有负责人，不等于 Agent 推荐结果
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 远端 Issue 创建时间
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 远端 Issue 最近更新时间
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    repo: Mapped[Repository] = relationship(back_populates="issues")


class PullRequest(Base):
    __tablename__ = "pull_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 内部 PR UUID
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    github_pr_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 远端稳定 PR ID
    number: Mapped[int] = mapped_column(Integer, index=True)  # 所属仓库内 PR 编号
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(50), default="open")
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    base_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 合入目标分支
    head_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 提出变更的分支
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 实际合并时间，为空不等于已合并
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    repo: Mapped[Repository] = relationship(back_populates="pull_requests")
    files: Mapped[list["PRFile"]] = relationship(back_populates="pr", cascade="all, delete-orphan")  # 本地同步的变更文件明细
    review_comments: Mapped[list["PRReviewComment"]] = relationship(back_populates="pr", cascade="all, delete-orphan")  # 本地同步的 Review 评论明细


class PRFile(Base):
    __tablename__ = "pr_files"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)  # 关联内部 PR 行，不是远端 #编号
    filename: Mapped[str] = mapped_column(String(500))  # 变更路径，供风险与代码定位
    status: Mapped[str] = mapped_column(String(50))  # 新增/修改/删除等远端文件变更类型
    additions: Mapped[int] = mapped_column(Integer, default=0)  # diff 中新增行数
    deletions: Mapped[int] = mapped_column(Integer, default=0)  # diff 中删除行数
    patch: Mapped[str | None] = mapped_column(Text, nullable=True)  # 远端提供的差异文本，可能缺失或截断

    pr: Mapped[PullRequest] = relationship(back_populates="files")


class PRReviewComment(Base):
    __tablename__ = "pr_review_comments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)  # 评论关联的内部 PR UUID
    github_comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 远端评论稳定 ID
    body: Mapped[str | None] = mapped_column(Text, nullable=True)  # Review 评论正文，作为外部资料处理
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # 评论所指文件
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 行内定位，可能缺失/来自原始行号
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pr: Mapped[PullRequest] = relationship(back_populates="review_comments")


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    github_run_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 远端 CI 运行 ID，下载日志用它
    name: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(80))  # 排队/进行中/完成等运行阶段
    conclusion: Mapped[str | None] = mapped_column(String(80), nullable=True)  # 成功/失败等运行结论，与阶段不同
    html_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    jobs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # Job 与 Step 的结构化执行明细
    logs_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # 失败日志文本，不执行其中命令
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    repo: Mapped[Repository] = relationship(back_populates="workflow_runs")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_type: Mapped[str] = mapped_column(String(80), index=True)  # Issue/PR/CI 等分析目标类型
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)  # 内部目标 UUID
    analysis_type: Mapped[str] = mapped_column(String(100), index=True)  # 分析任务种类
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 当时分析输入快照，便于复查
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 结构化分析结果
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 本次结果记录的模型标识
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 可检索文本片段和来源元数据；并不是把 embedding 数组直接塞在本表中。
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 一条文档切片 UUID，向量主键与它关联
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 知识所属仓库过滤边界
    source_type: Mapped[str] = mapped_column(String(80), index=True)  # 来源策略分类，决定是否可向量化
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)  # 原始对象 ID，同一来源可有多片
    title: Mapped[str] = mapped_column(String(500))  # 切片标题，通常参与编码
    content: Mapped[str] = mapped_column(Text)  # 关系库保留的文本正文
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 数据库列名 metadata，保存定位和检索策略
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBaseConfig(Base):
    __tablename__ = "knowledge_base_configs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)  # 每个仓库保存一套配置
    embedding_provider: Mapped[str] = mapped_column(String(80), default="qwen_local")  # 需与全局共享集合一致的向量提供方
    embedding_model: Mapped[str] = mapped_column(String(240), default="Qwen/Qwen3-Embedding-0.6B")  # 向量模型名称
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=1024)  # 向量长度，不是 token 数
    retrieval_method: Mapped[str] = mapped_column(String(40), default="hybrid")  # 向量、关键词或混合召回
    rerank_enabled: Mapped[bool] = mapped_column(default=True)  # 召回后是否再次排序
    rerank_provider: Mapped[str] = mapped_column(String(80), default="qwen_api")
    rerank_model: Mapped[str] = mapped_column(String(240), default="qwen3-rerank")
    top_k: Mapped[int] = mapped_column(Integer, default=3)  # 目标候选数量，过滤后可能更少
    score_threshold_enabled: Mapped[bool] = mapped_column(default=True)
    score_threshold: Mapped[float] = mapped_column(Float, default=0.8)  # 候选分门槛，不是回答准确率
    chunk_size: Mapped[int] = mapped_column(Integer, default=800)  # 切片字符上限
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)  # 相邻切片目标重叠字符量
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 一份上传原文件的生命周期记录，与拆出的多个 Document chunk 是一对多语义关系。
class KnowledgeSourceDocument(Base):
    __tablename__ = "knowledge_source_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 上传原文件身份，多个 chunk 共用 source_id
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    name: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(80), default="upload_file")
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)  # 后台解析/切片/编码的生命周期
    stored_path: Mapped[str] = mapped_column(Text)  # 服务器真实磁盘路径，不应公开给浏览器
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)  # 原文件 SHA-256，用于同仓库去重
    file_size: Mapped[int] = mapped_column(Integer, default=0)  # 上传字节数
    character_count: Mapped[int] = mapped_column(Integer, default=0)  # 提取后字符数
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)  # 切片数量
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)  # 后台失败原因
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 处理正常完成的时间


class RetrievalTestRun(Base):
    __tablename__ = "retrieval_test_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    query: Mapped[str] = mapped_column(Text)  # 当时实际测试查询
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 当时生效设置，不是永远随配置变化的引用
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 测试候选快照
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)  # 召回耗时，单位毫秒
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class CodeSymbol(Base):
    __tablename__ = "code_symbols"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    path: Mapped[str] = mapped_column(String(500), index=True)  # 当前/快照中的源码路径
    name: Mapped[str] = mapped_column(String(300), index=True)  # 类、函数等符号名称
    kind: Mapped[str] = mapped_column(String(80), default="symbol", index=True)  # 符号类型
    language: Mapped[str] = mapped_column(String(80), default="text", index=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 来源起始行号
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 来源结束行号
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    commit_sha: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)  # 分析对应提交，可能缺失
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # PR 快照范围，为空通常是主源码范围
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class CodeRelation(Base):
    __tablename__ = "code_relations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    source_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)  # 关系起点符号 UUID，轻量解析可能没有
    target_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)  # 可解析出的终点符号 UUID
    source_name: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)
    target_name: Mapped[str] = mapped_column(String(300), index=True)  # 即使没精确 ID 仍保留目标名称线索
    relation_type: Mapped[str] = mapped_column(String(80), default="calls", index=True)  # 调用等关系类别，不代表完整运行时依赖
    path: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 关系所属 PR 快照范围
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 持久的原始消息；压缩边界只改变模型活动投影，不应把这个 transcript 当作可随意删掉的缓存。
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 持久消息身份，翻页/反馈/精确回忆用它
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 隔离同仓库多段讨论
    role: Mapped[str] = mapped_column(String(40), index=True)  # user/assistant/system 等消息角色
    content: Mapped[str] = mapped_column(Text)  # 原始消息正文，压缩不在这里改写
    route: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 消息产生的业务路径
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 本消息对应调用轨迹
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 压缩边界、公开事件和诊断等扩展字段
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 封存一轮对话及其工具/事件快照，支持精确回忆与问题复查。
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 本轮用户消息 UUID
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 本轮回答消息 UUID
    route: Mapped[str] = mapped_column(String(100), default="unknown")
    transcript_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 本轮原文、工具与事件封存现场
    digest: Mapped[str] = mapped_column(Text, default="")  # 本轮摘要，不替代原始 transcript
    status: Mapped[str] = mapped_column(String(80), default="sealed", index=True)  # sealed 或带警告封存等状态
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 实际封存时间


class ThreadMemory(Base):
    __tablename__ = "thread_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")  # 压缩工作记忆，不是完整原文
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)  # 抽取的事实线索，仍可回来源核查
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 已记录的决策线索
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 尚未解决的问题
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)  # 需要继续处理的事项
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)  # 当前已提取的用户偏好
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)  # 项目背景与定位线索
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 记忆对应的来源引用
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)  # 已经合入的封存会话数量
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")  # 压缩工作记忆，不是完整原文
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)  # 抽取的事实线索，仍可回来源核查
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 已记录的决策线索
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 尚未解决的问题
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)  # 需要继续处理的事项
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)  # 当前已提取的用户偏好
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)  # 项目背景与定位线索
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 记忆对应的来源引用
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)  # 已经合入的封存会话数量
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 限定会话证据的范围
    source_type: Mapped[str] = mapped_column(String(80), index=True)  # 证据类别
    source_id: Mapped[str] = mapped_column(String(120), index=True)  # 可回查来源标识，以字符串兼容多种对象
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)  # 当前证据投影正文
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class KnowledgeGraphEdge(Base):
    __tablename__ = "knowledge_graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    from_type: Mapped[str] = mapped_column(String(80), index=True)  # 起点节点类型
    from_id: Mapped[str] = mapped_column(String(180), index=True)  # 起点对象键
    to_type: Mapped[str] = mapped_column(String(80), index=True)  # 终点节点类型
    to_id: Mapped[str] = mapped_column(String(180), index=True)  # 终点对象键
    relation: Mapped[str] = mapped_column(String(80), index=True)  # 连接含义，如引用/关联
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # 关系线索强度，不是因果证明
    source: Mapped[str] = mapped_column(String(120), default="explicit")  # 关系的生成依据类别
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class RecallEvent(Base):
    __tablename__ = "recall_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(120), index=True)  # 真正发起召回的工具
    query: Mapped[str] = mapped_column(Text, default="")  # 本次实际查询，可能是改写后的问题
    scope: Mapped[str] = mapped_column(String(120), default="project")  # 项目/会话等读取范围
    mode: Mapped[str] = mapped_column(String(80), default="hybrid")  # 混合检索或精确读取等方式
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)  # 实际返回数量
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 当时召回候选快照
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 当时关联的来源引用
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 尚未批准的长期记忆候选；人工审批状态决定它能否进入正式知识路径。
class MemoryCandidate(Base):
    __tablename__ = "memory_candidates"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id"), nullable=True, index=True)  # 这条提取来自哪轮封存会话
    kind: Mapped[str] = mapped_column(String(80), index=True)  # 事实、任务、偏好等候选类别
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)  # 待人工确认的提取正文
    status: Mapped[str] = mapped_column(String(80), default="pending", index=True)  # pending/approved/rejected 审阅状态
    source: Mapped[str] = mapped_column(String(120), default="session_sealer")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 人工批准或拒绝时间


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_message: Mapped[str] = mapped_column(Text)  # 触发本次运行的问题
    route: Mapped[str] = mapped_column(String(100))  # 实际主要业务路径
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 整轮工具执行记录
    final_answer: Mapped[str] = mapped_column(Text)  # 本次最终回答
    status: Mapped[str] = mapped_column(String(80), default="success")  # 本地执行记录状态，需结合 completion 解读
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), unique=True, index=True
    )  # 反馈绑定的真实持久消息，带唯一约束
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True, index=True)  # 可关联到当时 Agent 执行记录
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    rating: Mapped[str] = mapped_column(String(40), index=True)  # 有帮助/无帮助
    reason: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)  # 固定原因分类便于统计
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(40), default="open", index=True)  # 人工复查状态
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    notification_status: Mapped[str] = mapped_column(String(40), default="pending", index=True)  # 外部通知结果，与反馈保存成功不同
    notification_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True
    )


class AgentWorkflowRun(Base):
    __tablename__ = "agent_workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    goal: Mapped[str] = mapped_column(Text)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 工程目标最终计划快照
    observation_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # Observer 质检记录
    final_answer: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 耗时/轮数/任务计数等统计
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # 整体工作流执行状态
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentTaskRun(Base):
    __tablename__ = "agent_task_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_workflow_runs.id"), index=True)  # 关联数据库工作流 UUID
    task_id: Mapped[str] = mapped_column(String(120), index=True)  # 计划中的任务键，不是本行 UUID
    agent_name: Mapped[str] = mapped_column(String(120), index=True)
    task_type: Mapped[str] = mapped_column(String(120), index=True)
    claim_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 目标与来源边界快照
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 本次执行输入
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 专用任务结构化输出
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)  # 失败说明，不能只存成功任务
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))  # 评测显示名，不能替代基线 hash
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 进度、基线、逐题结果、指标与诊断
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 有副作用动作的待确认记录，保存目标、内容、确认人和执行结果，连接分析建议与人工授权。
class ActionDraft(Base):
    __tablename__ = "action_drafts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    draft_type: Mapped[str] = mapped_column(String(100), index=True)  # 评论/建 Issue/关闭等待执行动作类型
    target_type: Mapped[str] = mapped_column(String(100), index=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 内部目标 UUID，执行时还需转远端编号
    title: Mapped[str] = mapped_column(String(300), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    risk_level: Mapped[str] = mapped_column(String(80), default="medium", index=True)  # 风险提示，不替代审批权限
    status: Mapped[str] = mapped_column(String(80), default="pending_confirmation", index=True)  # 待确认/执行成功/失败/取消
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    confirmed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 实际确认用户
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # 远端返回结果，便于检查是否已生效
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)  # 失败现场，重试前先核对远端状态
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 操作审计独立记录请求与结果，便于追查“谁对哪个目标确认了什么”。
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 发起或确认操作的用户
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(160), index=True)  # 实际业务动作名
    target_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    target_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)  # 审计目标标识，兼容字符串形态
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # 这一次动作的结果
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 审计保存的输入
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 审计保存的执行结果或错误
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
