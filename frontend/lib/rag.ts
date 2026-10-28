// 知识库前端契约与格式化函数：区分原文件、切片、检索候选和问答来源。
// 文档处理状态对应后台流水线；score 是检索排序信号，不是回答正确率，answer_gate 才描述是否允许生成。
export type RetrievalMethod = "vector" | "fulltext" | "hybrid";

// provider/model/dimensions 是同一向量空间的组合；前端选项需要一起更新，服务端仍会再次校验。
export type KnowledgeConfig = {
  embedding_provider: string;  // 向量提供方，与 model/dimensions 共同构成契约
  embedding_model: string;  // 文档与查询需匹配的向量模型
  embedding_dimensions: number;  // 每条向量长度，不能用补零/截断凑数
  retrieval_method: RetrievalMethod;  // vector/fulltext/hybrid 召回方式
  rerank_enabled: boolean;  // 候选召回后是否重排
  rerank_provider: string;
  rerank_model: string;
  top_k: number;  // 最终候选目标条数
  score_threshold_enabled: boolean;  // 是否过滤低于指定分数的候选
  score_threshold: number;  // 排序信号门槛，不是回答正确率
  chunk_size: number;  // 单个文档片段字符上限
  chunk_overlap: number;  // 相邻片段目标重叠字符量
};

export type RuntimeModel = {
  provider: string;
  model: string;
  available: boolean;
  fallback?: string | null;
  enabled?: boolean;
  dimensions?: number;
};

export type KnowledgeBase = {
  id: string;
  name: string;
  full_name: string;
  description: string;
  provider: string;
  document_count: number;
  chunk_count: number;
  character_count: number;
  config: KnowledgeConfig;
  runtime: { embedding: RuntimeModel; rerank: RuntimeModel };
  created_at?: string;
  updated_at?: string;
};

export type KnowledgeDocument = {
  id: string;  // 原文件 UUID，不是切片 ID
  repo_id: string;
  name: string;
  source_type: string;
  status: "queued" | "parsing" | "splitting" | "embedding" | "completed" | "failed";  // 后台处理阶段，上传返回时通常仍 queued
  content_type?: string;
  file_size: number;  // 原始字节数
  character_count: number;  // 提取文本字符数
  chunk_count: number;  // 原文件拆成几片
  error_message?: string;  // 后台处理错误，供重试排查
  created_at?: string;
  completed_at?: string;
};

export type KnowledgeChunk = {
  id: string;
  position: number;
  title: string;
  content: string;
  character_count: number;
  enabled: boolean;
};

export type CatalogModel = {
  provider: string;
  model: string;
  label: string;
  available: boolean;
  recommended: boolean;
  fallback?: string | null;
  dimensions?: number;
};

export type ModelCatalog = {
  embedding_models: CatalogModel[];
  rerank_models: CatalogModel[];
  deepseek: { embedding_available: boolean; rerank_available: boolean; note: string };
};

export type RetrievalResult = {
  citation: number;  // 本次回答资料编号
  document_id: string;  // 命中切片 UUID
  source_id: string;  // 原始来源身份
  source_type: string;
  title: string;
  snippet: string;  // 当前展示/回答片段
  score: number;  // 当前排序分
  metadata: Record<string, unknown>;  // 章节、版本和行页等回溯信息
  retrieval: Record<string, unknown>;  // 召回/重排/父扩展/门控轨迹
};

export type RetrievalResponse = {
  id: string;
  query: string;
  results: RetrievalResult[];
  result_count: number;
  took_ms: number;
  retrieval_config: KnowledgeConfig & {
    runtime?: unknown;
    fusion_method?: string;
    answer_gate?: {
      decision?: "answer" | "ask_clarification" | "insufficient_evidence" | "conflict";
      reason?: string;
      evidence_count?: number;
      top_score?: number;
    };
    pipeline?: {
      retrieval_method?: string;
      fusion_method?: string;
      stages?: Array<Record<string, unknown> & { name?: string }>;
    };
  };
  created_at?: string;
};

export type RetrievalHistory = {
  id: string;
  query: string;
  result_count: number;
  took_ms: number;
  created_at?: string;
};

export const statusLabel: Record<KnowledgeDocument["status"], string> = {
  queued: "排队中",
  parsing: "解析中",
  splitting: "分段中",
  embedding: "向量化",
  completed: "可用",
  failed: "失败"
};

export function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(value?: string) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
    new Date(value)
  );
}
