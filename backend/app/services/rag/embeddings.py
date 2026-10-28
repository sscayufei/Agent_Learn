# Embedding 适配层：统一 provider/model/dimensions 的契约，再分流到本地或兼容 API。
# 文档向量与查询向量必须处于同一个向量空间；维度相同并不意味着不同模型可以混用。
# deterministic 是可复现的演示/测试实现，不代表真实语义模型；生产配置失败不能静默换模型。
import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import settings

def _deterministic_tokens(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


# 以稳定散列把词项映射到固定维度向量，便于演示和可重复测试。
# 相同输入/维度能得到相同结果，不需要下载模型或调用外部 API。
# 它反映的是散列词面特征，不具备训练语义模型的跨表达理解能力。
# 因此测试检索/页面流程可用它，但不能据此宣称真实语义召回质量已经验证。
def deterministic_embedding(text: str, dimensions: int = 128) -> list[float]:
    # 本地演示和 Eval 的兜底实现；生产环境请替换为 LLM embedding API。
    vector = [0.0] * dimensions  # 每个维度的词项计数先为 0；处理到映射到该维的词时才增加，不是预先生成语义信息
    # 每个词项分别通过稳定哈希映射一个桶，循环累积词面特征；同输入可复现，但不同词也可能碰撞。
    for token in _deterministic_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # 把哈希前两字节变成整数后取模，得到合法维度索引；不是在向量末尾追加新维度。
        index = int.from_bytes(digest[:2], "big") % dimensions
        # 同一词重复或不同词碰撞会累加到同一个桶，反映的是散列计数而非训练得到的语义权重。
        vector[index] += 1.0
    # 用 L2 长度做归一化，避免仅因文本更长就有更大的数值；空向量用 1 作分母防止除零。
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


# 严格验证模型输出长度是否等于契约维度；不符合就拒绝，不截断也不补零凑数。
# 截断/补零只让形状看似正确，却改变了向量空间和相似度含义，会留下更难发现的数据问题。
# 返回原向量方便在提供方适配器里逐条调用，错误信息包含提供方与实际/期望维度。
def _validate_vector_dimensions(vector: list[float], dimensions: int, provider: str) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError(
            f"Embedding provider '{provider}' returned {len(vector)} dimensions; expected {dimensions}. "
            "Refusing to truncate or pad vectors."
        )
    return vector


# 将配置里的别名、大小写和连字符收敛到少量内部 provider 名称。
# 例如 openai/api/remote 都可映射到 openai_compatible，后续分发不必写多套同义判断。
# 无法识别时显式报错，不悄悄切换 deterministic，否则旧索引与新查询会进入不同空间。
def normalize_embedding_provider(provider: str) -> str:
    provider = (provider or "").strip().lower().replace("-", "_")
    aliases = {
        "api": "openai_compatible",
        "openai": "openai_compatible",
        "remote": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "sentence_transformer": "local",
        "sentence_transformers": "local",
        "local_sentence_transformers": "local",
        "local": "local",
        "qwen_local": "qwen_local",
        "dashscope": "dashscope",
        "hash": "deterministic",
        "deterministic": "deterministic",
    }
    resolved = aliases.get(provider)
    if resolved is None:
        raise ValueError(f"Unsupported embedding provider: {provider or '<empty>'}")
    return resolved


# 当前项目多个仓库共享全局 Milvus collection，必须使用同一 provider/model/dimensions 组合。
# 先归一化 provider 别名，再把请求契约与全局配置完整比较；只比较维度不够。
# 例如两个模型都输出 1024 维，也不能把它们的向量直接混在一起计算有意义的相似度。
# 知识库可以保存配置，但执行时仍必须满足共享集合契约；这不是每个知识库任意混用模型的实现。
# 切换全局向量模型需要按发布流程选择正确集合并重建，不能靠改一个下拉框兼容旧索引。
def validate_embedding_contract(provider: str, model: str, dimensions: int) -> None:
    configured = (
        normalize_embedding_provider(settings.embedding_provider),
        settings.embedding_model.strip(),
        settings.embedding_dimensions,
    )
    # 比较完整三元组，避免“都是 1024 维”就误以为不同模型生成的向量可以互相检索。
    requested = (normalize_embedding_provider(provider), model.strip(), dimensions)
    if requested != configured:
        raise ValueError(
            "Embedding contract mismatch: "
            f"requested provider={requested[0]}, model={requested[1]}, dimensions={requested[2]}; "
            f"configured provider={configured[0]}, model={configured[1]}, dimensions={configured[2]}. "
            "All repositories sharing the Milvus collection must use one vector space."
        )


def embedding_contract() -> dict[str, Any]:
    return {
        "provider": normalize_embedding_provider(settings.embedding_provider),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


def embedding_provider_name() -> str:
    return normalize_embedding_provider(settings.embedding_provider)


def _embedding_payload(texts: list[str], dimensions: int, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
    }
    if dimensions:
        payload["dimensions"] = dimensions
    return payload


# 按统一 /embeddings 协议发送批次文本，密钥和基础地址来自明确的配置回退顺序。
# 接口数据可能附 index 而不是严格按请求顺序返回，先排序再提取 embedding，维持文本/向量对应。
# 先检查返回条数，再把元素转浮点并逐条校验维度，避免半批结果或形状错误进入存储。
# HTTP 失败通过 raise_for_status 抛出，交给后台文档任务或查询入口如实报告。
# 这里不会为失败请求静默换成另一个 Embedding 提供方。
def _embed_openai_compatible(
    texts: list[str],
    dimensions: int,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    api_key = api_key or settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or settings.llm_api_key
    base_url = (base_url or settings.embedding_base_url or settings.llm_base_url).rstrip("/")
    model = model or settings.embedding_model
    if not api_key:
        raise RuntimeError("embedding API key is not configured")

    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_embedding_payload(texts, dimensions, model),
        )
        response.raise_for_status()
        # 用响应 index 对齐原始输入位置；只按服务器返回顺序 zip 可能把 A 文本与 B 向量错配。
        rows = sorted(response.json().get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in rows]
        # 返回条数必须与输入条数一致；缺一条也不能继续 zip 入库，否则正文和向量可能错配。
        if len(vectors) != len(texts):
            raise ValueError("embedding response length mismatch")
        return [
            _validate_vector_dimensions([float(value) for value in vector], dimensions, "openai_compatible")
            for vector in vectors
        ]


# 按模型名与 device 缓存已加载的 SentenceTransformer，避免每个批次都重新加载权重。
# 本地依赖是可选安装，缺失时给明确安装路径，而不是伪装成本地模型已运行。
# 缓存属于当前 Python 进程；换进程或模型参数会产生独立加载，部署时仍需规划内存与首次加载成本。
@lru_cache(maxsize=4)
def _local_sentence_transformer(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError(
            "sentence-transformers is not installed; install backend/requirements-local-embedding.txt"
        ) from exc

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


# 获取缓存的本地模型，按批量编码生成归一化向量，再统一转成 Python 浮点列表。
# Qwen3 查询场景额外使用 query prompt，文档侧与查询侧可有模型建议的非对称编码方式。
# 归一化让向量适合后续 COSINE 等比较，但仍要逐条验证实际输出维度。
# device 由配置控制，本方法不自动承诺使用 GPU；没有对应资源时由依赖报告问题。
def _embed_local(
    texts: list[str],
    dimensions: int,
    *,
    model_name: str | None = None,
    is_query: bool = False,
) -> list[list[float]]:
    resolved_model = model_name or settings.local_embedding_model
    model = _local_sentence_transformer(resolved_model, settings.local_embedding_device)
    encode_kwargs: dict[str, Any] = {}
    # 这里仅针对该模型家族的查询提示约定，不给每份文档也加 query 提示。
    if is_query and "qwen3-embedding" in resolved_model.lower():
        encode_kwargs["prompt_name"] = "query"
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        **encode_kwargs,
    )
    output: list[list[float]] = []
    for vector in vectors:
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        output.append(_validate_vector_dimensions([float(value) for value in values], dimensions, "local"))
    return output


def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    dimensions = dimensions or settings.embedding_dimensions
    return embed_knowledge_texts(
        texts,
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=dimensions,
    )


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    return embed_texts([text], dimensions=dimensions)[0]


# 知识向量生成的统一分发入口，先处理空输入，再验证共享向量契约。
# qwen_local/local -> 本地模型；dashscope/openai_compatible -> 远端兼容接口；deterministic -> 演示散列向量。
# 所有路径返回 list[list[float]]，顺序与输入文本一致，方便上游批量配对。
# is_query 区分查询与文档；未知 provider 会报错，不用默认分支偷偷改变向量语义。
# 这里不存储数据，生成成功后是否写 Milvus 仍由 vector_store/处理流水线控制。
def embed_knowledge_texts(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
    is_query: bool = False,
) -> list[list[float]]:
    clean_texts = [text or "" for text in texts]
    if not clean_texts:
        return []
    # 验证发生在真实编码之前；不兼容配置应尽早失败，避免花费模型成本后才被向量库拒绝。
    validate_embedding_contract(provider, model, dimensions)
    resolved = normalize_embedding_provider(provider)
    if resolved in {"qwen_local", "local"}:
        return _embed_local(clean_texts, dimensions, model_name=model, is_query=is_query)
    if resolved == "dashscope":
        return _embed_openai_compatible(
            clean_texts,
            dimensions,
            model=model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_embedding_base_url,
        )
    if resolved == "openai_compatible":
        return _embed_openai_compatible(clean_texts, dimensions, model=model)
    if resolved == "deterministic":
        return [deterministic_embedding(text, dimensions) for text in clean_texts]
    raise ValueError(f"Unsupported embedding provider: {resolved}")


# 把单个查询包装成一条批次，沿用相同 provider/model/dimensions，并标记 is_query=True。
# 复用同一入口能确保查询与文档契约一致，同时允许本地模型为 query 应用指定提示。
# 取结果第一条是因为请求只含一条文本；批次条数错误已在提供方路径校验。
def embed_knowledge_query(text: str, *, provider: str, model: str, dimensions: int) -> list[float]:
    return embed_knowledge_texts(
        [text],
        provider=provider,
        model=model,
        dimensions=dimensions,
        is_query=True,
    )[0]
