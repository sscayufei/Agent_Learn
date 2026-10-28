# 测试阅读入口：验证召回候选经过重排后的顺序与相关性线索；重排只能处理已有候选，不能代替召回覆盖。
# 阅读方法：先看准备的数据/替身，再看调用入口，最后看 assert 锁定的行为；本文件断言保持原样。
from app.services.rag.rerank import rerank_documents


def test_rerank_prefers_exact_path_and_query_terms() -> None:
    candidates = [
        {
            "id": "general",
            "title": "README setup",
            "snippet": "General local development notes",
            "score": 0.8,
            "vector_score": 0.8,
            "keyword_score": 0.1,
            "metadata": {},
        },
        {
            "id": "auth",
            "title": "backend/app/auth/middleware.py#1",
            "snippet": "validate token and refresh session when login expires",
            "score": 0.5,
            "vector_score": 0.5,
            "keyword_score": 1.0,
            "metadata": {"path": "backend/app/auth/middleware.py"},
        },
    ]

    results = rerank_documents("login token auth middleware", candidates, limit=2)

    assert results[0]["id"] == "auth"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert "rank_reason" in results[0]
