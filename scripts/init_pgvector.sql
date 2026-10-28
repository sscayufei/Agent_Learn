-- PostgreSQL 初始化阶段启用 vector 扩展；本项目当前 RAG 向量检索由 Milvus 管理，二者不要混淆。
CREATE EXTENSION IF NOT EXISTS vector;
