# 数据库连接与请求级 Session 管理。Engine 管连接池，Session 管一次工作单元。
# get_db 通过 yield 把 Session 交给 FastAPI 依赖注入，结束时释放；业务提交点仍在路由/服务内。
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
