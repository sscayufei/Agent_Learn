# 日志初始化入口：统一日志格式和级别，让 API、Agent 与后台任务的输出便于串联排查。
import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
