# 仓库令牌的加密/解密工具；Webhook 签名验证在对应路由中处理。
# 加密用于降低数据库明文泄露风险；密钥管理和部署环境隔离仍需由运行方负责。
from cryptography.fernet import Fernet, InvalidToken


def _derive_demo_key(secret: str) -> bytes:
    # Fernet 需要 urlsafe base64 编码的 32 字节密钥。
    # 这里的确定性本地开发派生方式适合课程项目，不适合生产环境。
    import base64
    import hashlib

    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def encrypt_token(token: str, secret: str) -> str:
    return Fernet(_derive_demo_key(secret)).encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted: str, secret: str) -> str | None:
    try:
        return Fernet(_derive_demo_key(secret)).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None
