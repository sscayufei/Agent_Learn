# Skill Runtime 的导出入口；调用方通过此模块取得注册中心，不必依赖内部文件布局。
from app.services.skills.registry import SkillActivation, SkillDefinition, SkillManifest, SkillRegistry, SkillRegistryError, get_skill_registry

__all__ = ["SkillActivation", "SkillDefinition", "SkillManifest", "SkillRegistry", "SkillRegistryError", "get_skill_registry"]
