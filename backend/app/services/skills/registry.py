# Skill 注册中心：解析 SKILL.md 的清单与正文，建立名称/入口索引，再按显式请求或触发信号激活。
# 平时只给模型目录摘要，激活时才加载完整指令与有限引用资源，避免所有 Skill 同时挤占上下文。
# 版本、激活原因和指令摘要写进 trace，既能复现运行依据，也能发现入口或工具声明冲突。
import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillRegistryError(ValueError):
    pass


class SkillManifest(BaseModel):
    name: str  # 稳定机器名，用于显式选择与索引
    title: str  # 面向人的显示标题
    version: str  # 本份 Skill 的版本，便于复现执行依据
    description: str  # 目录摘要，帮助选择而非代替完整指令
    category: str = "general"  # 展示/组织分类
    entrypoint: str  # 专用执行工具入口，与其他 Skill 不应冲突
    tools: list[str] = Field(default_factory=list)  # 声明可使用的工具集合
    input_modes: list[str] = Field(default_factory=list)  # 适用输入方式的清单元数据
    triggers: list[str] = Field(default_factory=list)  # 自动激活时匹配的词面信号
    workflow_steps: list[str] = Field(default_factory=list)  # 声明工作步骤，供提示与轨迹核查
    output_contract: str = "freeform"  # 期望输出形态的说明
    safety_level: str = "read_only"  # 安全级别元数据，不代替实际授权检查


# 保存一个已加载 Skill 的清单、指令正文与来源路径，是目录中的定义，不代表当前已激活。
# 目录可以只向模型提供摘要，完整正文留到本次任务选中后再注入，节省上下文。
# trace_metadata 用于追踪身份/版本，prompt_summary 用于选择，两种输出服务不同用途。
class SkillDefinition(BaseModel):
    manifest: SkillManifest
    path: str
    instructions: str

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "skill_name": self.manifest.name,
            "skill_title": self.manifest.title,
            "skill_version": self.manifest.version,
            "skill_steps": self.manifest.workflow_steps,
        }

    def prompt_summary(self) -> str:
        steps = ", ".join(self.manifest.workflow_steps) if self.manifest.workflow_steps else "在 SKILL.md 中定义"
        return (
            f"- {self.manifest.name} v{self.manifest.version} -> {self.manifest.entrypoint}: "
            f"{self.manifest.description} 步骤：{steps}。安全级别：{self.manifest.safety_level}。"
        )


# 表示“这个任务为何、以何种模式激活了哪个版本的 Skill”，包含本次完整指令与受限资源。
# instruction_digest 对正文做稳定摘要，日志不必每次存完整文本也能区分版本/内容变化。
# activation_mode 区分显式、自动触发与专用入口，后续约束和界面解释会用到。
class SkillActivation(BaseModel):
    skill_name: str
    skill_title: str
    skill_version: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str
    activation_mode: str  # 显式选择、自动触发或执行入口激活
    activation_reason: str  # 本次任务为什么加载它
    instructions: str  # 本次真正注入模型的完整正文
    instruction_digest: str  # 指令正文的摘要，便于核对内容版本
    resources: list[dict[str, Any]] = Field(default_factory=list)  # 经过数量、大小与路径边界检查的引用材料

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "skill_title": self.skill_title,
            "skill_version": self.skill_version,
            "skill_steps": self.workflow_steps,
            "skill_activation_mode": self.activation_mode,
            "skill_activation_reason": self.activation_reason,
            "skill_instruction_digest": self.instruction_digest,
        }

    def prompt_block(self) -> str:
        resource_text = ""
        if self.resources:
            sections = []
            # 激活后按实际引用资源逐项组织辅助正文，空资源不产生伪内容块，也不把整个资源目录无差别加入提示。
            for resource in self.resources:
                content = str(resource.get("content") or "").strip()
                if content:
                    sections.append(f"### 资源：{resource['path']}\n{content}")
            if sections:
                resource_text = "\n\n按需加载的 Skill 资源：\n\n" + "\n\n".join(sections)
        return (
            f"## {self.skill_title} ({self.skill_name} v{self.skill_version})\n"
            f"激活方式：{self.activation_mode}；原因：{self.activation_reason}\n"
            f"允许工具：{', '.join(self.tools) or '无'}\n"
            f"输出契约：{self.output_contract}；安全级别：{self.safety_level}\n\n"
            f"{self.instructions.strip()}"
            f"{resource_text}"
        )


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._skills = self._load_skills()
        self._validate_unique_keys()
        self._by_name = {skill.manifest.name: skill for skill in self._skills}
        self._by_entrypoint = {skill.manifest.entrypoint: skill for skill in self._skills}

    def list_skills(self) -> list[SkillDefinition]:
        return sorted(self._skills, key=lambda skill: skill.manifest.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        return self._by_name.get(name)

    def skill_for_entrypoint(self, tool_name: str) -> SkillDefinition | None:
        return self._by_entrypoint.get(tool_name)

    def trace_metadata_by_tool(self) -> dict[str, dict[str, Any]]:
        return {tool_name: skill.trace_metadata() for tool_name, skill in self._by_entrypoint.items()}

    def annotate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("tool_name") or "")
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return dict(tool_call)
        return {**tool_call, **skill.trace_metadata()}

    def prompt_context(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "当前没有注册 DevFlow 技能。"
        return "\n".join(skill.prompt_summary() for skill in skills)

    def tool_description_suffix(self, tool_name: str) -> str:
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return ""
        return (
            f" 注册技能：{skill.manifest.title} v{skill.manifest.version}。"
            f"{skill.manifest.description}"
        )

    def tool_description_suffixes(self) -> dict[str, str]:
        return {tool_name: self.tool_description_suffix(tool_name) for tool_name in self._by_entrypoint}

    # 先查用户是否显式选择，找到就按显式模式激活，不再让自动分数替换用户选择。
    # 没有显式选择时，对目录触发词打分，过滤零分并按分数/名称稳定排序，最多取 max_auto_skills。
    # 自动触发是轻量词面规则，不是另一个自主语义模型；可能需要用户显式选择来消除歧义。
    # 最终通过 _activation 加载完整指令与引用资源，而不是把“目录可见”误算成实际执行。
    def activate(
        self,
        message: str,
        *,
        requested_name: str | None = None,
        max_auto_skills: int = 3,
    ) -> list[SkillActivation]:
        explicit = self._explicit_skills(message, requested_name)
        if explicit:
            return [self._activation(skill, "explicit", "用户明确指定 Skill") for skill in explicit]

        scored = [
            (self._activation_score(message, skill), skill)
            for skill in self._skills
        ]
        selected = [
            (score, skill)
            for score, skill in scored
            if score > 0
        ]
        selected.sort(key=lambda item: (-item[0], item[1].manifest.name))
        return [
            self._activation(skill, "automatic", f"消息命中 Skill 触发条件，匹配分 {score}")
            for score, skill in selected[:max_auto_skills]
        ]

    def activate_entrypoint(self, tool_name: str, *, reason: str = "专用 Agent 执行入口") -> SkillActivation | None:
        skill = self.skill_for_entrypoint(tool_name)
        if skill is None:
            return None
        return self._activation(skill, "entrypoint", reason)

    def _activation(self, skill: SkillDefinition, mode: str, reason: str) -> SkillActivation:
        instructions = skill.instructions.strip()
        return SkillActivation(
            skill_name=skill.manifest.name,
            skill_title=skill.manifest.title,
            skill_version=skill.manifest.version,
            entrypoint=skill.manifest.entrypoint,
            tools=skill.manifest.tools,
            workflow_steps=skill.manifest.workflow_steps,
            output_contract=skill.manifest.output_contract,
            safety_level=skill.manifest.safety_level,
            activation_mode=mode,
            activation_reason=reason,
            instructions=instructions,
            instruction_digest=hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16],
            resources=self._load_referenced_resources(skill),
        )

    # 支持请求字段或消息中明确的 Skill 名称/标题，优先满足用户指定。
    # 用户明确指定一个不存在的名字时抛 SkillRegistryError，不能默默切到相似 Skill。
    # 普通消息中没有 Skill 选择信号时返回空，让后续自动触发规则决定。
    # 这使“没匹配到”与“用户指定错了”有不同语义，错误提示也更可操作。
    def _explicit_skills(self, message: str, requested_name: str | None) -> list[SkillDefinition]:
        if requested_name:
            skill = self._skill_by_name_or_title(requested_name)
            if skill is None:
                raise SkillRegistryError(f"未找到用户指定的 Skill：{requested_name}")
            return [skill]

        lowered = message.lower()
        if "skill" not in lowered and "技能" not in message:
            return []
        selected = [
            skill
            for skill in self._skills
            if skill.manifest.name.lower() in lowered or skill.manifest.title.lower() in lowered
        ]
        if selected:
            return selected

        match = re.search(
            r"(?:use|using|用|使用|调用|启用)\s*([A-Za-z][A-Za-z0-9_-]+)\s*(?:skill|技能)",
            message,
            flags=re.IGNORECASE,
        )
        if match:
            raise SkillRegistryError(f"未找到用户指定的 Skill：{match.group(1)}")
        return []

    def _skill_by_name_or_title(self, value: str) -> SkillDefinition | None:
        normalized = value.strip().lower()
        return next(
            (
                skill
                for skill in self._skills
                if skill.manifest.name.lower() == normalized or skill.manifest.title.lower() == normalized
            ),
            None,
        )

    # 对英文标识使用边界匹配，减少短词作为长标识子串时误命中；其他触发词按文本包含判断。
    # 命中长度给予有限权重，避免一个超长触发词无限压过其他合理候选。
    # 返回的是排序分，不是置信概率；多项同分还会用名称保证顺序稳定。
    def _activation_score(self, message: str, skill: SkillDefinition) -> int:
        lowered = message.lower()
        score = 0  # 当前 Skill 的触发匹配分，开始还未命中任何触发词，所以为 0
        # 一个 Skill 可以声明多个触发线索，逐项归一化和匹配，再合成选择分；空触发词跳过。
        for trigger in skill.manifest.triggers:
            normalized = trigger.strip().lower()
            if not normalized:
                continue
            if re.fullmatch(r"[a-z0-9_+#.-]+", normalized):
                matched = re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", lowered)
            else:
                matched = normalized in lowered
            if matched:
                score += max(1, min(len(normalized), 8))
        return score

    # 只识别指令正文中明确引用的 references/templates/scripts/assets 相对资源。
    # 先限制引用总数，再 resolve 后确认仍在 Skill 目录内，拒绝 ../ 或链接越界。
    # 还要检查存在、单文件大小和可读文本后缀，避免一次激活读取过多/未知二进制内容。
    # 为每个资源记录路径、大小与摘要，trace 可说明本轮使用了哪些配套材料。
    # 加载 Python/脚本文件的文本不表示运行它们；执行仍需独立工具能力与授权边界。
    def _load_referenced_resources(self, skill: SkillDefinition) -> list[dict[str, Any]]:
        references = sorted(
            {
                match.group(1).strip()
                for match in re.finditer(
                    r"`((?:references|templates|scripts|assets)/[^`]+)`",
                    skill.instructions,
                )
            }
        )
        if len(references) > 8:
            raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 8 个")

        skill_root = Path(skill.path).resolve().parent
        resources: list[dict[str, Any]] = []
        text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".sh", ".ps1"}
        # 逐个解析正文引用的资源路径，先验证规范化路径仍在当前 Skill 根内，再读内容；不能直接相信 ../ 路径。
        for relative in references:
            resource_path = (skill_root / relative).resolve()
            try:
                resource_path.relative_to(skill_root)
            except ValueError as exc:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用了目录外资源：{relative}") from exc
            if not resource_path.is_file():
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源不存在：{relative}")
            if resource_path.stat().st_size > 64 * 1024:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 64 KiB：{relative}")
            content = resource_path.read_text(encoding="utf-8") if resource_path.suffix.lower() in text_suffixes else ""
            resources.append(
                {
                    "path": relative.replace("\\", "/"),
                    "content": content,
                    "size": resource_path.stat().st_size,
                    "digest": hashlib.sha256(resource_path.read_bytes()).hexdigest()[:16],
                }
            )
        return resources

    def _load_skills(self) -> list[SkillDefinition]:
        if not self.root.exists():
            return []
        skills: list[SkillDefinition] = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(_load_skill(skill_file))
        return skills

    # 启动注册时确保 Skill 名称与 entrypoint 互不冲突，避免同一个执行入口随机选了另一份说明。
    # 尽早抛出配置错误，比等到某次对话已经执行错误 Agent 后再排查更容易。
    # 这只验证目录映射唯一，不验证 Skill 正文是否一定正确或工具执行是否符合全部要求。
    def _validate_unique_keys(self) -> None:
        seen_names: set[str] = set()
        seen_entrypoints: set[str] = set()
        # 启动时逐个登记名称和入口，集合一旦发现重复就报错，避免运行时随加载顺序选错 Skill。
        for skill in self._skills:
            if skill.manifest.name in seen_names:
                raise SkillRegistryError(f"Skill 名称重复：{skill.manifest.name}")
            if skill.manifest.entrypoint in seen_entrypoints:
                raise SkillRegistryError(f"Skill entrypoint 重复：{skill.manifest.entrypoint}")
            seen_names.add(skill.manifest.name)
            seen_entrypoints.add(skill.manifest.entrypoint)


# 从 SKILL.md 读取清单和正文，并用清单模型规范化结构，再保留来源路径供后续资源解析。
# 清单负责名字、入口、触发条件等元数据；正文负责具体任务指令，两者不能混读。
# 读取的是本项目配置的 Skill 文件，不通过任意模型参数临时导入远端指令。
def _load_skill(path: Path) -> SkillDefinition:
    text = path.read_text(encoding="utf-8")
    manifest_data, instructions = _split_frontmatter(text, path)
    manifest = SkillManifest.model_validate(manifest_data)
    return SkillDefinition(manifest=manifest, path=str(path), instructions=instructions)


# 按文件前部的 frontmatter 边界分离结构化清单与指令正文，再验证是否符合预期形状。
# 清单缺失/边界错误应报告具体文件，不能把整篇 Markdown 当成某个字段继续运行。
# 正文内容后续真实进入模型提示，教学注释不能为了说明而修改 Skill 指令文本。
def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"{path} 必须以 frontmatter 开头")

    frontmatter_lines: list[str] = []
    # 尚未找到 frontmatter 的关闭分隔行；用 None 区分“没找到”与合法的第 0 个位置。
    body_start = None
    # 跳过开头分隔符后逐行找关闭符，找到后剩余部分才是执行正文，避免把元数据与指令混在一起。
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = index + 1
            break
        frontmatter_lines.append(line)
    if body_start is None:
        raise SkillRegistryError(f"{path} 缺少 frontmatter 结束标记")

    return _parse_frontmatter(frontmatter_lines), "\n".join(lines[body_start:]).strip()


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    # 当前清单解析器逐行识别键值与列表，current_list_key 记住列表归属；它不是支持所有 YAML 语法的通用解析器。
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current_list_key is None:
                raise SkillRegistryError("列表项缺少父级 key")
            data.setdefault(current_list_key, [])
            if not isinstance(data[current_list_key], list):
                raise SkillRegistryError(f"{current_list_key} 不是列表")
            data[current_list_key].append(_parse_value(stripped[2:]))
            continue
        if ":" not in raw_line:
            raise SkillRegistryError(f"frontmatter 行无效：{raw_line}")
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillRegistryError("frontmatter key 不能为空")
        if value == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_value(value)
            current_list_key = None
    return data


def _parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    root = Path(__file__).resolve().parents[2] / "skills"
    return SkillRegistry(root)
