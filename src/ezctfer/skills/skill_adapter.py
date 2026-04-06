"""
Skill Adapter - 将 skills 转换为 langchain 工具
"""

import re

from langchain_core.tools import tool

from .skill_loader import SkillMetadata, load_all_skills, list_skill_names
from ..config.log import log_info, log_warning, log_debug


def _extract_skill_body(content: str) -> str:
    """
    去掉 SKILL.md 头部 front matter，只保留正文。
    """
    front_matter_pattern = r"^\ufeff?---\s*\r?\n.*?\r?\n---\s*\r?\n?"
    return re.sub(front_matter_pattern, "", content, count=1, flags=re.DOTALL).lstrip()


@tool
def get_skill(skill_name: str) -> str:
    """
    获取指定的 skill，返回其完整的方法论和指导原则。
    
    参数：
    - skill_name: skill 名称（必需）
    
    可用的 skills：请查看工具描述中的列表
    """
    # 加载所有 skills
    skills = load_all_skills()
    
    # 检查 skill 是否存在
    if skill_name not in skills:
        available_skills = ", ".join(list_skill_names())
        return f"错误：找不到名为 '{skill_name}' 的 skill。\n\n可用的 skills 列表：{available_skills}"
    
    skill = skills[skill_name]
    
    log_info(f"📚 使用 skill: {skill_name}")
    
    # 返回 skill 正文内容（去掉头部文档信息）
    return _extract_skill_body(skill.content)


def update_get_skill_description() -> None:
    """
    更新 get_skill 工具的描述，动态列出所有可用的 skills
    """
    skills = load_all_skills()
    skill_names = list_skill_names()
    
    # 构建技能列表描述
    skills_desc = ""
    for name in skill_names:
        skill = skills[name]
        skills_desc += f"\n  - {name}: {skill.description}"
    
    # 更新工具描述
    get_skill.description = f"""获取指定的 skill，返回其完整的方法论和指导原则。

参数：
- skill_name: skill 名称（必需）

可用的 skills：{skills_desc}

使用示例：get_skill(skill_name="pua")
"""
    
    # log_debug(f"已更新 get_skill 工具描述，包含 {len(skill_names)} 个 skills")


# 初始化时更新工具描述
_update_done = False


def init_skill_tools() -> None:
    """
    初始化 skill 工具（加载 skills 并更新描述）
    """
    global _update_done
    
    if not _update_done:
        # 预加载 skills（触发扫描）
        load_all_skills()
        
        # 更新工具描述
        update_get_skill_description()
        
        _update_done = True
        
        skill_count = len(list_skill_names())
        log_info(f"📚 Skill 工具已初始化，共 {skill_count} 个 skills 可用")


def get_skill_tool():
    """
    获取 get_skill 工具
    
    Returns:
        get_tool langchain 工具
    """
    # 确保已初始化
    if not _update_done:
        init_skill_tools()
    
    return get_skill
