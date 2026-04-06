"""
Skills 模块 - 用于管理和加载 LLM Skills

该模块提供了扫描、加载和使用 skills 目录下的所有 skill 的功能。
每个 skill 应该是一个独立的目录，包含 SKILL.md 文件。
"""

from .skill_loader import (
    SkillMetadata,
    parse_skill_md,
    scan_skills_directory,
    get_skills_dir,
    load_all_skills,
    get_skill,
    list_skill_names
)

from .skill_adapter import (
    get_skill,
    get_skill_tool,
    init_skill_tools,
    update_get_skill_description
)

__all__ = [
    # skill_loader
    'SkillMetadata',
    'parse_skill_md',
    'scan_skills_directory',
    'get_skills_dir',
    'load_all_skills',
    'get_skill',
    'list_skill_names',
    # skill_adapter
    'get_skill_tool',
    'init_skill_tools',
    'update_get_skill_description',
]