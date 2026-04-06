"""
Skill Loader - 扫描和加载 skills 目录下的所有 skills
"""

import os
import re
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from ..config.log import log_info, log_warning, log_debug


@dataclass
class SkillMetadata:
    """Skill 元数据"""
    name: str
    description: str
    version: str = "1.0.0"
    homepage: str = ""
    license: str = ""
    skill_path: str = ""
    content: str = ""  # 完整的 SKILL.md 内容


def parse_skill_md(file_path: str) -> Optional[SkillMetadata]:
    """
    解析 SKILL.md 文件，提取元数据
    
    Args:
        file_path: SKILL.md 文件路径
        
    Returns:
        SkillMetadata 对象，如果解析失败返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 YAML front matter (--- 之间的内容)
        front_matter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        
        if not front_matter_match:
            log_warning(f"SKILL.md 文件格式错误，缺少 front matter: {file_path}")
            return None
        
        front_matter = front_matter_match.group(1)
        
        # 解析元数据
        metadata = {}
        for line in front_matter.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip().strip('"')
        
        # 必需字段检查
        if 'name' not in metadata:
            log_warning(f"SKILL.md 缺少 name 字段: {file_path}")
            return None
        
        if 'description' not in metadata:
            log_warning(f"SKILL.md 缺少 description 字段: {file_path}")
            return None
        
        return SkillMetadata(
            name=metadata.get('name', ''),
            description=metadata.get('description', ''),
            version=metadata.get('version', '1.0.0'),
            homepage=metadata.get('homepage', ''),
            license=metadata.get('license', ''),
            skill_path=os.path.dirname(file_path),
            content=content
        )
    
    except Exception as e:
        log_warning(f"解析 SKILL.md 失败 {file_path}: {e}")
        return None


def scan_skills_directory(skills_dir: str) -> Dict[str, SkillMetadata]:
    """
    扫描 skills 目录，加载所有 skill
    
    Args:
        skills_dir: skills 目录路径
        
    Returns:
        字典，key 为 skill name，value 为 SkillMetadata
    """
    skills: Dict[str, SkillMetadata] = {}
    
    if not os.path.exists(skills_dir):
        log_warning(f"Skills 目录不存在: {skills_dir}")
        return skills
    
    log_info(f"📚 扫描 skills 目录: {skills_dir}")
    
    # 遍历 skills 目录下的所有子目录
    for entry in os.listdir(skills_dir):
        skill_path = os.path.join(skills_dir, entry)
        
        # 只处理目录
        if not os.path.isdir(skill_path):
            continue
        
        # 检查是否有 SKILL.md 文件
        skill_md_path = os.path.join(skill_path, 'SKILL.md')
        if not os.path.exists(skill_md_path):
            # log_debug(f"跳过没有 SKILL.md 的目录: {entry}")
            continue
        
        # 解析 skill
        metadata = parse_skill_md(skill_md_path)
        if metadata:
            skills[metadata.name] = metadata
            log_info(f"  ✓ 加载 skill: {metadata.name} (v{metadata.version})")
    
    # log_info(f"📚 共加载 {len(skills)} 个 skills")
    
    return skills


# 获取 skills 目录路径
def get_skills_dir() -> str:
    """
    获取 skills 目录的绝对路径
    
    Returns:
        skills 目录路径
    """
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    skills_dir = current_dir.parent / 'skills'
    return str(skills_dir)


# 全局缓存已加载的 skills
_loaded_skills: Dict[str, SkillMetadata] = {}


def load_all_skills() -> Dict[str, SkillMetadata]:
    """
    加载所有 skills（带缓存）
    
    Returns:
        字典，key 为 skill name，value 为 SkillMetadata
    """
    global _loaded_skills
    
    if not _loaded_skills:
        skills_dir = get_skills_dir()
        _loaded_skills = scan_skills_directory(skills_dir)
    
    return _loaded_skills


def get_skill(name: str) -> Optional[SkillMetadata]:
    """
    获取指定名称的 skill
    
    Args:
        name: skill 名称
        
    Returns:
        SkillMetadata 对象，如果不存在返回 None
    """
    skills = load_all_skills()
    return skills.get(name)


def list_skill_names() -> list[str]:
    """
    获取所有 skill 名称列表
    
    Returns:
        skill 名称列表
    """
    skills = load_all_skills()
    return list(skills.keys())