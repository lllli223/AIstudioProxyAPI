"""
主要设置配置模块
包含环境变量配置、路径配置、代理配置等运行时设置
"""

import os
from typing import Optional, Dict, List
from models import Message
from asyncio import Lock

# --- 全局日志控制配置 ---
DEBUG_LOGS_ENABLED = os.environ.get('DEBUG_LOGS_ENABLED', 'false').lower() in ('true', '1', 'yes')
TRACE_LOGS_ENABLED = os.environ.get('TRACE_LOGS_ENABLED', 'false').lower() in ('true', '1', 'yes')

# --- 认证相关配置 ---
AUTO_SAVE_AUTH = os.environ.get('AUTO_SAVE_AUTH', '').lower() in ('1', 'true', 'yes')
AUTH_SAVE_TIMEOUT = int(os.environ.get('AUTH_SAVE_TIMEOUT', '30'))

# --- 路径配置 ---
AUTH_PROFILES_DIR = os.path.join(os.path.dirname(__file__), '..', 'auth_profiles')
ACTIVE_AUTH_DIR = os.path.join(AUTH_PROFILES_DIR, 'active')
SAVED_AUTH_DIR = os.path.join(AUTH_PROFILES_DIR, 'saved')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
APP_LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

last_api_messages_synced_to_page: Optional[List['Message']] = None
page_sync_cache_lock: Optional[Lock] = None

# --- 代理配置 ---
# 注意：代理配置现在在 api_utils/app.py 中动态设置，根据 STREAM_PORT 环境变量决定
NO_PROXY_ENV = os.environ.get('NO_PROXY')

def get_environment_variable(key: str, default: str = '') -> str:
    """获取环境变量值"""
    return os.environ.get(key, default)

def get_boolean_env(key: str, default: bool = False) -> bool:
    """获取布尔型环境变量"""
    value = os.environ.get(key, '').lower()
    if default:
        return value not in ('false', '0', 'no', 'off')
    else:
        return value in ('true', '1', 'yes', 'on')

def get_int_env(key: str, default: int = 0) -> int:
    """获取整型环境变量"""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default 