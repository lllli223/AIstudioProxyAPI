import asyncio
import multiprocessing
import random
import time
import json
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Tuple, Callable, Set
import os
import traceback
from contextlib import asynccontextmanager
import sys
import platform
import logging
import logging.handlers
import socket # 保留 socket 以便在 __main__ 中进行简单的直接运行提示
from asyncio import Queue, Lock, Future, Task, Event

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette import status
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from playwright.async_api import Page as AsyncPage, Browser as AsyncBrowser, Playwright as AsyncPlaywright, Error as PlaywrightAsyncError, expect as expect_async, BrowserContext as AsyncBrowserContext, Locator, TimeoutError
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import uuid
import datetime
import aiohttp
import stream
import queue


# --- stream queue ---
STREAM_QUEUE:Optional[multiprocessing.Queue] = None
STREAM_PROCESS = None

STREAM_TIMEOUT_LOG_STATE = {
    "consecutive_timeouts": 0,
    "last_error_log_time": 0.0, # 使用 time.monotonic()
    "suppress_until_time": 0.0, # 使用 time.monotonic()
    "max_initial_errors": 3,
    "warning_interval_after_suppress": 60.0, # seconds
    "suppress_duration_after_initial_burst": 300.0, # seconds
}

# --- 全局添加标记常量 ---
USER_INPUT_START_MARKER_SERVER = "__USER_INPUT_START__"
USER_INPUT_END_MARKER_SERVER = "__USER_INPUT_END__"

# --- 全局日志控制配置 ---
DEBUG_LOGS_ENABLED = os.environ.get('DEBUG_LOGS_ENABLED', 'false').lower() in ('true', '1', 'yes')
TRACE_LOGS_ENABLED = os.environ.get('TRACE_LOGS_ENABLED', 'false').lower() in ('true', '1', 'yes')

# --- Configuration ---
AI_STUDIO_URL_PATTERN = 'aistudio.google.com/'
RESPONSE_COMPLETION_TIMEOUT = 300000 # 5 minutes total timeout (in ms)
INITIAL_WAIT_MS_BEFORE_POLLING = 500 # ms, initial wait before polling for response completion
POLLING_INTERVAL = 300 # ms
POLLING_INTERVAL_STREAM = 180 # ms
SILENCE_TIMEOUT_MS = 40000 # ms
POST_SPINNER_CHECK_DELAY_MS = 500
FINAL_STATE_CHECK_TIMEOUT_MS = 1500
POST_COMPLETION_BUFFER = 700
CLEAR_CHAT_VERIFY_TIMEOUT_MS = 5000
CLEAR_CHAT_VERIFY_INTERVAL_MS = 400
CLICK_TIMEOUT_MS = 5000
CLIPBOARD_READ_TIMEOUT_MS = 5000
PSEUDO_STREAM_DELAY = 0.01
EDIT_MESSAGE_BUTTON_SELECTOR = 'ms-chat-turn:last-child .actions-container button.toggle-edit-button'
MESSAGE_TEXTAREA_SELECTOR = 'ms-chat-turn:last-child ms-text-chunk ms-autosize-textarea'
FINISH_EDIT_BUTTON_SELECTOR = 'ms-chat-turn:last-child .actions-container button.toggle-edit-button[aria-label="Stop editing"]'

AUTH_PROFILES_DIR = os.path.join(os.path.dirname(__file__), 'auth_profiles')
ACTIVE_AUTH_DIR = os.path.join(AUTH_PROFILES_DIR, 'active')
SAVED_AUTH_DIR = os.path.join(AUTH_PROFILES_DIR, 'saved')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
APP_LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

# --- 全局代理设置 ---

PROXY_SERVER_ENV = "http://127.0.0.1:3120/"
STREAM_PROXY_SERVER_ENV = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')

NO_PROXY_ENV = os.environ.get('NO_PROXY')
AUTO_SAVE_AUTH = os.environ.get('AUTO_SAVE_AUTH', '').lower() in ('1', 'true', 'yes')
AUTH_SAVE_TIMEOUT = int(os.environ.get('AUTH_SAVE_TIMEOUT', '30'))

PLAYWRIGHT_PROXY_SETTINGS: Optional[Dict[str, str]] = None
if PROXY_SERVER_ENV:
    PLAYWRIGHT_PROXY_SETTINGS = {'server': PROXY_SERVER_ENV}
    if NO_PROXY_ENV:
        PLAYWRIGHT_PROXY_SETTINGS['bypass'] = NO_PROXY_ENV.replace(',', ';')

# --- Constants ---
MODEL_NAME = 'AI-Studio_Camoufox-Proxy'
CHAT_COMPLETION_ID_PREFIX = 'chatcmpl-'
MODELS_ENDPOINT_URL_CONTAINS = "MakerSuiteService/ListModels"
DEFAULT_FALLBACK_MODEL_ID = "no model list"

# --- Selectors ---
PROMPT_TEXTAREA_SELECTOR = 'ms-prompt-input-wrapper ms-autosize-textarea textarea'
SYSTEM_INSTRUCTIONS_BUTTON_SELECTOR = 'button[data-test-si][aria-label="System instructions"]'
SYSTEM_INSTRUCTIONS_TEXTAREA_SELECTOR = 'textarea[placeholder="Optional tone and style instructions for the model"].toolbar-expand-textarea'
INPUT_SELECTOR = PROMPT_TEXTAREA_SELECTOR
INPUT_SELECTOR2 = PROMPT_TEXTAREA_SELECTOR
SUBMIT_BUTTON_SELECTOR = 'button[aria-label="Run"].run-button'
RESPONSE_CONTAINER_SELECTOR = 'ms-chat-turn .chat-turn-container.model'
RESPONSE_TEXT_SELECTOR = 'ms-cmark-node.cmark-node'
LOADING_SPINNER_SELECTOR = 'button[aria-label="Run"].run-button svg .stoppable-spinner'
OVERLAY_SELECTOR = 'div.cdk-overlay-backdrop'
WAIT_FOR_ELEMENT_TIMEOUT_MS = 10000 # Timeout for waiting for elements like overlays
ERROR_TOAST_SELECTOR = 'div.toast.warning, div.toast.error'
CLEAR_CHAT_BUTTON_SELECTOR = 'button[data-test-clear="outside"][aria-label="Clear chat"]'
CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR = 'button.mdc-button:has-text("Continue")'
MORE_OPTIONS_BUTTON_SELECTOR = 'div.actions-container div ms-chat-turn-options div > button'
COPY_MARKDOWN_BUTTON_SELECTOR = 'button.mat-mdc-menu-item:nth-child(4)'
COPY_MARKDOWN_BUTTON_SELECTOR_ALT = 'div[role="menu"] button:has-text("Copy Markdown")'
MAX_OUTPUT_TOKENS_SELECTOR = 'input[aria-label="Maximum output tokens"]'
STOP_SEQUENCE_INPUT_SELECTOR = 'input[aria-label="Add stop token"]'
MAT_CHIP_REMOVE_BUTTON_SELECTOR = 'mat-chip-set mat-chip-row button[aria-label*="Remove"]'
TOP_P_INPUT_SELECTOR = 'div.settings-item-column:has(h3:text-is("Top P")) input[type="number"].slider-input'
TEMPERATURE_INPUT_SELECTOR = 'div[data-test-id="temperatureSliderContainer"] input[type="number"].slider-input'


# --- Global State ---
playwright_manager: Optional[AsyncPlaywright] = None
browser_instance: Optional[AsyncBrowser] = None
page_instance: Optional[AsyncPage] = None
is_playwright_ready = False
is_browser_connected = False
is_page_ready = False
is_initializing = False

global_model_list_raw_json: Optional[List[Any]] = None
parsed_model_list: List[Dict[str, Any]] = []
model_list_fetch_event = asyncio.Event()

current_ai_studio_model_id: Optional[str] = None
model_switching_lock: Optional[Lock] = None

excluded_model_ids: Set[str] = set()
EXCLUDED_MODELS_FILENAME = "excluded_models.txt"

request_queue: Optional[Queue] = None
processing_lock: Optional[Lock] = None
worker_task: Optional[Task] = None

page_params_cache: Dict[str, Any] = {}
params_cache_lock: Optional[Lock] = None

# 新增：用于缓存上次成功同步到页面的API消息列表及其锁
last_api_messages_synced_to_page: Optional[List[Message]] = None
page_sync_cache_lock: Optional[Lock] = None

logger = logging.getLogger("AIStudioProxyServer")
log_ws_manager = None

# --- StreamToLogger, WebSocketConnectionManager, WebSocketLogHandler ---
class StreamToLogger:
    def __init__(self, logger_instance, log_level=logging.INFO):
        self.logger = logger_instance
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        try:
            temp_linebuf = self.linebuf + buf
            self.linebuf = ''
            for line in temp_linebuf.splitlines(True):
                if line.endswith(('\n', '\r')):
                    self.logger.log(self.log_level, line.rstrip())
                else:
                    self.linebuf += line
        except Exception as e:
            print(f"StreamToLogger 错误: {e}", file=sys.__stderr__)

    def flush(self):
        try:
            if self.linebuf != '':
                self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''
        except Exception as e:
            print(f"StreamToLogger Flush 错误: {e}", file=sys.__stderr__)

    def isatty(self):
        return False

class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket 日志客户端已连接: {client_id}")
        try:
            await websocket.send_text(json.dumps({
                "type": "connection_status",
                "status": "connected",
                "message": "已连接到实时日志流。",
                "timestamp": datetime.datetime.now().isoformat()
            }))
        except Exception as e:
            logger.warning(f"向 WebSocket 客户端 {client_id} 发送欢迎消息失败: {e}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket 日志客户端已断开: {client_id}")

    async def broadcast(self, message: str):
        if not self.active_connections:
            return
        disconnected_clients = []
        active_conns_copy = list(self.active_connections.items())
        for client_id, connection in active_conns_copy:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                logger.info(f"[WS Broadcast] 客户端 {client_id} 在广播期间断开连接。")
                disconnected_clients.append(client_id)
            except RuntimeError as e:
                 if "Connection is closed" in str(e):
                     logger.info(f"[WS Broadcast] 客户端 {client_id} 的连接已关闭。")
                     disconnected_clients.append(client_id)
                 else:
                     logger.error(f"广播到 WebSocket {client_id} 时发生运行时错误: {e}")
                     disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"广播到 WebSocket {client_id} 时发生未知错误: {e}")
                disconnected_clients.append(client_id)
        if disconnected_clients:
             for client_id_to_remove in disconnected_clients:
                 self.disconnect(client_id_to_remove)

class WebSocketLogHandler(logging.Handler):
    def __init__(self, manager: WebSocketConnectionManager):
        super().__init__()
        self.manager = manager
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def emit(self, record: logging.LogRecord):
        if self.manager and self.manager.active_connections:
            try:
                log_entry_str = self.format(record)
                try:
                     current_loop = asyncio.get_running_loop()
                     current_loop.create_task(self.manager.broadcast(log_entry_str))
                except RuntimeError:
                     pass
            except Exception as e:
                print(f"WebSocketLogHandler 错误: 广播日志失败 - {e}", file=sys.__stderr__)

# --- 日志设置函数 ---
def setup_server_logging(log_level_name: str = "INFO", redirect_print_str: str = "false"):
    global logger, log_ws_manager
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    redirect_print = redirect_print_str.lower() in ('true', '1', 'yes')
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ACTIVE_AUTH_DIR, exist_ok=True)
    os.makedirs(SAVED_AUTH_DIR, exist_ok=True)
    file_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(log_level)
    logger.propagate = False
    if os.path.exists(APP_LOG_FILE_PATH):
        try:
            os.remove(APP_LOG_FILE_PATH)
        except OSError as e:
            print(f"警告 (setup_server_logging): 尝试移除旧的 app.log 文件 '{APP_LOG_FILE_PATH}' 失败: {e}。将依赖 mode='w' 进行截断。", file=sys.__stderr__)
    file_handler = logging.handlers.RotatingFileHandler(
        APP_LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8', mode='w'
    )
    file_handler.setFormatter(file_log_formatter)
    logger.addHandler(file_handler)
    if log_ws_manager is None:
        print("严重警告 (setup_server_logging): log_ws_manager 未初始化！WebSocket 日志功能将不可用。", file=sys.__stderr__)
    else:
        ws_handler = WebSocketLogHandler(log_ws_manager)
        ws_handler.setLevel(logging.INFO)
        logger.addHandler(ws_handler)
    console_server_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s [SERVER] - %(message)s')
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_server_log_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if redirect_print:
        print("--- 注意：server.py 正在将其 print 输出重定向到日志系统 (文件、WebSocket 和控制台记录器) ---", file=original_stderr)
        stdout_redirect_logger = logging.getLogger("AIStudioProxyServer.stdout")
        stdout_redirect_logger.setLevel(logging.INFO)
        stdout_redirect_logger.propagate = True
        sys.stdout = StreamToLogger(stdout_redirect_logger, logging.INFO)
        stderr_redirect_logger = logging.getLogger("AIStudioProxyServer.stderr")
        stderr_redirect_logger.setLevel(logging.ERROR)
        stderr_redirect_logger.propagate = True
        sys.stderr = StreamToLogger(stderr_redirect_logger, logging.ERROR)
    else:
        print("--- server.py 的 print 输出未被重定向到日志系统 (将使用原始 stdout/stderr) ---", file=original_stderr)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR) # <-- 添加此行
    logger.info("=" * 5 + " AIStudioProxyServer 日志系统已在 lifespan 中初始化 " + "=" * 5)
    logger.info(f"日志级别设置为: {logging.getLevelName(log_level)}")
    logger.info(f"日志文件路径: {APP_LOG_FILE_PATH}")
    logger.info(f"控制台日志处理器已添加。")
    logger.info(f"Print 重定向 (由 SERVER_REDIRECT_PRINT 环境变量控制): {'启用' if redirect_print else '禁用'}")
    return original_stdout, original_stderr

def restore_original_streams(original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print("已恢复 server.py 的原始 stdout 和 stderr 流。", file=sys.__stderr__)

# --- Pydantic Models ---
class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class MessageContentItem(BaseModel):
    type: str
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContentItem], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = MODEL_NAME
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    top_p: Optional[float] = None

# --- Custom Exception ---
class ClientDisconnectedError(Exception):
    pass

# --- Helper Functions ---
def prepare_combined_prompt(messages: List[Message], req_id: str) -> Tuple[Optional[str], str]:
    logger.info(f"[{req_id}] (准备提示) 正在从 {len(messages)} 条消息准备组合提示和提取系统提示。")
    combined_parts = []
    system_prompt_content: Optional[str] = None
    first_system_message_index = -1

    # 寻找第一个系统消息并提取其内容
    for i, msg in enumerate(messages):
        if msg.role == 'system':
            if isinstance(msg.content, str) and msg.content.strip():
                system_prompt_content = msg.content.strip()
                logger.info(f"[{req_id}] (准备提示) 在索引 {i} 找到并提取系统提示: '{system_prompt_content[:80]}...'")
            else:
                # 如果系统消息内容为空或非字符串，则视为无有效系统提示，将用空字符串表示清空
                logger.info(f"[{req_id}] (准备提示) 在索引 {i} 找到系统消息，但内容为空或非字符串，系统提示将为空字符串。")
                system_prompt_content = "" # 表示清空
            first_system_message_index = i
            break # 只处理第一个系统消息

    role_map_ui = {"user": "用户", "assistant": "助手", "tool": "工具"} # "system" 角色不再在这里处理
    turn_separator = "\n---\n"

    # 组合用户和助手消息
    for i, msg in enumerate(messages):
        if i == first_system_message_index: # 跳过已处理的系统消息
            continue
        if msg.role == 'system': # 跳过所有其他系统消息
            logger.info(f"[{req_id}] (准备提示) 跳过在索引 {i} 的系统消息 (主提示组合阶段)。")
            continue

        if combined_parts: # 在角色之间添加分隔符，但不是在最开始
            combined_parts.append(turn_separator)

        role_prefix_ui = f"{role_map_ui.get(msg.role, msg.role.capitalize())}:\n"
        current_turn_parts = [role_prefix_ui]
        content_str = ""

        if isinstance(msg.content, str):
            content_str = msg.content.strip()
        elif isinstance(msg.content, list): # 处理 content 是列表的情况 (例如多模态消息)
            text_parts = []
            for item_model in msg.content:
                if isinstance(item_model, dict): # 兼容旧的 dict 格式
                    item_type = item_model.get('type')
                    if item_type == 'text' and isinstance(item_model.get('text'), str):
                        text_parts.append(item_model['text'])
                    else:
                        logger.warning(f"[{req_id}] (准备提示) 警告: 在索引 {i} 的消息中忽略非文本或未知类型的 dict content item: 类型={item_type}")
                elif isinstance(item_model, MessageContentItem): # 处理 Pydantic 模型
                    if item_model.type == 'text' and isinstance(item_model.text, str):
                        text_parts.append(item_model.text)
                    else:
                        logger.warning(f"[{req_id}] (准备提示) 警告: 在索引 {i} 的消息中忽略非文本或未知类型的 MessageContentItem: 类型={item_model.type}")
            content_str = "\n".join(text_parts).strip()
        elif msg.content is None and msg.role == 'assistant' and hasattr(msg, 'tool_calls') and msg.tool_calls:
            pass # 允许助手消息只有工具调用而无文本内容
        elif msg.content is None and msg.role == 'tool':
             logger.warning(f"[{req_id}] (准备提示) 警告: 角色 'tool' 在索引 {i} 的 content 为 None，这通常不符合预期。")
        else: # 其他意外情况
            logger.warning(f"[{req_id}] (准备提示) 警告: 角色 {msg.role} 在索引 {i} 的内容类型意外 ({type(msg.content)}) 或为 None。将尝试转换为空字符串。")
            content_str = str(msg.content or "").strip()

        if content_str:
            current_turn_parts.append(content_str)

        # 处理工具调用 (主要用于助手消息)
        if msg.role == 'assistant' and hasattr(msg, 'tool_calls') and msg.tool_calls:
            if content_str: # 如果已有文本内容，在工具调用前加换行
                current_turn_parts.append("\n")
            tool_call_visualizations = []
            if msg.tool_calls: # 确保 tool_calls 存在且非空
                for tool_call in msg.tool_calls:
                    # 兼容字典和 Pydantic 模型表示的 tool_call
                    func_name, formatted_args = None, "{}"
                    if isinstance(tool_call, dict) and tool_call.get('type') == 'function':
                        function_call_dict = tool_call.get('function')
                        if isinstance(function_call_dict, dict):
                            func_name = function_call_dict.get('name')
                            func_args_str = function_call_dict.get('arguments')
                            try:
                                parsed_args = json.loads(func_args_str if func_args_str else '{}')
                                formatted_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
                            except (json.JSONDecodeError, TypeError):
                                formatted_args = func_args_str if func_args_str is not None else "{}"
                    elif isinstance(tool_call, ToolCall) and tool_call.type == 'function':
                        func_name = tool_call.function.name
                        try:
                            parsed_args = json.loads(tool_call.function.arguments or '{}')
                            formatted_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
                        except (json.JSONDecodeError, TypeError):
                            formatted_args = tool_call.function.arguments or "{}"
                    
                    if func_name:
                        tool_call_visualizations.append(
                            f"请求调用函数: {func_name}\n参数:\n{formatted_args}"
                        )
            if tool_call_visualizations:
                current_turn_parts.append("\n".join(tool_call_visualizations))
        
        # 处理工具消息的名称和ID (如果适用)
        if msg.role == 'tool' and hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            if hasattr(msg, 'name') and msg.name and content_str: # 有名称和内容
                pass # 内容已在上面处理
            elif not content_str: # 无内容
                 logger.warning(f"[{req_id}] (准备提示) 警告: 角色 'tool' (ID: {msg.tool_call_id}, Name: {getattr(msg, 'name', 'N/A')}) 在索引 {i} 的 content 为空。")

        # 添加当前轮次到主提示 (如果非空)
        if len(current_turn_parts) > 1 or (msg.role == 'assistant' and hasattr(msg, 'tool_calls') and msg.tool_calls): # 有内容或有工具调用
            combined_parts.append("".join(current_turn_parts))
        elif not combined_parts and not current_turn_parts: # 跳过完全空的第一条消息
            logger.info(f"[{req_id}] (准备提示) 跳过角色 {msg.role} 在索引 {i} 的空消息 (且无工具调用)。")
        elif len(current_turn_parts) == 1 and not combined_parts: # 跳过只有前缀的第一条消息
             logger.info(f"[{req_id}] (准备提示) 跳过角色 {msg.role} 在索引 {i} 的空消息 (只有前缀)。")

    final_user_assistant_prompt = "".join(combined_parts)
    if final_user_assistant_prompt: # 只为用户/助手提示添加结尾换行
        final_user_assistant_prompt += "\n"

    preview_text = final_user_assistant_prompt[:300].replace('\n', '\\n')
    logger.info(f"[{req_id}] (准备提示) 组合的用户/助手提示长度: {len(final_user_assistant_prompt)}。预览: '{preview_text}...'")
    if system_prompt_content is not None: # 即使是空字符串也记录
        logger.info(f"[{req_id}] (准备提示) 提取的系统提示 (长度 {len(system_prompt_content)}): '{system_prompt_content[:80] if system_prompt_content else '[空字符串]'}'")
    else:
        logger.info(f"[{req_id}] (准备提示) 未提取到系统提示 (将为 None)。")
        
    return system_prompt_content, final_user_assistant_prompt

def validate_chat_request(messages: List[Message], req_id: str) -> Dict[str, Optional[str]]:
    if not messages:
        raise ValueError(f"[{req_id}] 无效请求: 'messages' 数组缺失或为空。")
    if not any(msg.role != 'system' for msg in messages):
        raise ValueError(f"[{req_id}] 无效请求: 未找到用户或助手消息。")
    logger.info(f"[{req_id}] (校验) 对 {len(messages)} 条消息的基本校验通过。")
    return {}

async def set_system_prompt_in_page(
    page: AsyncPage,
    system_prompt: str,
    req_id: str,
    check_client_disconnected: Callable
):
    """设置页面上的系统提示词，使用缓存避免不必要的交互。"""
    normalized_current_prompt = system_prompt if system_prompt is not None else ""

    # 此函数在 params_cache_lock 外部被调用，但其内部逻辑应在 params_cache_lock 内执行
    # 为确保安全，这里再次获取锁，或者依赖调用方已获取锁。
    # 鉴于此函数可能被多处调用，内部获取锁更安全。
    # async with params_cache_lock: # 移除此处的锁，依赖外部调用时已获取
    cached_system_prompt = page_params_cache.get("system_prompt")

    if cached_system_prompt is not None and cached_system_prompt == normalized_current_prompt:
        logger.info(f"[{req_id}] 系统提示 ('{normalized_current_prompt[:30]}...') 与缓存值一致。跳过页面交互。")
        return

    logger.info(f"[{req_id}] 尝试设置系统提示 (长度 {len(normalized_current_prompt)}): '{normalized_current_prompt[:80]}...'")

    open_button_locator = page.locator(SYSTEM_INSTRUCTIONS_BUTTON_SELECTOR).first
    textarea_locator = page.locator(SYSTEM_INSTRUCTIONS_TEXTAREA_SELECTOR).first

    try:
        check_client_disconnected("系统提示 - 开始设置前: ")

        # 检查文本框是否已可见
        # 使用 is_visible() 而不是 expect().to_be_visible() 来避免在元素不存在时抛出 AssertionError
        # is_visible() 在元素不存在时会快速返回 False (在超时内)
        if await textarea_locator.is_visible(timeout=1000): # 短暂超时检查，如果它已经在了
            logger.info(f"[{req_id}] 系统提示文本框已可见。")
        else:
            logger.info(f"[{req_id}] 系统提示文本框初始不可见或不存在，尝试点击打开按钮...")
            await expect_async(open_button_locator).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("系统提示 - 打开按钮可见后: ")
            await open_button_locator.click(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("系统提示 - 点击打开按钮后: ")
            # 点击后，再断言文本框变得可见
            await expect_async(textarea_locator).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            logger.info(f"[{req_id}] 系统提示文本框已打开/可见。")
                # 填充文本框
        await textarea_locator.fill(normalized_current_prompt, timeout=CLICK_TIMEOUT_MS)
        check_client_disconnected("系统提示 - 填充文本框后: ")
        
        # 验证填充是否成功
        filled_value = await textarea_locator.input_value(timeout=2000)
        check_client_disconnected("系统提示 - 读取填充值后: ")

        if filled_value == normalized_current_prompt:
            logger.info(f"[{req_id}] ✅ 系统提示已成功设置为 (长度 {len(normalized_current_prompt)}): '{normalized_current_prompt[:80]}...'")
            page_params_cache["system_prompt"] = normalized_current_prompt # 更新缓存
        else:
            logger.warning(f"[{req_id}] ⚠️ 系统提示填充后验证失败。期望: '{normalized_current_prompt[:80]}...', 实际: '{filled_value[:80]}...'. 缓存未更新。")
            await save_error_snapshot(f"system_prompt_fill_verify_fail_{req_id}")
            # 不更新缓存，以便下次重试或让错误更明显
            
    except TimeoutError as e: # Playwright 的操作超时通常是 TimeoutError (playwright.sync_api.Error 的子类)
        logger.error(f"[{req_id}] 操作超时: {e}")
        # 根据需要处理超时，可能需要 check_client_disconnected
        check_client_disconnected(f"系统提示 - 超时后 ({e}): ")
        raise # 重新抛出异常或进行其他错误处理
    except Exception as e:
        logger.error(f"[{req_id}] ❌ 设置系统提示时发生未知错误: {e}", exc_info=True) # exc_info=True 会记录 traceback
        check_client_disconnected(f"系统提示 - 发生未知错误后 ({e}): ")
        raise # 重新抛出异常或进行其他错误处理
        
    except ClientDisconnectedError:
        logger.info(f"[{req_id}] 客户端在设置系统提示时断开连接。")
        page_params_cache.pop("system_prompt", None) # 清除缓存，因为操作未完成
        raise
    except PlaywrightAsyncError as pw_err:
        logger.error(f"[{req_id}] ❌ 操作页面设置系统提示时发生Playwright错误: {pw_err}")
        page_params_cache.pop("system_prompt", None) # 出错时清除缓存
        await save_error_snapshot(f"system_prompt_playwright_error_{req_id}")
        # 不向上抛出，允许主流程继续，但系统提示可能未按预期设置
    except Exception as e:
        logger.exception(f"[{req_id}] ❌ 设置系统提示时发生未知错误")
        page_params_cache.pop("system_prompt", None) # 出错时清除缓存
        await save_error_snapshot(f"system_prompt_unknown_error_{req_id}")
        # 同上，不向上抛出

async def get_raw_text_content(response_element: Locator, previous_text: str, req_id: str) -> str:
    raw_text = previous_text
    try:
        await response_element.wait_for(state='attached', timeout=1000)
        pre_element = response_element.locator('pre').last
        pre_found_and_visible = False
        try:
            await pre_element.wait_for(state='visible', timeout=250)
            pre_found_and_visible = True
        except PlaywrightAsyncError: pass
        if pre_found_and_visible:
            try:
                raw_text = await pre_element.inner_text(timeout=500)
            except PlaywrightAsyncError as pre_err:
                if DEBUG_LOGS_ENABLED:
                    error_message_first_line = pre_err.message.split('\n')[0]
                    logger.warning(f"[{req_id}] 从可见的 <pre> 获取 innerText 失败: {error_message_first_line}")
                try:
                     raw_text = await response_element.inner_text(timeout=1000)
                except PlaywrightAsyncError as e_parent:
                     if DEBUG_LOGS_ENABLED:
                         logger.warning(f"[{req_id}] 在 <pre> 获取失败后，从父元素获取 inner_text 失败: {e_parent}。返回先前文本。")
                     raw_text = previous_text
        else:
            try:
                 raw_text = await response_element.inner_text(timeout=1500)
            except PlaywrightAsyncError as e_parent:
                 if DEBUG_LOGS_ENABLED:
                     logger.warning(f"[{req_id}] 从父元素获取 inner_text 失败 (无 pre 元素): {e_parent}。返回先前文本。")
                 raw_text = previous_text
        if raw_text and isinstance(raw_text, str):
            replacements = {
                "": ""
            }
            cleaned_text = raw_text
            found_junk = False
            for junk, replacement in replacements.items():
                if junk in cleaned_text:
                    cleaned_text = cleaned_text.replace(junk, replacement)
                    found_junk = True
            if found_junk:
                cleaned_text = "\n".join([line.strip() for line in cleaned_text.splitlines() if line.strip()])
                if DEBUG_LOGS_ENABLED:
                     logger.debug(f"[{req_id}] (清理) 已移除响应文本中的已知UI元素。")
                raw_text = cleaned_text
        return raw_text
    except PlaywrightAsyncError:
        return previous_text
    except Exception as e_general:
         logger.warning(f"[{req_id}] getRawTextContent 中发生意外错误: {e_general}。返回先前文本。")
         return previous_text

def generate_sse_chunk(delta: str, req_id: str, model: str) -> str:
    chunk = {
        "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}-{random.randint(100, 999)}",
        "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
    }
    return f"data: {json.dumps(chunk)}\n\n"

def generate_sse_stop_chunk(req_id: str, model: str, reason: str = "stop") -> str:
    chunk = {
        "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}-{random.randint(100, 999)}",
        "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
    }
    return f"data: {json.dumps(chunk)}\n\n"

def generate_sse_error_chunk(message: str, req_id: str, error_type: str = "server_error") -> str:
    error_payload = {"error": {"message": f"[{req_id}] {message}", "type": error_type}}
    return f"data: {json.dumps(error_payload)}\n\n"

async def _initialize_page_logic(browser: AsyncBrowser):
    logger.info("--- 初始化页面逻辑 (连接到现有浏览器) ---")
    temp_context: Optional[AsyncBrowserContext] = None
    storage_state_path_to_use: Optional[str] = None
    launch_mode = os.environ.get('LAUNCH_MODE', 'debug')
    logger.info(f"   检测到启动模式: {launch_mode}")
    loop = asyncio.get_running_loop()
    if launch_mode == 'headless' or launch_mode == 'virtual_headless':
        auth_filename = os.environ.get('ACTIVE_AUTH_JSON_PATH')
        if auth_filename:
            constructed_path = auth_filename
            if os.path.exists(constructed_path):
                storage_state_path_to_use = constructed_path
                logger.info(f"   无头模式将使用的认证文件: {constructed_path}")
            else:
                logger.error(f"{launch_mode} 模式认证文件无效或不存在: '{constructed_path}'")
                raise RuntimeError(f"{launch_mode} 模式认证文件无效: '{constructed_path}'")
        else:
            logger.error(f"{launch_mode} 模式需要 ACTIVE_AUTH_JSON_PATH 环境变量，但未设置或为空。")
            raise RuntimeError(f"{launch_mode} 模式需要 ACTIVE_AUTH_JSON_PATH。")
    elif launch_mode == 'debug':
        logger.info(f"   调试模式: 尝试从环境变量 ACTIVE_AUTH_JSON_PATH 加载认证文件...")
        auth_filepath_from_env = os.environ.get('ACTIVE_AUTH_JSON_PATH')
        if auth_filepath_from_env and os.path.exists(auth_filepath_from_env):
            storage_state_path_to_use = auth_filepath_from_env
            logger.info(f"   调试模式将使用的认证文件 (来自环境变量): {storage_state_path_to_use}")
        elif auth_filepath_from_env:
            logger.warning(f"   调试模式下环境变量 ACTIVE_AUTH_JSON_PATH 指向的文件不存在: '{auth_filepath_from_env}'。不加载认证文件。")
        else:
            logger.info("   调试模式下未通过环境变量提供认证文件。将使用浏览器当前状态。")
    elif launch_mode == "direct_debug_no_browser":
        logger.info("   direct_debug_no_browser 模式：不加载 storage_state，不进行浏览器操作。")
    else:
        logger.warning(f"   ⚠️ 警告: 未知的启动模式 '{launch_mode}'。不加载 storage_state。")
    try:
        logger.info("创建新的浏览器上下文...")
        context_options: Dict[str, Any] = {'viewport': {'width': 460, 'height': 800}}
        if storage_state_path_to_use:
            context_options['storage_state'] = storage_state_path_to_use
            logger.info(f"   (使用 storage_state='{os.path.basename(storage_state_path_to_use)}')")
        else:
            logger.info("   (不使用 storage_state)")
        if PLAYWRIGHT_PROXY_SETTINGS:
            context_options['proxy'] = PLAYWRIGHT_PROXY_SETTINGS
            logger.info(f"   (浏览器上下文将使用代理: {PLAYWRIGHT_PROXY_SETTINGS['server']})")
        else:
            logger.info("   (浏览器上下文不使用显式代理配置)")
        context_options['ignore_https_errors'] = True
        logger.info("   (浏览器上下文将忽略 HTTPS 错误)")
        temp_context = await browser.new_context(**context_options)
        found_page: Optional[AsyncPage] = None
        pages = temp_context.pages
        target_url_base = f"https://{AI_STUDIO_URL_PATTERN}"
        target_full_url = f"{target_url_base}prompts/new_chat"
        login_url_pattern = 'accounts.google.com'
        current_url = ""
        for p_iter in pages:
            try:
                page_url_to_check = p_iter.url
                if not p_iter.is_closed() and target_url_base in page_url_to_check and "/prompts/" in page_url_to_check:
                    found_page = p_iter
                    current_url = page_url_to_check
                    logger.info(f"   找到已打开的 AI Studio 页面: {current_url}")
                    if found_page:
                        logger.info(f"   为已存在的页面 {found_page.url} 添加模型列表响应监听器。")
                        found_page.on("response", _handle_model_list_response)
                    break
            except PlaywrightAsyncError as pw_err_url:
                logger.warning(f"   检查页面 URL 时出现 Playwright 错误: {pw_err_url}")
            except AttributeError as attr_err_url:
                logger.warning(f"   检查页面 URL 时出现属性错误: {attr_err_url}")
            except Exception as e_url_check:
                logger.warning(f"   检查页面 URL 时出现其他未预期错误: {e_url_check} (类型: {type(e_url_check).__name__})")
        if not found_page:
            logger.info(f"-> 未找到合适的现有页面，正在打开新页面并导航到 {target_full_url}...")
            found_page = await temp_context.new_page()
            if found_page:
                logger.info(f"   为新创建的页面添加模型列表响应监听器 (导航前)。")
                found_page.on("response", _handle_model_list_response)
            try:
                await found_page.goto(target_full_url, wait_until="domcontentloaded", timeout=90000)
                current_url = found_page.url
                logger.info(f"-> 新页面导航尝试完成。当前 URL: {current_url}")
            except Exception as new_page_nav_err:
                await save_error_snapshot("init_new_page_nav_fail")
                error_str = str(new_page_nav_err)
                if "NS_ERROR_NET_INTERRUPT" in error_str:
                    logger.error("\n" + "="*30 + " 网络导航错误提示 " + "="*30)
                    logger.error(f"❌ 导航到 '{target_full_url}' 失败，出现网络中断错误 (NS_ERROR_NET_INTERRUPT)。")
                    logger.error("   这通常表示浏览器在尝试加载页面时连接被意外断开。")
                    logger.error("   可能的原因及排查建议:")
                    logger.error("     1. 网络连接: 请检查你的本地网络连接是否稳定，并尝试在普通浏览器中访问目标网址。")
                    logger.error("     2. AI Studio 服务: 确认 aistudio.google.com 服务本身是否可用。")
                    logger.error("     3. 防火墙/代理/VPN: 检查本地防火墙、杀毒软件、代理或 VPN 设置。")
                    logger.error("     4. Camoufox 服务: 确认 launch_camoufox.py 脚本是否正常运行。")
                    logger.error("     5. 系统资源问题: 确保系统有足够的内存和 CPU 资源。")
                    logger.error("="*74 + "\n")
                raise RuntimeError(f"导航新页面失败: {new_page_nav_err}") from new_page_nav_err
        if login_url_pattern in current_url:
            if launch_mode == 'headless':
                logger.error("无头模式下检测到重定向至登录页面，认证可能已失效。请更新认证文件。")
                raise RuntimeError("无头模式认证失败，需要更新认证文件。")
            else:
                print(f"\n{'='*20} 需要操作 {'='*20}", flush=True)
                login_prompt = "   检测到可能需要登录。如果浏览器显示登录页面，请在浏览器窗口中完成 Google 登录，然后在此处按 Enter 键继续..."
                print(USER_INPUT_START_MARKER_SERVER, flush=True)
                await loop.run_in_executor(None, input, login_prompt)
                print(USER_INPUT_END_MARKER_SERVER, flush=True)
                logger.info("   用户已操作，正在检查登录状态...")
                try:
                    await found_page.wait_for_url(f"**/{AI_STUDIO_URL_PATTERN}**", timeout=180000)
                    current_url = found_page.url
                    if login_url_pattern in current_url:
                        logger.error("手动登录尝试后，页面似乎仍停留在登录页面。")
                        raise RuntimeError("手动登录尝试后仍在登录页面。")
                    logger.info("   ✅ 登录成功！请不要操作浏览器窗口，等待后续提示。")
                    print("\n" + "="*50, flush=True)
                    print("   【用户交互】需要您的输入!", flush=True)
                    save_auth_prompt = "   是否要将当前的浏览器认证状态保存到文件？ (y/N): "
                    should_save_auth_choice = ''
                    if AUTO_SAVE_AUTH and launch_mode == 'debug':
                        logger.info("   自动保存认证模式已启用，将自动保存认证状态...")
                        should_save_auth_choice = 'y'
                    else:
                        print(USER_INPUT_START_MARKER_SERVER, flush=True)
                        try:
                            auth_save_input_future = loop.run_in_executor(None, input, save_auth_prompt)
                            should_save_auth_choice = await asyncio.wait_for(auth_save_input_future, timeout=AUTH_SAVE_TIMEOUT)
                        except asyncio.TimeoutError:
                            print(f"   输入等待超时({AUTH_SAVE_TIMEOUT}秒)。默认不保存认证状态。", flush=True)
                            should_save_auth_choice = 'n'
                        finally:
                            print(USER_INPUT_END_MARKER_SERVER, flush=True)
                    if should_save_auth_choice.strip().lower() == 'y':
                        os.makedirs(SAVED_AUTH_DIR, exist_ok=True)
                        default_auth_filename = f"auth_state_{int(time.time())}.json"
                        print(USER_INPUT_START_MARKER_SERVER, flush=True)
                        filename_prompt_str = f"   请输入保存的文件名 (默认为: {default_auth_filename}): "
                        chosen_auth_filename = ''
                        try:
                            filename_input_future = loop.run_in_executor(None, input, filename_prompt_str)
                            chosen_auth_filename = await asyncio.wait_for(filename_input_future, timeout=AUTH_SAVE_TIMEOUT)
                        except asyncio.TimeoutError:
                            print(f"   输入文件名等待超时({AUTH_SAVE_TIMEOUT}秒)。将使用默认文件名: {default_auth_filename}", flush=True)
                        finally:
                            print(USER_INPUT_END_MARKER_SERVER, flush=True)
                        final_auth_filename = chosen_auth_filename.strip() or default_auth_filename
                        if not final_auth_filename.endswith(".json"):
                            final_auth_filename += ".json"
                        auth_save_path = os.path.join(SAVED_AUTH_DIR, final_auth_filename)
                        try:
                            await temp_context.storage_state(path=auth_save_path)
                            print(f"   ✅ 认证状态已成功保存到: {auth_save_path}", flush=True)
                        except Exception as save_state_err:
                            logger.error(f"   ❌ 保存认证状态失败: {save_state_err}", exc_info=True)
                            print(f"   ❌ 保存认证状态失败: {save_state_err}", flush=True)
                    else:
                        print("   好的，不保存认证状态。", flush=True)
                    print("="*50 + "\n", flush=True)
                except Exception as wait_login_err:
                    await save_error_snapshot("init_login_wait_fail")
                    logger.error(f"登录提示后未能检测到 AI Studio URL 或保存状态时出错: {wait_login_err}", exc_info=True)
                    raise RuntimeError(f"登录提示后未能检测到 AI Studio URL: {wait_login_err}") from wait_login_err
        elif target_url_base not in current_url or "/prompts/" not in current_url:
            await save_error_snapshot("init_unexpected_page")
            logger.error(f"初始导航后页面 URL 意外: {current_url}。期望包含 '{target_url_base}' 和 '/prompts/'。")
            raise RuntimeError(f"初始导航后出现意外页面: {current_url}。")
        logger.info(f"-> 确认当前位于 AI Studio 对话页面: {current_url}")
        await found_page.bring_to_front()
        try:
            input_wrapper_locator = found_page.locator('ms-prompt-input-wrapper')
            await expect_async(input_wrapper_locator).to_be_visible(timeout=35000)
            await expect_async(found_page.locator(INPUT_SELECTOR)).to_be_visible(timeout=10000)
            logger.info("-> ✅ 核心输入区域可见。")
            model_name_locator = found_page.locator('mat-select[data-test-ms-model-selector] div.model-option-content span.gmat-body-medium')
            try:
                model_name_on_page = await model_name_locator.first.inner_text(timeout=5000)
                logger.info(f"-> 🤖 页面检测到的当前模型: {model_name_on_page}")
            except PlaywrightAsyncError as e:
                logger.error(f"获取模型名称时出错 (model_name_locator): {e}")
                raise
            result_page_instance = found_page
            result_page_ready = True
            logger.info(f"✅ 页面逻辑初始化成功。")
            return result_page_instance, result_page_ready
        except Exception as input_visible_err:
             await save_error_snapshot("init_fail_input_timeout")
             logger.error(f"页面初始化失败：核心输入区域未在预期时间内变为可见。最后的 URL 是 {found_page.url}", exc_info=True)
             raise RuntimeError(f"页面初始化失败：核心输入区域未在预期时间内变为可见。最后的 URL 是 {found_page.url}") from input_visible_err
    except Exception as e_init_page:
        logger.critical(f"❌ 页面逻辑初始化期间发生严重意外错误: {e_init_page}", exc_info=True)
        if temp_context:
            try:
                logger.info(f"   尝试关闭临时的浏览器上下文 due to initialization error.")
                await temp_context.close()
                logger.info("   ✅ 临时浏览器上下文已关闭。")
            except Exception as close_err:
                 logger.warning(f"   ⚠️ 关闭临时浏览器上下文时出错: {close_err}")
        await save_error_snapshot("init_unexpected_error")
        raise RuntimeError(f"页面初始化意外错误: {e_init_page}") from e_init_page

async def _close_page_logic():
    global page_instance, is_page_ready
    logger.info("--- 运行页面逻辑关闭 --- ")
    if page_instance and not page_instance.is_closed():
        try:
            await page_instance.close()
            logger.info("   ✅ 页面已关闭")
        except PlaywrightAsyncError as pw_err:
            logger.warning(f"   ⚠️ 关闭页面时出现Playwright错误: {pw_err}")
        except asyncio.TimeoutError as timeout_err:
            logger.warning(f"   ⚠️ 关闭页面时超时: {timeout_err}")
        except Exception as other_err:
            logger.error(f"   ⚠️ 关闭页面时出现意外错误: {other_err} (类型: {type(other_err).__name__})", exc_info=True)
    page_instance = None
    is_page_ready = False
    logger.info("页面逻辑状态已重置。")
    return None, False

async def _handle_model_list_response(response: Any):
    global global_model_list_raw_json, parsed_model_list, model_list_fetch_event, logger, MODELS_ENDPOINT_URL_CONTAINS, DEBUG_LOGS_ENABLED, excluded_model_ids
    if MODELS_ENDPOINT_URL_CONTAINS in response.url and response.ok:
        logger.info(f"捕获到潜在的模型列表响应来自: {response.url} (状态: {response.status})")
        try:
            data = await response.json()
            models_array_container = None
            if isinstance(data, list) and data:
                if isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
                    logger.info("检测到三层列表结构 data[0][0] is list. models_array_container 设置为 data[0]。")
                    models_array_container = data[0]
                elif isinstance(data[0], list) and data[0] and isinstance(data[0][0], str):
                    logger.info("检测到两层列表结构 data[0][0] is str. models_array_container 设置为 data。")
                    models_array_container = data
                elif isinstance(data[0], dict):
                    logger.info("检测到根列表，元素为字典。直接使用 data 作为 models_array_container。")
                    models_array_container = data
                else:
                    logger.warning(f"未知的列表嵌套结构。data[0] 类型: {type(data[0]) if data else 'N/A'}。data[0] 预览: {str(data[0])[:200] if data else 'N/A'}")
            elif isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    models_array_container = data['data']
                elif 'models' in data and isinstance(data['models'], list):
                    models_array_container = data['models']
                else:
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (dict, list)):
                            models_array_container = value
                            logger.info(f"模型列表数据在 '{key}' 键下通过启发式搜索找到。")
                            break
                    if models_array_container is None:
                        logger.warning("在字典响应中未能自动定位模型列表数组。")
                        if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
                        return
            else:
                logger.warning(f"接收到的模型列表数据既不是列表也不是字典: {type(data)}")
                if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
                return
            if models_array_container is not None:
                new_parsed_list = []
                for entry_in_container in models_array_container:
                    model_fields_list = None
                    if isinstance(entry_in_container, dict):
                        potential_id = entry_in_container.get('id', entry_in_container.get('model_id', entry_in_container.get('modelId')))
                        if potential_id: model_fields_list = entry_in_container
                        else: model_fields_list = list(entry_in_container.values())
                    elif isinstance(entry_in_container, list):
                        model_fields_list = entry_in_container
                    else:
                        logger.debug(f"Skipping entry of unknown type: {type(entry_in_container)}")
                        continue
                    if not model_fields_list:
                        logger.debug("Skipping entry because model_fields_list is empty or None.")
                        continue
                    model_id_path_str = None
                    display_name_candidate = ""
                    description_candidate = "N/A"
                    default_max_output_tokens_val = None
                    default_top_p_val = None
                    default_temperature_val = 1.0
                    supported_max_output_tokens_val = None
                    current_model_id_for_log = "UnknownModelYet"
                    try:
                        if isinstance(model_fields_list, list):
                            if not (len(model_fields_list) > 0 and isinstance(model_fields_list[0], (str, int, float))):
                                logger.debug(f"Skipping list-based model_fields due to invalid first element: {str(model_fields_list)[:100]}")
                                continue
                            model_id_path_str = str(model_fields_list[0])
                            current_model_id_for_log = model_id_path_str.split('/')[-1] if model_id_path_str and '/' in model_id_path_str else model_id_path_str
                            display_name_candidate = str(model_fields_list[3]) if len(model_fields_list) > 3 else ""
                            description_candidate = str(model_fields_list[4]) if len(model_fields_list) > 4 else "N/A"
                            if len(model_fields_list) > 6 and model_fields_list[6] is not None:
                                try:
                                    val_int = int(model_fields_list[6])
                                    default_max_output_tokens_val = val_int
                                    supported_max_output_tokens_val = val_int
                                except (ValueError, TypeError):
                                    logger.warning(f"模型 {current_model_id_for_log}: 无法将列表索引6的值 '{model_fields_list[6]}' 解析为 max_output_tokens。")
                            if len(model_fields_list) > 9 and model_fields_list[9] is not None:
                                try:
                                    raw_top_p = float(model_fields_list[9])
                                    if not (0.0 <= raw_top_p <= 1.0):
                                        logger.warning(f"模型 {current_model_id_for_log}: 原始 top_p值 {raw_top_p} (来自列表索引9) 超出 [0,1] 范围，将裁剪。")
                                        default_top_p_val = max(0.0, min(1.0, raw_top_p))
                                    else:
                                        default_top_p_val = raw_top_p
                                except (ValueError, TypeError):
                                    logger.warning(f"模型 {current_model_id_for_log}: 无法将列表索引9的值 '{model_fields_list[9]}' 解析为 top_p。")
                        elif isinstance(model_fields_list, dict):
                            model_id_path_str = str(model_fields_list.get('id', model_fields_list.get('model_id', model_fields_list.get('modelId'))))
                            current_model_id_for_log = model_id_path_str.split('/')[-1] if model_id_path_str and '/' in model_id_path_str else model_id_path_str
                            display_name_candidate = str(model_fields_list.get('displayName', model_fields_list.get('display_name', model_fields_list.get('name', ''))))
                            description_candidate = str(model_fields_list.get('description', "N/A"))
                            mot_parsed = model_fields_list.get('maxOutputTokens', model_fields_list.get('defaultMaxOutputTokens', model_fields_list.get('outputTokenLimit')))
                            if mot_parsed is not None:
                                try:
                                    val_int = int(mot_parsed)
                                    default_max_output_tokens_val = val_int
                                    supported_max_output_tokens_val = val_int
                                except (ValueError, TypeError):
                                     logger.warning(f"模型 {current_model_id_for_log}: 无法将字典值 '{mot_parsed}' 解析为 max_output_tokens。")
                            top_p_parsed = model_fields_list.get('topP', model_fields_list.get('defaultTopP'))
                            if top_p_parsed is not None:
                                try:
                                    raw_top_p = float(top_p_parsed)
                                    if not (0.0 <= raw_top_p <= 1.0):
                                        logger.warning(f"模型 {current_model_id_for_log}: 原始 top_p值 {raw_top_p} (来自字典) 超出 [0,1] 范围，将裁剪。")
                                        default_top_p_val = max(0.0, min(1.0, raw_top_p))
                                    else:
                                        default_top_p_val = raw_top_p
                                except (ValueError, TypeError):
                                    logger.warning(f"模型 {current_model_id_for_log}: 无法将字典值 '{top_p_parsed}' 解析为 top_p。")
                            temp_parsed = model_fields_list.get('temperature', model_fields_list.get('defaultTemperature'))
                            if temp_parsed is not None:
                                try: default_temperature_val = float(temp_parsed)
                                except (ValueError, TypeError):
                                    logger.warning(f"模型 {current_model_id_for_log}: 无法将字典值 '{temp_parsed}' 解析为 temperature。")
                        else:
                            logger.debug(f"Skipping entry because model_fields_list is not list or dict: {type(model_fields_list)}")
                            continue
                    except Exception as e_parse_fields:
                        logger.error(f"解析模型字段时出错 for entry {str(entry_in_container)[:100]}: {e_parse_fields}")
                        continue
                    if model_id_path_str and model_id_path_str.lower() != "none":
                        simple_model_id_str = model_id_path_str.split('/')[-1] if '/' in model_id_path_str else model_id_path_str
                        if simple_model_id_str in excluded_model_ids:
                            logger.info(f"模型 '{simple_model_id_str}' 在排除列表 excluded_model_ids 中，已跳过。")
                            continue
                        final_display_name_str = display_name_candidate if display_name_candidate else simple_model_id_str.replace("-", " ").title()
                        model_entry_dict = {
                            "id": simple_model_id_str, "object": "model", "created": int(time.time()),
                            "owned_by": "ai_studio", "display_name": final_display_name_str,
                            "description": description_candidate, "raw_model_path": model_id_path_str,
                            "default_temperature": default_temperature_val,
                            "default_max_output_tokens": default_max_output_tokens_val,
                            "supported_max_output_tokens": supported_max_output_tokens_val,
                            "default_top_p": default_top_p_val
                        }
                        new_parsed_list.append(model_entry_dict)
                    else:
                        logger.debug(f"Skipping entry due to invalid model_id_path: {model_id_path_str} from entry {str(entry_in_container)[:100]}")
                if new_parsed_list:
                    parsed_model_list = sorted(new_parsed_list, key=lambda m: m.get('display_name', '').lower())
                    global_model_list_raw_json = json.dumps({"data": parsed_model_list, "object": "list"})
                    if DEBUG_LOGS_ENABLED:
                        log_output = f"成功解析和更新模型列表。总共解析模型数: {len(parsed_model_list)}.\n"
                        for i, item in enumerate(parsed_model_list[:min(3, len(parsed_model_list))]):
                            log_output += f"  Model {i+1}: ID={item.get('id')}, Name={item.get('display_name')}, Temp={item.get('default_temperature')}, MaxTokDef={item.get('default_max_output_tokens')}, MaxTokSup={item.get('supported_max_output_tokens')}, TopP={item.get('default_top_p')}\n"
                        logger.info(log_output)
                    if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
                elif not parsed_model_list:
                    logger.warning("解析后模型列表仍然为空。")
                    if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
            else:
                logger.warning("models_array_container 为 None，无法解析模型列表。")
                if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
        except json.JSONDecodeError as json_err:
            logger.error(f"解析模型列表JSON失败: {json_err}. 响应 (前500字): {await response.text()[:500]}")
        except Exception as e_handle_list_resp:
            logger.exception(f"处理模型列表响应时发生未知错误: {e_handle_list_resp}")
        finally:
            if not model_list_fetch_event.is_set():
                logger.info("处理模型列表响应结束，强制设置 model_list_fetch_event。")
                model_list_fetch_event.set()

async def signal_camoufox_shutdown():
    logger.info("   尝试发送关闭信号到 Camoufox 服务器 (此功能可能已由父进程处理)...")
    ws_endpoint = os.environ.get('CAMOUFOX_WS_ENDPOINT')
    if not ws_endpoint:
        logger.warning("   ⚠️ 无法发送关闭信号：未找到 CAMOUFOX_WS_ENDPOINT 环境变量。")
        return
    if not browser_instance or not browser_instance.is_connected():
        logger.warning("   ⚠️ 浏览器实例已断开或未初始化，跳过关闭信号发送。")
        return
    try:
        await asyncio.sleep(0.2)
        logger.info("   ✅ (模拟) 关闭信号已处理。")
    except Exception as e:
        logger.error(f"   ⚠️ 发送关闭信号过程中捕获异常: {e}", exc_info=True)

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app_param: FastAPI):
    global playwright_manager, browser_instance, page_instance, worker_task
    global is_playwright_ready, is_browser_connected, is_page_ready, is_initializing
    global logger, log_ws_manager, model_list_fetch_event, current_ai_studio_model_id, excluded_model_ids
    global request_queue, processing_lock, model_switching_lock, page_params_cache, params_cache_lock
    # 新增全局变量的引用
    global last_api_messages_synced_to_page, page_sync_cache_lock
    true_original_stdout, true_original_stderr = sys.stdout, sys.stderr
    global STREAM_QUEUE ,STREAM_PROCESS, PROXY_SERVER_ENV, STREAM_PROXY_SERVER_ENV, STREAM_PORT, PROXY_SERVER_ENV
    global PLAYWRIGHT_PROXY_SETTINGS
    initial_stdout_before_redirect, initial_stderr_before_redirect = sys.stdout, sys.stderr

    if log_ws_manager is None:
        log_ws_manager = WebSocketConnectionManager()
    log_level_env = os.environ.get('SERVER_LOG_LEVEL', 'INFO')
    redirect_print_env = os.environ.get('SERVER_REDIRECT_PRINT', 'false')
    initial_stdout_before_redirect, initial_stderr_before_redirect = setup_server_logging(
        log_level_name=log_level_env,
        redirect_print_str=redirect_print_env
    )

    PROXY_SERVER_ENV = "http://127.0.0.1:3120/"
    STREAM_PROXY_SERVER_ENV = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')

    STREAM_PORT = os.environ.get('STREAM_PORT')
    if STREAM_PORT == '0':
        PROXY_SERVER_ENV = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    elif STREAM_PORT is not None:
        PROXY_SERVER_ENV = f"http://127.0.0.1:{STREAM_PORT}/"

    PLAYWRIGHT_PROXY_SETTINGS = None
    if PROXY_SERVER_ENV:
        PLAYWRIGHT_PROXY_SETTINGS = {'server': PROXY_SERVER_ENV}
        if NO_PROXY_ENV:
            PLAYWRIGHT_PROXY_SETTINGS['bypass'] = NO_PROXY_ENV.replace(',', ';')

    if STREAM_PORT != '0':
        logger.info(f"STREAM 代理启动中，端口: {STREAM_PORT}")
        STREAM_QUEUE = multiprocessing.Queue()
        if STREAM_PORT is None:
            port = 3120
        else:
            port = int(STREAM_PORT)
        logger.info(f"STREAM 代理使用上游代理服务器：{STREAM_PROXY_SERVER_ENV}")
        STREAM_PROCESS = multiprocessing.Process(target=stream.start, args=(STREAM_QUEUE, port, STREAM_PROXY_SERVER_ENV))
        STREAM_PROCESS.start()
        logger.info("STREAM 代理启动完毕")
    else:
        logger.info("STREAM 代理已禁用")

    request_queue = asyncio.Queue()
    processing_lock = asyncio.Lock()
    model_switching_lock = asyncio.Lock()
    model_list_fetch_event = asyncio.Event()
    params_cache_lock = asyncio.Lock()
    # 初始化新的锁
    page_sync_cache_lock = asyncio.Lock()
    if PLAYWRIGHT_PROXY_SETTINGS:
        logger.info(f"--- 代理配置检测到 (由 server.py 的 lifespan 记录) ---")
        logger.info(f"   将使用代理服务器: {PLAYWRIGHT_PROXY_SETTINGS['server']}")
        if 'bypass' in PLAYWRIGHT_PROXY_SETTINGS:
            logger.info(f"   绕过代理的主机: {PLAYWRIGHT_PROXY_SETTINGS['bypass']}")
        logger.info(f"-----------------------")
    else:
        logger.info("--- 未检测到 HTTP_PROXY 或 HTTPS_PROXY 环境变量，不使用代理 (由 server.py 的 lifespan 记录) ---")
    load_excluded_models(EXCLUDED_MODELS_FILENAME)
    is_initializing = True
    logger.info("\n" + "="*60 + "\n          🚀 AI Studio Proxy Server (FastAPI App Lifespan) 🚀\n" + "="*60)
    logger.info(f"FastAPI 应用生命周期: 启动中...")
    try:
        logger.info(f"   启动 Playwright...")
        playwright_manager = await async_playwright().start()
        is_playwright_ready = True
        logger.info(f"   ✅ Playwright 已启动。")
        ws_endpoint = os.environ.get('CAMOUFOX_WS_ENDPOINT')
        launch_mode = os.environ.get('LAUNCH_MODE', 'unknown')
        if not ws_endpoint:
            if launch_mode == "direct_debug_no_browser":
                logger.warning("CAMOUFOX_WS_ENDPOINT 未设置，但 LAUNCH_MODE 表明不需要浏览器。跳过浏览器连接。")
                is_browser_connected = False
                is_page_ready = False
                model_list_fetch_event.set()
            else:
                logger.error("未找到 CAMOUFOX_WS_ENDPOINT 环境变量。Playwright 将无法连接到浏览器。")
                raise ValueError("CAMOUFOX_WS_ENDPOINT 环境变量缺失。")
        else:
            logger.info(f"   连接到 Camoufox 服务器 (浏览器 WebSocket 端点) 于: {ws_endpoint}")
            try:
                browser_instance = await playwright_manager.firefox.connect(ws_endpoint, timeout=30000)
                is_browser_connected = True
                logger.info(f"   ✅ 已连接到浏览器实例: 版本 {browser_instance.version}")
                temp_page_instance, temp_is_page_ready = await _initialize_page_logic(browser_instance)
                if temp_page_instance and temp_is_page_ready:
                    page_instance = temp_page_instance
                    is_page_ready = temp_is_page_ready
                    await _handle_initial_model_state_and_storage(page_instance)
                    # 页面初始化成功后，清空页面消息缓存
                    async with page_sync_cache_lock:
                        last_api_messages_synced_to_page = None
                        logger.info(f"   页面初始化成功，已清空 last_api_messages_synced_to_page。")
                else:
                    is_page_ready = False
                    if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
            except Exception as connect_err:
                logger.error(f"未能连接到 Camoufox 服务器 (浏览器) 或初始化页面失败: {connect_err}", exc_info=True)
                if launch_mode != "direct_debug_no_browser":
                    raise RuntimeError(f"未能连接到 Camoufox 或初始化页面: {connect_err}") from connect_err
                else:
                    is_browser_connected = False
                    is_page_ready = False
                    if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
        if is_page_ready and is_browser_connected and not model_list_fetch_event.is_set():
            logger.info("等待模型列表捕获 (最多等待15秒)...")
            try:
                await asyncio.wait_for(model_list_fetch_event.wait(), timeout=15.0)
                if model_list_fetch_event.is_set():
                    logger.info("模型列表事件已触发。")
                else:
                    logger.warning("模型列表事件等待后仍未设置。")
            except asyncio.TimeoutError:
                logger.warning("等待模型列表捕获超时。将使用默认或空列表。")
            finally:
                if not model_list_fetch_event.is_set():
                    model_list_fetch_event.set()
        elif not (is_page_ready and is_browser_connected):
             if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
        if (is_page_ready and is_browser_connected) or launch_mode == "direct_debug_no_browser":
             logger.info(f"   启动请求处理 Worker...")
             worker_task = asyncio.create_task(queue_worker())
             logger.info(f"   ✅ 请求处理 Worker 已启动。")
        elif launch_mode == "direct_debug_no_browser":
            logger.warning("浏览器和页面未就绪 (direct_debug_no_browser 模式)，请求处理 Worker 未启动。API 可能功能受限。")
        else:
             logger.error("页面或浏览器初始化失败，无法启动 Worker。")
             if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
             raise RuntimeError("页面或浏览器初始化失败，无法启动 Worker。")
        logger.info(f"✅ FastAPI 应用生命周期: 启动完成。服务已就绪。")
        is_initializing = False
        yield
    except Exception as startup_err:
        logger.critical(f"❌ FastAPI 应用生命周期: 启动期间发生严重错误: {startup_err}", exc_info=True)
        if not model_list_fetch_event.is_set(): model_list_fetch_event.set()
        if worker_task and not worker_task.done(): worker_task.cancel()
        if browser_instance and browser_instance.is_connected():
            try: await browser_instance.close()
            except: pass
        if playwright_manager:
            try: await playwright_manager.stop()
            except: pass
        raise RuntimeError(f"应用程序启动失败: {startup_err}") from startup_err
    finally:
        logger.info("STREAM 代理关闭中")
        STREAM_PROCESS.terminate()

        is_initializing = False
        logger.info(f"\nFastAPI 应用生命周期: 关闭中...")
        if worker_task and not worker_task.done():
             logger.info(f"   正在取消请求处理 Worker...")
             worker_task.cancel()
             try:
                 await asyncio.wait_for(worker_task, timeout=5.0)
                 logger.info(f"   ✅ 请求处理 Worker 已停止/取消。")
             except asyncio.TimeoutError: logger.warning(f"   ⚠️ Worker 等待超时。")
             except asyncio.CancelledError: logger.info(f"   ✅ 请求处理 Worker 已确认取消。")
             except Exception as wt_err: logger.error(f"   ❌ 等待 Worker 停止时出错: {wt_err}", exc_info=True)
        if page_instance and not page_instance.is_closed():
            try:
                logger.info("Lifespan 清理：移除模型列表响应监听器。")
                page_instance.remove_listener("response", _handle_model_list_response)
            except Exception as e:
                logger.debug(f"Lifespan 清理：移除监听器时发生非严重错误或监听器本不存在: {e}")
        if page_instance:
            await _close_page_logic()
        if browser_instance:
            logger.info(f"   正在关闭与浏览器实例的连接...")
            try:
                if browser_instance.is_connected():
                    await browser_instance.close()
                    logger.info(f"   ✅ 浏览器连接已关闭。")
                else: logger.info(f"   ℹ️ 浏览器先前已断开连接。")
            except Exception as close_err: logger.error(f"   ❌ 关闭浏览器连接时出错: {close_err}", exc_info=True)
            finally: browser_instance = None; is_browser_connected = False; is_page_ready = False
        if playwright_manager:
            logger.info(f"   停止 Playwright...")
            try:
                await playwright_manager.stop()
                logger.info(f"   ✅ Playwright 已停止。")
            except Exception as stop_err: logger.error(f"   ❌ 停止 Playwright 时出错: {stop_err}", exc_info=True)
            finally: playwright_manager = None; is_playwright_ready = False
        # 重置页面同步缓存
        last_api_messages_synced_to_page = None
        logger.info(f"   重置 last_api_messages_synced_to_page 缓存。")
        restore_original_streams(initial_stdout_before_redirect, initial_stderr_before_redirect)
        restore_original_streams(true_original_stdout, true_original_stderr)
        logger.info(f"✅ FastAPI 应用生命周期: 关闭完成。")

# --- 鉴权配置 ---
API_KEY = os.getenv("API_KEY") # 从环境变量获取 API Key
api_key_scheme = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(api_key_scheme)):
    """
    验证 API 密钥。
    如果环境变量 PROXY_API_KEY 未设置，则不进行验证。
    """
    if API_KEY: # 仅当配置了 API_KEY 时才强制验证
        if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
            logger.warning(f"Invalid API Key attempt. Provided scheme: {credentials.scheme if credentials else 'None'}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    # 如果 API_KEY 未设置，或者密钥有效，则返回凭据 (或任何非 None 值表示成功)
    return credentials


# --- FastAPI App 定义 ---
app = FastAPI(
    title="AI Studio Proxy Server (集成模式)",
    description="通过 Playwright与 AI Studio 交互的代理服务器。",
    version="0.6.0-integrated-auth", # 版本号更新
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    index_html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_html_path):
        # 为测试创建虚拟文件
        with open(index_html_path, "w") as f: f.write("<h1>Test Index</h1>")
        logger.warning(f"Created dummy index.html at {index_html_path}")
        # logger.error(f"index.html not found at {index_html_path}")
        # raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html_path)

@app.get("/webui.css")
async def get_css():
    css_path = os.path.join(os.path.dirname(__file__), "webui.css")
    if not os.path.exists(css_path):
        with open(css_path, "w") as f: f.write("/* Test CSS */")
        logger.warning(f"Created dummy webui.css at {css_path}")
        # logger.error(f"webui.css not found at {css_path}")
        # raise HTTPException(status_code=404, detail="webui.css not found")
    return FileResponse(css_path, media_type="text/css")

@app.get("/webui.js")
async def get_js():
    js_path = os.path.join(os.path.dirname(__file__), "webui.js")
    if not os.path.exists(js_path):
        with open(js_path, "w") as f: f.write("// Test JS")
        logger.warning(f"Created dummy webui.js at {js_path}")
        # logger.error(f"webui.js not found at {js_path}")
        # raise HTTPException(status_code=404, detail="webui.js not found")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/api/info")
async def get_api_info(request: Request):
    server_port = request.url.port
    if not server_port and hasattr(request.app.state, 'server_port'):
        server_port = request.app.state.server_port
    if not server_port:
        server_port = os.environ.get('SERVER_PORT_INFO', '8000') # SERVER_PORT_INFO 通常用于运行时的实际端口
    
    # 尝试从请求头中获取 host，这对于反向代理后的场景更准确
    host_header = request.headers.get('host')
    if host_header:
        # 如果 host 头包含端口，则使用它；否则，附加 server_port
        if ':' in host_header:
            host_with_port = host_header
        else:
            # 对于标准端口（80 for http, 443 for https），host 头可能不包含端口
            # 但为了API URL的明确性，我们可能仍想显示配置的端口或默认端口
            # 这里的逻辑可以根据具体需求调整
            scheme_from_header = request.headers.get('x-forwarded-proto', 'http')
            if (scheme_from_header == 'http' and server_port == '80') or \
               (scheme_from_header == 'https' and server_port == '443'):
                host_with_port = host_header # 标准端口，URL中通常不显式写出
            else:
                host_with_port = f"{host_header}:{server_port}"
    else:
        host_with_port = f"127.0.0.1:{server_port}" # 回退到本地地址

    scheme = request.headers.get('x-forwarded-proto', 'http')
    base_url = f"{scheme}://{host_with_port}"
    api_base = f"{base_url}/v1"
    effective_model_name = current_ai_studio_model_id if current_ai_studio_model_id else MODEL_NAME
    
    api_key_required = bool(API_KEY) # API Key是否已配置
    message = "API Key is required." if api_key_required else "API Key is not required (PROXY_API_KEY not set)."

    return JSONResponse(content={
        "model_name": effective_model_name,
        "api_base_url": api_base,
        "server_base_url": base_url,
        "api_key_required": api_key_required,
        "message": message
    })

@app.get("/health")
async def health_check():
    is_worker_running = bool(worker_task and not worker_task.done()) if worker_task else False # 确保 worker_task 存在
    launch_mode = os.environ.get('LAUNCH_MODE', 'unknown')
    browser_page_critical = launch_mode != "direct_debug_no_browser"
    
    core_ready_conditions = [not is_initializing, is_playwright_ready]
    if browser_page_critical:
        core_ready_conditions.extend([is_browser_connected, is_page_ready])
    
    is_core_ready = all(core_ready_conditions)
    status_val = "OK" if is_core_ready and is_worker_running else "Error"
    q_size = request_queue.qsize() if request_queue else -1
    
    status_message_parts = []
    if is_initializing: status_message_parts.append("初始化进行中")
    if not is_playwright_ready: status_message_parts.append("Playwright 未就绪")
    if browser_page_critical:
        if not is_browser_connected: status_message_parts.append("浏览器未连接")
        if not is_page_ready: status_message_parts.append("页面未就绪")
    if not is_worker_running: status_message_parts.append("Worker 未运行")
    
    status_payload = { # Renamed to avoid conflict with starlette.status
        "status": status_val,
        "message": "",
        "details": {
            "playwrightReady": is_playwright_ready,
            "browserConnected": is_browser_connected,
            "pageReady": is_page_ready,
            "initializing": is_initializing,
            "workerRunning": is_worker_running,
            "queueLength": q_size,
            "launchMode": launch_mode,
            "browserAndPageCritical": browser_page_critical
        }
    }
    if status_val == "OK":
        status_payload["message"] = f"服务运行中;队列长度: {q_size}。"
        return JSONResponse(content=status_payload, status_code=200)
    else:
        status_payload["message"] = f"服务不可用;问题: {(', '.join(status_message_parts) if status_message_parts else '未知原因')}. 队列长度: {q_size}."
        return JSONResponse(content=status_payload, status_code=503)

@app.get("/v1/models", dependencies=[Depends(verify_api_key)]) # 应用鉴权
async def list_models():
    logger.info("[API] 收到 /v1/models 请求。")
    
    # 确保 page_instance 存在且未关闭
    if not model_list_fetch_event.is_set() and page_instance and not page_instance.is_closed():
        logger.info("/v1/models: 模型列表事件未设置，尝试页面刷新以触发捕获...")
        try:
            listener_attached = False
            # 检查监听器是否已附加的逻辑可能需要根据 Playwright 的具体实现调整
            # 这里的 _events 检查是一个示例，实际可能不同
            if hasattr(page_instance, '_events') and "response" in page_instance._events:
                for handler_slot_or_func in page_instance._events["response"]:
                    actual_handler = getattr(handler_slot_or_func, 'handler', handler_slot_or_func) # Playwright 内部可能包装 handler
                    if actual_handler == _handle_model_list_response:
                        listener_attached = True
                        break
            
            if not listener_attached:
                logger.info("/v1/models: 响应监听器似乎不存在或已被移除，尝试重新添加。")
                page_instance.on("response", _handle_model_list_response)
            
            await page_instance.reload(wait_until="domcontentloaded", timeout=20000)
            logger.info(f"页面已刷新。等待模型列表事件 (最多10秒)...")
            await asyncio.wait_for(model_list_fetch_event.wait(), timeout=10.0) # 等待事件被设置
        except asyncio.TimeoutError:
            logger.warning("/v1/models: 刷新后等待模型列表事件超时。")
        except PlaywrightAsyncError as reload_err: # 假设 PlaywrightAsyncError 是 Playwright 的特定错误类型
            logger.error(f"/v1/models: 刷新页面失败: {reload_err}")
        except Exception as e:
            logger.error(f"/v1/models: 尝试触发模型列表捕获时发生错误: {e}")
        # finally:
        #    如果超时，事件不会被设置，不需要在这里强制设置，除非有特定逻辑
        #    if not model_list_fetch_event.is_set():
        #        logger.info("/v1/models: 捕获尝试后，模型列表事件仍未设置。")
                # model_list_fetch_event.set() # 如果超时后仍希望继续，可以设置，但可能数据不完整

    if parsed_model_list:
        final_model_list = [m for m in parsed_model_list if m.get("id") not in excluded_model_ids]
        logger.info(f"返回过滤后的 {len(final_model_list)} 个模型 (原缓存 {len(parsed_model_list)} 个)。排除的有: {excluded_model_ids.intersection(set(m.get('id') for m in parsed_model_list))}")
        return {"object": "list", "data": final_model_list}
    else:
        logger.warning("模型列表为空或未成功获取。返回默认后备模型。")
        fallback_model_obj = {
            "id": DEFAULT_FALLBACK_MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "camoufox-proxy-fallback",
            "display_name": DEFAULT_FALLBACK_MODEL_ID.replace("-", " ").title(),
            "description": "Default fallback model.",
            "raw_model_path": f"models/{DEFAULT_FALLBACK_MODEL_ID}" # 遵循格式
        }
        return {"object": "list", "data": [fallback_model_obj]}

# --- Helper: Detect Error ---
async def detect_and_extract_page_error(page: AsyncPage, req_id: str) -> Optional[str]:
    error_toast_locator = page.locator(ERROR_TOAST_SELECTOR).last
    try:
        await error_toast_locator.wait_for(state='visible', timeout=500)
        message_locator = error_toast_locator.locator('span.content-text')
        error_message = await message_locator.text_content(timeout=500)
        if error_message:
             logger.error(f"[{req_id}]    检测到并提取错误消息: {error_message}")
             return error_message.strip()
        else:
             logger.warning(f"[{req_id}]    检测到错误提示框，但无法提取消息。")
             return "检测到错误提示框，但无法提取特定消息。"
    except PlaywrightAsyncError: return None
    except Exception as e:
        logger.warning(f"[{req_id}]    检查页面错误时出错: {e}")
        return None

# --- Snapshot Helper ---
async def save_error_snapshot(error_name: str = 'error'):
    name_parts = error_name.split('_')
    req_id = name_parts[-1] if len(name_parts) > 1 and len(name_parts[-1]) == 7 else None
    base_error_name = error_name if not req_id else '_'.join(name_parts[:-1])
    log_prefix = f"[{req_id}]" if req_id else "[无请求ID]"
    page_to_snapshot = page_instance
    if not browser_instance or not browser_instance.is_connected() or not page_to_snapshot or page_to_snapshot.is_closed():
        logger.warning(f"{log_prefix} 无法保存快照 ({base_error_name})，浏览器/页面不可用。")
        return
    logger.info(f"{log_prefix} 尝试保存错误快照 ({base_error_name})...")
    timestamp = int(time.time() * 1000)
    error_dir = os.path.join(os.path.dirname(__file__), 'errors_py')
    try:
        os.makedirs(error_dir, exist_ok=True)
        filename_suffix = f"{req_id}_{timestamp}" if req_id else f"{timestamp}"
        filename_base = f"{base_error_name}_{filename_suffix}"
        screenshot_path = os.path.join(error_dir, f"{filename_base}.png")
        html_path = os.path.join(error_dir, f"{filename_base}.html")
        try:
            await page_to_snapshot.screenshot(path=screenshot_path, full_page=True, timeout=15000)
            logger.info(f"{log_prefix}   快照已保存到: {screenshot_path}")
        except Exception as ss_err:
            logger.error(f"{log_prefix}   保存屏幕截图失败 ({base_error_name}): {ss_err}")
        try:
            content = await page_to_snapshot.content()
            f = None
            try:
                f = open(html_path, 'w', encoding='utf-8')
                f.write(content)
                logger.info(f"{log_prefix}   HTML 已保存到: {html_path}")
            except Exception as write_err:
                logger.error(f"{log_prefix}   保存 HTML 失败 ({base_error_name}): {write_err}")
            finally:
                if f:
                    try:
                        f.close()
                        logger.debug(f"{log_prefix}   HTML 文件已正确关闭")
                    except Exception as close_err:
                        logger.error(f"{log_prefix}   关闭 HTML 文件时出错: {close_err}")
        except Exception as html_err:
            logger.error(f"{log_prefix}   获取页面内容失败 ({base_error_name}): {html_err}")
    except Exception as dir_err:
        logger.error(f"{log_prefix}   创建错误目录或保存快照时发生其他错误 ({base_error_name}): {dir_err}")

# --- Get response via Edit Button ---
async def get_response_via_edit_button(
    page: AsyncPage,
    req_id: str,
    check_client_disconnected: Callable
) -> Optional[str]:
    logger.info(f"[{req_id}] (Helper) 尝试通过编辑按钮获取响应...")
    last_message_container = page.locator('ms-chat-turn').last
    edit_button = last_message_container.get_by_label("Edit")
    finish_edit_button = last_message_container.get_by_label("Stop editing")
    autosize_textarea_locator = last_message_container.locator('ms-autosize-textarea')
    actual_textarea_locator = autosize_textarea_locator.locator('textarea')
    try:
        logger.info(f"[{req_id}]   - 尝试悬停最后一条消息以显示 'Edit' 按钮...")
        try:
            # 对消息容器执行悬停操作
            await last_message_container.hover(timeout=CLICK_TIMEOUT_MS / 2) # 使用一半的点击超时作为悬停超时
            await asyncio.sleep(0.3) # 等待悬停效果生效
            check_client_disconnected("编辑响应 - 悬停后: ")
        except Exception as hover_err:
            logger.warning(f"[{req_id}]   - (get_response_via_edit_button) 悬停最后一条消息失败 (忽略): {type(hover_err).__name__}")
            # 即使悬停失败，也继续尝试后续操作，Playwright的expect_async可能会处理
        
        logger.info(f"[{req_id}]   - 定位并点击 'Edit' 按钮...")
        try:
            await expect_async(edit_button).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("编辑响应 - 'Edit' 按钮可见后: ")
            await edit_button.click(timeout=CLICK_TIMEOUT_MS)
            logger.info(f"[{req_id}]   - 'Edit' 按钮已点击。")
        except Exception as edit_btn_err:
            logger.error(f"[{req_id}]   - 'Edit' 按钮不可见或点击失败: {edit_btn_err}")
            await save_error_snapshot(f"edit_response_edit_button_failed_{req_id}")
            return None
        check_client_disconnected("编辑响应 - 点击 'Edit' 按钮后: ")
        await asyncio.sleep(0.3)
        check_client_disconnected("编辑响应 - 点击 'Edit' 按钮后延时后: ")
        logger.info(f"[{req_id}]   - 从文本区域获取内容...")
        response_content = None
        textarea_failed = False
        try:
            await expect_async(autosize_textarea_locator).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("编辑响应 - autosize-textarea 可见后: ")
            try:
                data_value_content = await autosize_textarea_locator.get_attribute("data-value")
                check_client_disconnected("编辑响应 - get_attribute data-value 后: ")
                if data_value_content is not None:
                    response_content = str(data_value_content)
                    logger.info(f"[{req_id}]   - 从 data-value 获取内容成功。")
            except Exception as data_val_err:
                logger.warning(f"[{req_id}]   - 获取 data-value 失败: {data_val_err}")
                check_client_disconnected("编辑响应 - get_attribute data-value 错误后: ")
            if response_content is None:
                logger.info(f"[{req_id}]   - data-value 获取失败或为None，尝试从内部 textarea 获取 input_value...")
                try:
                    await expect_async(actual_textarea_locator).to_be_visible(timeout=CLICK_TIMEOUT_MS/2)
                    input_val_content = await actual_textarea_locator.input_value(timeout=CLICK_TIMEOUT_MS/2)
                    check_client_disconnected("编辑响应 - input_value 后: ")
                    if input_val_content is not None:
                        response_content = str(input_val_content)
                        logger.info(f"[{req_id}]   - 从 input_value 获取内容成功。")
                except Exception as input_val_err:
                     logger.warning(f"[{req_id}]   - 获取 input_value 也失败: {input_val_err}")
                     check_client_disconnected("编辑响应 - input_value 错误后: ")
            if response_content is not None:
                response_content = response_content.strip()
                content_preview = response_content[:100].replace('\\n', '\\\\n')
                logger.info(f"[{req_id}]   - ✅ 最终获取内容 (长度={len(response_content)}): '{content_preview}...'")
            else:
                logger.warning(f"[{req_id}]   - 所有方法 (data-value, input_value) 内容获取均失败或返回 None。")
                textarea_failed = True
        except Exception as textarea_err:
            logger.error(f"[{req_id}]   - 定位或处理文本区域时失败: {textarea_err}")
            textarea_failed = True
            response_content = None
            check_client_disconnected("编辑响应 - 获取文本区域错误后: ")
        if not textarea_failed:
            logger.info(f"[{req_id}]   - 定位并点击 'Stop editing' 按钮...")
            try:
                await expect_async(finish_edit_button).to_be_visible(timeout=CLICK_TIMEOUT_MS)
                check_client_disconnected("编辑响应 - 'Stop editing' 按钮可见后: ")
                await finish_edit_button.click(timeout=CLICK_TIMEOUT_MS)
                logger.info(f"[{req_id}]   - 'Stop editing' 按钮已点击。")
            except Exception as finish_btn_err:
                logger.warning(f"[{req_id}]   - 'Stop editing' 按钮不可见或点击失败: {finish_btn_err}")
                await save_error_snapshot(f"edit_response_finish_button_failed_{req_id}")
            check_client_disconnected("编辑响应 - 点击 'Stop editing' 后: ")
            await asyncio.sleep(0.2)
            check_client_disconnected("编辑响应 - 点击 'Stop editing' 后延时后: ")
        else:
             logger.info(f"[{req_id}]   - 跳过点击 'Stop editing' 按钮，因为文本区域读取失败。")
        return response_content
    except ClientDisconnectedError:
        logger.info(f"[{req_id}] (Helper Edit) 客户端断开连接。")
        raise
    except Exception as e:
        logger.exception(f"[{req_id}] 通过编辑按钮获取响应过程中发生意外错误")
        await save_error_snapshot(f"edit_response_unexpected_error_{req_id}")
        return None

# --- Get response via Copy Button ---
async def get_response_via_copy_button(
    page: AsyncPage,
    req_id: str,
    check_client_disconnected: Callable
) -> Optional[str]:
    logger.info(f"[{req_id}] (Helper) 尝试通过复制按钮获取响应...")
    last_message_container = page.locator('ms-chat-turn').last
    more_options_button = last_message_container.get_by_label("Open options")
    copy_markdown_button = page.get_by_role("menuitem", name="Copy markdown")
    try:
        logger.info(f"[{req_id}]   - 尝试悬停最后一条消息以显示选项...")
        await last_message_container.hover(timeout=CLICK_TIMEOUT_MS)
        check_client_disconnected("复制响应 - 悬停后: ")
        await asyncio.sleep(0.5)
        check_client_disconnected("复制响应 - 悬停后延时后: ")
        logger.info(f"[{req_id}]   - 已悬停。")
        logger.info(f"[{req_id}]   - 定位并点击 '更多选项' 按钮...")
        try:
            await expect_async(more_options_button).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("复制响应 - 更多选项按钮可见后: ")
            await more_options_button.click(timeout=CLICK_TIMEOUT_MS)
            logger.info(f"[{req_id}]   - '更多选项' 已点击 (通过 get_by_label)。")
        except Exception as more_opts_err:
            logger.error(f"[{req_id}]   - '更多选项' 按钮 (通过 get_by_label) 不可见或点击失败: {more_opts_err}")
            await save_error_snapshot(f"copy_response_more_options_failed_{req_id}")
            return None
        check_client_disconnected("复制响应 - 点击更多选项后: ")
        await asyncio.sleep(0.5)
        check_client_disconnected("复制响应 - 点击更多选项后延时后: ")
        logger.info(f"[{req_id}]   - 定位并点击 '复制 Markdown' 按钮...")
        copy_success = False
        try:
            await expect_async(copy_markdown_button).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            check_client_disconnected("复制响应 - 复制按钮可见后: ")
            await copy_markdown_button.click(timeout=CLICK_TIMEOUT_MS, force=True)
            copy_success = True
            logger.info(f"[{req_id}]   - 已点击 '复制 Markdown' (通过 get_by_role)。")
        except Exception as copy_err:
            logger.error(f"[{req_id}]   - '复制 Markdown' 按钮 (通过 get_by_role) 点击失败: {copy_err}")
            await save_error_snapshot(f"copy_response_copy_button_failed_{req_id}")
            return None
        if not copy_success:
             logger.error(f"[{req_id}]   - 未能点击 '复制 Markdown' 按钮。")
             return None
        check_client_disconnected("复制响应 - 点击复制按钮后: ")
        await asyncio.sleep(0.5)
        check_client_disconnected("复制响应 - 点击复制按钮后延时后: ")
        logger.info(f"[{req_id}]   - 正在读取剪贴板内容...")
        try:
            clipboard_content = await page.evaluate('navigator.clipboard.readText()')
            check_client_disconnected("复制响应 - 读取剪贴板后: ")
            if clipboard_content:
                content_preview = clipboard_content[:100].replace('\n', '\\\\n')
                logger.info(f"[{req_id}]   - ✅ 成功获取剪贴板内容 (长度={len(clipboard_content)}): '{content_preview}...'")
                return clipboard_content
            else:
                logger.error(f"[{req_id}]   - 剪贴板内容为空。")
                return None
        except Exception as clipboard_err:
            if "clipboard-read" in str(clipboard_err):
                 logger.error(f"[{req_id}]   - 读取剪贴板失败: 可能是权限问题。错误: {clipboard_err}")
            else:
                 logger.error(f"[{req_id}]   - 读取剪贴板失败: {clipboard_err}")
            await save_error_snapshot(f"copy_response_clipboard_read_failed_{req_id}")
            return None
    except ClientDisconnectedError:
        logger.info(f"[{req_id}] (Helper Copy) 客户端断开连接。")
        raise
    except Exception as e:
        logger.exception(f"[{req_id}] 复制响应过程中发生意外错误")
        await save_error_snapshot(f"copy_response_unexpected_error_{req_id}")
        return None

# --- Wait for Response Completion ---
async def _wait_for_response_completion(
    page: AsyncPage,
    prompt_textarea_locator: Locator,
    submit_button_locator: Locator,
    edit_button_locator: Locator,
    req_id: str,
    check_client_disconnected_func: Callable,
    current_chat_id: Optional[str],
    timeout_ms=RESPONSE_COMPLETION_TIMEOUT,
    initial_wait_ms=INITIAL_WAIT_MS_BEFORE_POLLING
) -> bool:
    logger.info(f"[{req_id}] (WaitV3) 开始等待响应完成... (超时: {timeout_ms}ms)")
    await asyncio.sleep(initial_wait_ms / 1000) # Initial brief wait
    
    start_time = time.time()
    wait_timeout_ms_short = 3000 # 3 seconds for individual element checks
    
    consecutive_empty_input_submit_disabled_count = 0
    
    while True:
        if check_client_disconnected_func(current_chat_id, req_id):
            logger.info(f"[{req_id}] (WaitV3) 客户端断开连接，中止等待。")
            return False

        current_time_elapsed_ms = (time.time() - start_time) * 1000
        if current_time_elapsed_ms > timeout_ms:
            logger.error(f"[{req_id}] (WaitV3) 等待响应完成超时 ({timeout_ms}ms)。")
            await save_error_snapshot(f"wait_completion_v3_overall_timeout_{req_id}")
            return False

        if check_client_disconnected_func(current_chat_id, req_id): return False

        # --- 主要条件: 输入框空 & 提交按钮禁用 ---
        is_input_empty = await prompt_textarea_locator.input_value() == ""
        is_submit_disabled = False
        try:
            is_submit_disabled = await submit_button_locator.is_disabled(timeout=wait_timeout_ms_short)
        except TimeoutError:
            logger.warning(f"[{req_id}] (WaitV3) 检查提交按钮是否禁用超时。为本次检查假定其未禁用。")
        
        if check_client_disconnected_func(current_chat_id, req_id): return False

        if is_input_empty and is_submit_disabled:
            consecutive_empty_input_submit_disabled_count += 1
            if DEBUG_LOGS_ENABLED:
                logger.debug(f"[{req_id}] (WaitV3) 主要条件满足: 输入框空，提交按钮禁用 (计数: {consecutive_empty_input_submit_disabled_count})。")

            # --- 最终确认: 编辑按钮可见 ---
            try:
                if await edit_button_locator.is_visible(timeout=wait_timeout_ms_short):
                    logger.info(f"[{req_id}] (WaitV3) ✅ 响应完成: 输入框空，提交按钮禁用，编辑按钮可见。")
                    return True # 明确完成
            except TimeoutError:
                if DEBUG_LOGS_ENABLED:
                    logger.debug(f"[{req_id}] (WaitV3) 主要条件满足后，检查编辑按钮可见性超时。")
            
            if check_client_disconnected_func(current_chat_id, req_id): return False

            # 启发式完成: 如果主要条件持续满足，但编辑按钮仍未出现
            if consecutive_empty_input_submit_disabled_count >= 3: # 例如，大约 1.5秒 (3 * 0.5秒轮询)
                logger.warning(f"[{req_id}] (WaitV3) 响应可能已完成 (启发式): 输入框空，提交按钮禁用，但在 {consecutive_empty_input_submit_disabled_count} 次检查后编辑按钮仍未出现。假定完成。后续若内容获取失败，可能与此有关。")
                # 不再在此处保存快照: await save_error_snapshot(f"wait_completion_v3_heuristic_no_edit_{req_id}")
                return True # 启发式完成
        else: # 主要条件 (输入框空 & 提交按钮禁用) 未满足
            consecutive_empty_input_submit_disabled_count = 0 # 重置计数器
            if DEBUG_LOGS_ENABLED:
                reasons = []
                if not is_input_empty: reasons.append("输入框非空")
                if not is_submit_disabled: reasons.append("提交按钮非禁用")
                logger.debug(f"[{req_id}] (WaitV3) 主要条件未满足 ({', '.join(reasons)}). 继续轮询...")

        await asyncio.sleep(0.5) # 轮询间隔

# --- Get Final Response Content ---
async def _get_final_response_content(
    page: AsyncPage,
    req_id: str,
    check_client_disconnected: Callable
) -> Optional[str]:
    logger.info(f"[{req_id}] (Helper GetContent) 开始获取最终响应内容...")
    response_content = await get_response_via_edit_button(
        page, req_id, check_client_disconnected
    )
    if response_content is not None:
        logger.info(f"[{req_id}] (Helper GetContent) ✅ 成功通过编辑按钮获取内容。")
        return response_content
    logger.warning(f"[{req_id}] (Helper GetContent) 编辑按钮方法失败或返回空，回退到复制按钮方法...")
    response_content = await get_response_via_copy_button(
        page, req_id, check_client_disconnected
    )
    if response_content is not None:
        logger.info(f"[{req_id}] (Helper GetContent) ✅ 成功通过复制按钮获取内容。")
        return response_content
    logger.error(f"[{req_id}] (Helper GetContent) 所有获取响应内容的方法均失败。")
    await save_error_snapshot(f"get_content_all_methods_failed_{req_id}")
    return None

# --- Queue Worker ---
async def queue_worker():
    logger.info("--- 队列 Worker 已启动 ---")
    was_last_request_streaming = False
    last_request_completion_time = 0
    while True:
        request_item = None; result_future = None; req_id = "UNKNOWN"; completion_event = None
        try:
            queue_size = request_queue.qsize()
            if queue_size > 0:
                checked_count = 0
                items_to_requeue = []
                processed_ids = set()
                while checked_count < queue_size and checked_count < 10:
                    try:
                        item = request_queue.get_nowait()
                        item_req_id = item.get("req_id", "unknown")
                        if item_req_id in processed_ids:
                             items_to_requeue.append(item)
                             continue
                        processed_ids.add(item_req_id)
                        if not item.get("cancelled", False):
                            item_http_request = item.get("http_request")
                            if item_http_request:
                                try:
                                    if await item_http_request.is_disconnected():
                                        logger.info(f"[{item_req_id}] (Worker Queue Check) 检测到客户端已断开，标记为取消。")
                                        item["cancelled"] = True
                                        item_future = item.get("result_future")
                                        if item_future and not item_future.done():
                                            item_future.set_exception(HTTPException(status_code=499, detail=f"[{item_req_id}] Client disconnected while queued."))
                                except Exception as check_err:
                                    logger.error(f"[{item_req_id}] (Worker Queue Check) Error checking disconnect: {check_err}")
                        items_to_requeue.append(item)
                        checked_count += 1
                    except asyncio.QueueEmpty:
                        break
                for item in items_to_requeue:
                    await request_queue.put(item)
            request_item = await request_queue.get()
            req_id = request_item["req_id"]
            request_data = request_item["request_data"]
            http_request = request_item["http_request"]
            result_future = request_item["result_future"]
            if request_item.get("cancelled", False):
                logger.info(f"[{req_id}] (Worker) 请求已取消，跳过。")
                if not result_future.done(): result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] 请求已被用户取消"))
                request_queue.task_done(); continue
            is_streaming_request = request_data.stream
            logger.info(f"[{req_id}] (Worker) 取出请求。模式: {'流式' if is_streaming_request else '非流式'}")
            current_time = time.time()
            if was_last_request_streaming and is_streaming_request and (current_time - last_request_completion_time < 1.0):
                delay_time = max(0.5, 1.0 - (current_time - last_request_completion_time))
                logger.info(f"[{req_id}] (Worker) 连续流式请求，添加 {delay_time:.2f}s 延迟...")
                await asyncio.sleep(delay_time)
            if await http_request.is_disconnected():
                 logger.info(f"[{req_id}] (Worker) 客户端在等待锁时断开。取消。")
                 if not result_future.done(): result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] 客户端关闭了请求"))
                 request_queue.task_done(); continue
            logger.info(f"[{req_id}] (Worker) 等待处理锁...")
            async with processing_lock:
                logger.info(f"[{req_id}] (Worker) 已获取处理锁。开始核心处理...")
                if await http_request.is_disconnected():
                     logger.info(f"[{req_id}] (Worker) 客户端在获取锁后断开。取消。")
                     if not result_future.done(): result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] 客户端关闭了请求"))
                elif result_future.done():
                     logger.info(f"[{req_id}] (Worker) Future 在处理前已完成/取消。跳过。")
                else:
                    returned_value = await _process_request_refactored(
                        req_id, request_data, http_request, result_future
                    )

                    completion_event, submit_btn_loc, client_disco_checker = None, None, None
                    current_request_was_streaming = False # Variable to track if the current request was streaming

                    if isinstance(returned_value, tuple) and len(returned_value) == 3:
                        completion_event, submit_btn_loc, client_disco_checker = returned_value
                        # A non-None completion_event signifies a streaming request
                        if completion_event is not None:
                            current_request_was_streaming = True
                            logger.info(f"[{req_id}] (Worker) _process_request_refactored returned stream info (event, locator, checker).")
                        else:
                            # This case (tuple of Nones) means it was likely a non-streaming path within _process_request_refactored
                            # or an early exit where stream-specific objects weren't fully initialized.
                            current_request_was_streaming = False # Explicitly false
                            logger.info(f"[{req_id}] (Worker) _process_request_refactored returned a tuple, but completion_event is None (likely non-stream or early exit).")
                    elif returned_value is None:
                        # Explicit None return is for non-streaming success from _process_request_refactored
                        current_request_was_streaming = False
                        logger.info(f"[{req_id}] (Worker) _process_request_refactored returned non-stream completion (None).")
                    else:
                        current_request_was_streaming = False
                        logger.warning(f"[{req_id}] (Worker) _process_request_refactored returned unexpected type: {type(returned_value)}")

                    if completion_event: # This implies current_request_was_streaming is True
                         logger.info(f"[{req_id}] (Worker) 等待流式生成器完成信号...")
                         try:
                              await asyncio.wait_for(completion_event.wait(), timeout=RESPONSE_COMPLETION_TIMEOUT/1000 + 60)
                              logger.info(f"[{req_id}] (Worker) ✅ 流式生成器完成信号收到。")

                              if submit_btn_loc and client_disco_checker:
                                  logger.info(f"[{req_id}] (Worker) 流式响应完成，等待发送按钮禁用...")
                                  wait_timeout_ms = 30000  # 30 seconds
                                  try:
                                      # Check disconnect before starting the potentially long wait
                                      client_disco_checker("流式响应后等待发送按钮禁用 - 前置检查: ")
                                      await asyncio.sleep(0.5) # Give UI a moment to update after stream completion
                                      await expect_async(submit_btn_loc).to_be_disabled(timeout=wait_timeout_ms)
                                      logger.info(f"[{req_id}] ✅ 发送按钮已禁用。")
                                  except PlaywrightAsyncError as e_pw_disabled:
                                      logger.warning(f"[{req_id}] ⚠️ 流式响应后等待发送按钮禁用超时或错误: {e_pw_disabled}")
                                      await save_error_snapshot(f"stream_post_submit_button_disabled_timeout_{req_id}")
                                  except ClientDisconnectedError:
                                      logger.info(f"[{req_id}] 客户端在流式响应后等待发送按钮禁用时断开连接。")
                                      # This error will be caught by the outer try/except in the worker loop if it needs to propagate
                                  except Exception as e_disable_wait:
                                      logger.exception(f"[{req_id}] ❌ 流式响应后等待发送按钮禁用时发生意外错误。")
                                      await save_error_snapshot(f"stream_post_submit_button_disabled_unexpected_{req_id}")
                              elif current_request_was_streaming: # Log if stream but no locators/checker
                                  logger.warning(f"[{req_id}] (Worker) 流式请求但 submit_btn_loc 或 client_disco_checker 未提供。跳过按钮禁用等待。")

                         except asyncio.TimeoutError:
                              logger.warning(f"[{req_id}] (Worker) ⚠️ 等待流式生成器完成信号超时。")
                              if not result_future.done(): result_future.set_exception(HTTPException(status_code=504, detail=f"[{req_id}] Stream generation timed out waiting for completion signal."))
                         except ClientDisconnectedError as cd_err: # Catch disconnect during event.wait()
                              logger.info(f"[{req_id}] (Worker) 客户端在等待流式完成事件时断开: {cd_err}")
                              if not result_future.done(): result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] Client disconnected during stream event wait."))
                         except Exception as ev_wait_err:
                              logger.error(f"[{req_id}] (Worker) ❌ 等待流式完成事件时出错: {ev_wait_err}")
                              if not result_future.done(): result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Error waiting for stream completion: {ev_wait_err}"))
            # 清空流式队列缓存
            logger.info(f"[{req_id}] (Worker) 尝试清空流式队列缓存...")
            await clear_stream_queue()   
            logger.info(f"[{req_id}] (Worker) 释放处理锁。")
            was_last_request_streaming = is_streaming_request
            last_request_completion_time = time.time()
        except asyncio.CancelledError:
            logger.info("--- 队列 Worker 被取消 ---")
            if result_future and not result_future.done(): result_future.cancel("Worker cancelled")
            break
        except Exception as e:
            logger.error(f"[{req_id}] (Worker) ❌ 处理请求时发生意外错误: {e}", exc_info=True)
            if result_future and not result_future.done():
                result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] 服务器内部错误: {e}"))
            await save_error_snapshot(f"worker_loop_error_{req_id}")
        finally:
             if request_item: request_queue.task_done()
    logger.info("--- 队列 Worker 已停止 ---")

# --- Helper function to use external helper service ---
async def use_helper_get_response(helper_endpoint, helper_sapisid) -> AsyncGenerator[str, None]:
    headers = {
        'Cookie': f'SAPISID={helper_sapisid}',
        'Accept': 'text/event-stream'
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(helper_endpoint, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Helper Error: HTTP {response.status}")
                    yield "[ERROR]" # Indicate error to caller
                    return # Stop generation

                async for line in response.content:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line and not decoded_line.startswith(':'):
                        if decoded_line.startswith('data:'):
                            data = decoded_line[5:].strip()
                            if data:
                                yield data
                        else: # Should not happen with SSE, but yield anyway
                            yield decoded_line
    except aiohttp.ClientError as e:
        logger.error(f"Error connecting to helper server: {e}")
        raise # Re-raise to be caught by caller
    except Exception as e:
        logger.error(f"Unexpected error in use_helper_get_response: {e}")
        raise # Re-raise


async def use_stream_response(req_id: str) -> AsyncGenerator[Any, None]: # 添加 req_id
    total_empty = 0
    log_state = STREAM_TIMEOUT_LOG_STATE # Access global state

    while True:
        data_chunk = None
        try:
            if STREAM_QUEUE is None: # 检查 STREAM_QUEUE 是否为 None
                logger.error(f"[{req_id}] STREAM_QUEUE is None in use_stream_response.")
                yield {"done": True, "reason": "stream_system_error", "body": "Auxiliary stream not available.", "function": []}
                return
            data_chunk = await asyncio.to_thread(STREAM_QUEUE.get_nowait)

            if data_chunk is not None:
                total_empty = 0 # Reset counter on successful read
                if log_state["consecutive_timeouts"] > 0: # 使用字典访问
                    logger.info(f"[{req_id}] Auxiliary stream data received after {log_state['consecutive_timeouts']} consecutive empty reads/timeouts. Resetting.")
                    log_state["consecutive_timeouts"] = 0
                    log_state["suppress_until_time"] = 0.0
                
                data = json.loads(data_chunk)
                yield data
                if data.get("done") is True:
                    return
        except queue.Empty: # 更具体的异常
            total_empty += 1
        except json.JSONDecodeError as json_e: # 更具体的异常
            logger.error(f"[{req_id}] JSONDecodeError in use_stream_response: {json_e}. Data: '{data_chunk}'")
            total_empty += 1
        except Exception as e_q_get: # 通用异常作为后备
            logger.error(f"[{req_id}] Unexpected error getting from STREAM_QUEUE: {e_q_get}", exc_info=True)
            total_empty += 1

        if total_empty > 300: # Timeout condition
            log_state["consecutive_timeouts"] += 1
            current_time = time.monotonic() # 使用 monotonic time

            if log_state["consecutive_timeouts"] <= log_state["max_initial_errors"]:
                logger.error(f"[{req_id}] Auxiliary stream data timeout (Attempt {total_empty}, Consecutive global: {log_state['consecutive_timeouts']}). Helper service might be down.")
                log_state["last_error_log_time"] = current_time
                # Set suppress_until_time only after the initial burst of errors.
                if log_state["consecutive_timeouts"] == log_state["max_initial_errors"]:
                    log_state["suppress_until_time"] = current_time + log_state["suppress_duration_after_initial_burst"]
            elif current_time >= log_state["suppress_until_time"]:
                logger.warning(f"[{req_id}] Auxiliary stream continues to time out (Global consecutive: {log_state['consecutive_timeouts']}). Last error log ~{current_time - log_state['last_error_log_time']:.0f}s ago. Next warning in {log_state['warning_interval_after_suppress']:.0f}s.")
                log_state["last_error_log_time"] = current_time
                log_state["suppress_until_time"] = current_time + log_state["warning_interval_after_suppress"]
            # Else: Log is suppressed

            yield {"done": True, "reason": "internal_timeout", "body": "", "function": []} # 特定超时信号
            return
        
        await asyncio.sleep(0.1) # 异步休眠
async def clear_stream_queue():
    if STREAM_QUEUE is None:
        logger.info("流队列未初始化或已被禁用，跳过清空操作。")
        return
    while True:
        try:
            data_chunk = await asyncio.to_thread(STREAM_QUEUE.get_nowait)
            # logger.info(f"清空流式队列缓存，丢弃数据: {data_chunk}")
        except queue.Empty:
            logger.info("流式队列已清空 (捕获到 queue.Empty)。")
            break
        except Exception as e:
            logger.error(f"清空流式队列时发生意外错误: {e}", exc_info=True)
            break
    logger.info("流式队列缓存清空完毕。")


# --- Core Request Processing Logic ---
async def _process_request_refactored(
    req_id: str,
    request: ChatCompletionRequest,
    http_request: Request,
    result_future: Future
) -> Optional[Tuple[Event, Locator, Callable[[str], bool]]]:
    model_actually_switched_in_current_api_call = False
    logger.info(f"[{req_id}] (Refactored Process) 开始处理请求...")
    logger.info(f"[{req_id}]   请求参数 - Model: {request.model}, Stream: {request.stream}")
    logger.info(f"[{req_id}]   请求参数 - Temperature: {request.temperature}")
    logger.info(f"[{req_id}]   请求参数 - Max Output Tokens: {request.max_output_tokens}")
    logger.info(f"[{req_id}]   请求参数 - Stop Sequences: {request.stop}")
    logger.info(f"[{req_id}]   请求参数 - Top P: {request.top_p}")
    is_streaming = request.stream
    page: Optional[AsyncPage] = page_instance
    completion_event: Optional[Event] = None
    requested_model = request.model
    model_id_to_use = None
    needs_model_switching = False
    if requested_model and requested_model != MODEL_NAME:
        requested_model_parts = requested_model.split('/')
        requested_model_id = requested_model_parts[-1] if len(requested_model_parts) > 1 else requested_model
        logger.info(f"[{req_id}] 请求使用模型: {requested_model_id}")
        if parsed_model_list:
            valid_model_ids = [m.get("id") for m in parsed_model_list]
            if requested_model_id not in valid_model_ids:
                logger.error(f"[{req_id}] ❌ 无效的模型ID: {requested_model_id}。可用模型: {valid_model_ids}")
                raise HTTPException(status_code=400, detail=f"[{req_id}] Invalid model '{requested_model_id}'. Available models: {', '.join(valid_model_ids)}")
        model_id_to_use = requested_model_id
        global current_ai_studio_model_id
        if current_ai_studio_model_id != model_id_to_use:
            needs_model_switching = True
            logger.info(f"[{req_id}] 需要切换模型: 当前={current_ai_studio_model_id} -> 目标={model_id_to_use}")
        else:
            logger.info(f"[{req_id}] 请求模型与当前模型相同 ({model_id_to_use})，无需切换")
    else:
        logger.info(f"[{req_id}] 未指定具体模型或使用代理模型名称，将使用当前模型: {current_ai_studio_model_id or '未知'}")
    client_disconnected_event = Event()
    disconnect_check_task = None
    input_field_locator = page.locator(INPUT_SELECTOR) if page else None # Handle page=None
    submit_button_locator = page.locator(SUBMIT_BUTTON_SELECTOR) if page else None # Handle page=None

    async def check_disconnect_periodically():
        while not client_disconnected_event.is_set():
            try:
                if await http_request.is_disconnected():
                    logger.info(f"[{req_id}] (Disco Check Task) 客户端断开。设置事件并尝试停止。")
                    client_disconnected_event.set()
                    try:
                        if submit_button_locator and await submit_button_locator.is_enabled(timeout=1500):
                             if input_field_locator and await input_field_locator.input_value(timeout=1500) == '':
                                 logger.info(f"[{req_id}] (Disco Check Task)   点击停止...")
                                 await submit_button_locator.click(timeout=3000, force=True)
                    except Exception as click_err: logger.warning(f"[{req_id}] (Disco Check Task) 停止按钮点击失败: {click_err}")
                    if not result_future.done(): result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] 客户端在处理期间关闭了请求"))
                    break
                await asyncio.sleep(1.0)
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"[{req_id}] (Disco Check Task) 错误: {e}")
                client_disconnected_event.set()
                if not result_future.done(): result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Internal disconnect checker error: {e}"))
                break
    disconnect_check_task = asyncio.create_task(check_disconnect_periodically())
    def check_client_disconnected(*args):
        msg_to_log = ""
        if len(args) == 1 and isinstance(args[0], str):
            msg_to_log = args[0]

        if client_disconnected_event.is_set():
            # req_id is from the outer scope of _process_request_refactored
            logger.info(f"[{req_id}] {msg_to_log}检测到客户端断开连接事件。")
            raise ClientDisconnectedError(f"[{req_id}] Client disconnected event set.")
        return False
    try:
        if not page or page.is_closed() or not is_page_ready:
            raise HTTPException(status_code=503, detail=f"[{req_id}] AI Studio 页面丢失或未就绪。", headers={"Retry-After": "30"})
        check_client_disconnected("Initial Page Check: ")
        if needs_model_switching and model_id_to_use:
            async with model_switching_lock:
                model_before_switch_attempt = current_ai_studio_model_id
                if current_ai_studio_model_id != model_id_to_use:
                    logger.info(f"[{req_id}] 获取锁后准备切换: 当前内存中模型={current_ai_studio_model_id}, 目标={model_id_to_use}")
                    switch_success = await switch_ai_studio_model(page, model_id_to_use, req_id)
                    if switch_success:
                        current_ai_studio_model_id = model_id_to_use
                        model_actually_switched_in_current_api_call = True
                        logger.info(f"[{req_id}] ✅ 模型切换成功。全局模型状态已更新为: {current_ai_studio_model_id}")
                    else:
                        logger.warning(f"[{req_id}] ❌ 模型切换至 {model_id_to_use} 失败 (AI Studio 未接受或覆盖了更改)。")
                        active_model_id_after_fail = model_before_switch_attempt
                        try:
                            final_prefs_str_after_fail = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
                            if final_prefs_str_after_fail:
                                final_prefs_obj_after_fail = json.loads(final_prefs_str_after_fail)
                                model_path_in_final_prefs = final_prefs_obj_after_fail.get("promptModel")
                                if model_path_in_final_prefs and isinstance(model_path_in_final_prefs, str):
                                    active_model_id_after_fail = model_path_in_final_prefs.split('/')[-1]
                        except Exception as read_final_prefs_err:
                            logger.error(f"[{req_id}] 切换失败后读取最终 localStorage 出错: {read_final_prefs_err}")
                        current_ai_studio_model_id = active_model_id_after_fail
                        logger.info(f"[{req_id}] 全局模型状态在切换失败后设置为 (或保持为): {current_ai_studio_model_id}")
                        actual_displayed_model_name = "未知 (无法读取)"
                        try:
                            model_wrapper_locator = page.locator('#mat-select-value-0 mat-select-trigger').first
                            actual_displayed_model_name = await model_wrapper_locator.inner_text(timeout=3000)
                        except Exception:
                            pass
                        raise HTTPException(
                            status_code=422,
                            detail=f"[{req_id}] AI Studio 未能应用所请求的模型 '{model_id_to_use}' 或该模型不受支持。请选择 AI Studio 网页界面中可用的模型。当前实际生效的模型 ID 为 '{current_ai_studio_model_id}', 页面显示为 '{actual_displayed_model_name}'."
                        )
                else:
                    logger.info(f"[{req_id}] 获取锁后发现模型已是目标模型 {current_ai_studio_model_id}，无需切换")
        async with params_cache_lock:
            cached_model_for_params = page_params_cache.get("last_known_model_id_for_params")
            if model_actually_switched_in_current_api_call or \
               (current_ai_studio_model_id is not None and current_ai_studio_model_id != cached_model_for_params):
                action_taken = "Invalidating" if page_params_cache else "Initializing"
                logger.info(f"[{req_id}] {action_taken} parameter cache. Reason: Model context changed (switched this call: {model_actually_switched_in_current_api_call}, current model: {current_ai_studio_model_id}, cache model: {cached_model_for_params}).")
                page_params_cache.clear()
                if current_ai_studio_model_id:
                    page_params_cache["last_known_model_id_for_params"] = current_ai_studio_model_id
            else:
                logger.debug(f"[{req_id}] Parameter cache for model '{cached_model_for_params}' remains valid (current model: '{current_ai_studio_model_id}', switched this call: {model_actually_switched_in_current_api_call}).")
        
        try: validate_chat_request(request.messages, req_id)
        except ValueError as e: raise HTTPException(status_code=400, detail=f"[{req_id}] 无效请求: {e}")

        system_prompt_content, prepared_prompt = prepare_combined_prompt(request.messages, req_id)
        check_client_disconnected("After Prompt Prep: ")

        logger.info(f"[{req_id}] (Refactored Process) 开始清空聊天记录...")
        try:
            clear_chat_button_locator = page.locator(CLEAR_CHAT_BUTTON_SELECTOR)
            confirm_button_locator = page.locator(CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR)
            overlay_locator = page.locator(OVERLAY_SELECTOR)

            can_attempt_clear = False
            try:
                await expect_async(clear_chat_button_locator).to_be_enabled(timeout=3000)
                can_attempt_clear = True
                logger.info(f"[{req_id}] “清空聊天”按钮可用，继续清空流程。")
            except Exception as e_enable:
                is_new_chat_url = '/prompts/new_chat' in page.url.rstrip('/')
                if is_new_chat_url:
                    logger.info(f"[{req_id}] “清空聊天”按钮不可用 (预期，因为在 new_chat 页面)。跳过清空操作。")
                else:
                    logger.warning(f"[{req_id}] 等待“清空聊天”按钮可用失败: {e_enable}。清空操作可能无法执行。")
            
            check_client_disconnected("清空聊天 - “清空聊天”按钮可用性检查后: ")

            if can_attempt_clear:
                overlay_initially_visible = False
                try:
                    if await overlay_locator.is_visible(timeout=1000): # Short timeout for initial check
                        overlay_initially_visible = True
                        logger.info(f"[{req_id}] 清空聊天确认遮罩层已可见。直接点击“继续”。")
                except TimeoutError:
                    logger.info(f"[{req_id}] 清空聊天确认遮罩层初始不可见 (检查超时或未找到)。")
                    overlay_initially_visible = False
                except Exception as e_vis_check:
                    logger.warning(f"[{req_id}] 检查遮罩层可见性时发生错误: {e_vis_check}。假定不可见。")
                    overlay_initially_visible = False
                
                check_client_disconnected("清空聊天 - 初始遮罩层检查后 (can_attempt_clear=True): ")

                if overlay_initially_visible:
                    logger.info(f"[{req_id}] 点击“继续”按钮 (遮罩层已存在): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
                    await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
                else:
                    logger.info(f"[{req_id}] 点击“清空聊天”按钮: {CLEAR_CHAT_BUTTON_SELECTOR}")
                    await clear_chat_button_locator.click(timeout=CLICK_TIMEOUT_MS)
                    check_client_disconnected("清空聊天 - 点击“清空聊天”后: ")
                    try:
                        logger.info(f"[{req_id}] 等待清空聊天确认遮罩层出现: {OVERLAY_SELECTOR}")
                        await expect_async(overlay_locator).to_be_visible(timeout=WAIT_FOR_ELEMENT_TIMEOUT_MS)
                        logger.info(f"[{req_id}] 清空聊天确认遮罩层已出现。")
                    except TimeoutError:
                        error_msg = f"等待清空聊天确认遮罩层超时 (点击清空按钮后)。请求 ID: {req_id}"
                        logger.error(error_msg)
                        await save_error_snapshot(f"clear_chat_overlay_timeout_{req_id}")
                        raise PlaywrightAsyncError(error_msg)
                    
                    check_client_disconnected("清空聊天 - 遮罩层出现后: ")
                    logger.info(f"[{req_id}] 点击“继续”按钮 (在对话框中): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
                    await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
                
                check_client_disconnected("清空聊天 - 点击“继续”后: ")

                max_retries_disappear = 3
                for attempt_disappear in range(max_retries_disappear):
                    try:
                        logger.info(f"[{req_id}] 等待清空聊天确认按钮/对话框消失 (尝试 {attempt_disappear + 1}/{max_retries_disappear})...")
                        await expect_async(confirm_button_locator).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS)
                        await expect_async(overlay_locator).to_be_hidden(timeout=1000)
                        logger.info(f"[{req_id}] ✅ 清空聊天确认对话框已成功消失。")
                        break
                    except TimeoutError:
                        logger.warning(f"[{req_id}] ⚠️ 等待清空聊天确认对话框消失超时 (尝试 {attempt_disappear + 1}/{max_retries_disappear})。")
                        if attempt_disappear < max_retries_disappear - 1:
                            confirm_still_visible = False; overlay_still_visible = False
                            try: confirm_still_visible = await confirm_button_locator.is_visible(timeout=200)
                            except: pass
                            try: overlay_still_visible = await overlay_locator.is_visible(timeout=200)
                            except: pass
                            if confirm_still_visible: logger.warning(f"[{req_id}] 确认按钮在点击和等待后仍可见。")
                            if overlay_still_visible: logger.warning(f"[{req_id}] 遮罩层在点击和等待后仍可见。")
                            await asyncio.sleep(1.0)
                            check_client_disconnected(f"清空聊天 - 重试消失检查 {attempt_disappear + 1} 前: ")
                            continue
                        else:
                            error_msg = f"达到最大重试次数。清空聊天确认对话框未消失。请求 ID: {req_id}"
                            logger.error(error_msg)
                            await save_error_snapshot(f"clear_chat_dialog_disappear_timeout_{req_id}")
                            raise PlaywrightAsyncError(error_msg)
                    except ClientDisconnectedError:
                        logger.info(f"[{req_id}] 客户端在等待清空确认对话框消失时断开连接。")
                        raise
                    check_client_disconnected(f"清空聊天 - 消失检查尝试 {attempt_disappear + 1} 后: ")
                
                last_response_container = page.locator(RESPONSE_CONTAINER_SELECTOR).last
                await asyncio.sleep(0.5)
                check_client_disconnected("After Clear Post-Delay (New Logic): ")
                try:
                    await expect_async(last_response_container).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS - 500)
                    logger.info(f"[{req_id}] ✅ 聊天已成功清空 (验证通过 - 最后响应容器隐藏)。")
                except Exception as verify_err:
                    logger.warning(f"[{req_id}] ⚠️ 警告: 清空聊天验证失败 (最后响应容器未隐藏): {verify_err}")
            else:
                # If can_attempt_clear is False and it wasn't a new_chat_url, it means clear button wasn't enabled.
                # Log this situation if not already handled by the e_enable exception logging.
                if not ('/prompts/new_chat' in page.url.rstrip('/')): # Avoid logging if it was expected on new_chat
                    logger.warning(f"[{req_id}] 由于“清空聊天”按钮初始不可用，未执行清空操作。")

            check_client_disconnected("After Clear Chat Logic (New): ")
        except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as clear_err:
            if isinstance(clear_err, ClientDisconnectedError): raise # Re-raise to be caught by outer handler
            logger.error(f"[{req_id}] ❌ 错误: 清空聊天阶段出错: {clear_err}")
            await save_error_snapshot(f"clear_chat_error_{req_id}")
            # Potentially raise HTTPException here if clearing chat is critical and fails
        except Exception as clear_exc:
            logger.exception(f"[{req_id}] ❌ 错误: 清空聊天阶段意外错误")
            await save_error_snapshot(f"clear_chat_unexpected_{req_id}")
            # Potentially raise HTTPException here

        check_client_disconnected("After Clear Chat Logic: ")

        # 参数设置块 (包括新的系统提示设置)
        async with params_cache_lock:
            # 设置系统提示词 (新逻辑)
            if page and not page.is_closed():
                logger.info(f"[{req_id}] (Refactored Process) 检查并设置系统提示...")
                await set_system_prompt_in_page(
                    page,
                    system_prompt_content if system_prompt_content is not None else "", # Pass empty string if None
                    req_id,
                    check_client_disconnected
                )
            check_client_disconnected("系统提示调整 - 逻辑完成后: ")

            # 设置温度 (现有逻辑)
            if request.temperature is not None and page and not page.is_closed():
                logger.info(f"[{req_id}] (Refactored Process) 检查并调整温度设置...")
                requested_temp = request.temperature
                clamped_temp = max(0.0, min(2.0, requested_temp))
                if clamped_temp != requested_temp:
                    logger.warning(f"[{req_id}] 请求的温度 {requested_temp} 超出范围 [0, 2]，已调整为 {clamped_temp}")
                cached_temp = page_params_cache.get("temperature")
                if cached_temp is not None and abs(cached_temp - clamped_temp) < 0.001:
                    logger.info(f"[{req_id}] 温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 一致。跳过页面交互。")
                else:
                    logger.info(f"[{req_id}] 请求温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 不一致或缓存中无值。需要与页面交互。")
                    temp_input_locator = page.locator(TEMPERATURE_INPUT_SELECTOR)
                    try:
                        await expect_async(temp_input_locator).to_be_visible(timeout=5000)
                        check_client_disconnected("温度调整 - 输入框可见后: ")
                        current_temp_str = await temp_input_locator.input_value(timeout=3000)
                        check_client_disconnected("温度调整 - 读取输入框值后: ")
                        current_temp_float = float(current_temp_str)
                        logger.info(f"[{req_id}] 页面当前温度: {current_temp_float}, 请求调整后温度: {clamped_temp}")
                        if abs(current_temp_float - clamped_temp) < 0.001:
                            logger.info(f"[{req_id}] 页面当前温度 ({current_temp_float}) 与请求温度 ({clamped_temp}) 一致。更新缓存并跳过写入。")
                            page_params_cache["temperature"] = current_temp_float
                        else:
                            logger.info(f"[{req_id}] 页面温度 ({current_temp_float}) 与请求温度 ({clamped_temp}) 不同，正在更新...")
                            await temp_input_locator.fill(str(clamped_temp), timeout=5000)
                            check_client_disconnected("温度调整 - 填充输入框后: ")
                            await asyncio.sleep(0.1)
                            new_temp_str = await temp_input_locator.input_value(timeout=3000)
                            new_temp_float = float(new_temp_str)
                            if abs(new_temp_float - clamped_temp) < 0.001:
                                logger.info(f"[{req_id}] ✅ 温度已成功更新为: {new_temp_float}。更新缓存。")
                                page_params_cache["temperature"] = new_temp_float
                            else:
                                logger.warning(f"[{req_id}] ⚠️ 温度更新后验证失败。页面显示: {new_temp_float}, 期望: {clamped_temp}。清除缓存中的温度。")
                                page_params_cache.pop("temperature", None)
                                await save_error_snapshot(f"temperature_verify_fail_{req_id}")
                    except ValueError as ve:
                        logger.error(f"[{req_id}] 转换温度值为浮点数时出错: '{current_temp_str if 'current_temp_str' in locals() else '未知值'}'. 错误: {ve}。清除缓存中的温度。")
                        page_params_cache.pop("temperature", None)
                        await save_error_snapshot(f"temperature_value_error_{req_id}")
                    except PlaywrightAsyncError as pw_err:
                        logger.error(f"[{req_id}] ❌ 操作温度输入框时发生Playwright错误: {pw_err}。清除缓存中的温度。")
                        page_params_cache.pop("temperature", None)
                        await save_error_snapshot(f"temperature_playwright_error_{req_id}")
                    except ClientDisconnectedError:
                        logger.info(f"[{req_id}] 客户端在调整温度时断开连接。")
                        raise
                    except Exception as e_temp:
                        logger.exception(f"[{req_id}] ❌ 调整温度时发生未知错误。清除缓存中的温度。")
                        page_params_cache.pop("temperature", None)
                        await save_error_snapshot(f"temperature_unknown_error_{req_id}")
            check_client_disconnected("温度调整 - 逻辑完成后: ")

            # 设置最大输出 Token (现有逻辑)
            if request.max_output_tokens is not None and page and not page.is_closed():
                logger.info(f"[{req_id}] (Refactored Process) 检查并调整最大输出 Token 设置...")
                requested_max_tokens = request.max_output_tokens
                min_val_for_tokens = 1
                max_val_for_tokens_from_model = 65536
                if model_id_to_use and parsed_model_list:
                    current_model_data = next((m for m in parsed_model_list if m.get("id") == model_id_to_use), None)
                    if current_model_data and current_model_data.get("supported_max_output_tokens") is not None:
                        try:
                            supported_tokens = int(current_model_data["supported_max_output_tokens"])
                            if supported_tokens > 0: max_val_for_tokens_from_model = supported_tokens
                            else: logger.warning(f"[{req_id}] 模型 {model_id_to_use} supported_max_output_tokens 无效: {supported_tokens}")
                        except (ValueError, TypeError): logger.warning(f"[{req_id}] 模型 {model_id_to_use} supported_max_output_tokens 解析失败: {current_model_data['supported_max_output_tokens']}")
                    else: logger.warning(f"[{req_id}] 未找到模型 {model_id_to_use} 的 supported_max_output_tokens 数据。")
                else: logger.warning(f"[{req_id}] model_id_to_use ('{model_id_to_use}') 或 parsed_model_list 不可用，使用默认 tokens 上限。")
                clamped_max_tokens = max(min_val_for_tokens, min(max_val_for_tokens_from_model, requested_max_tokens))
                if clamped_max_tokens != requested_max_tokens:
                    logger.warning(f"[{req_id}] 请求的最大输出 Tokens {requested_max_tokens} 超出模型范围 [{min_val_for_tokens}, {max_val_for_tokens_from_model}]，已调整为 {clamped_max_tokens}")
                cached_max_tokens = page_params_cache.get("max_output_tokens")
                if cached_max_tokens is not None and cached_max_tokens == clamped_max_tokens:
                    logger.info(f"[{req_id}] 最大输出 Tokens ({clamped_max_tokens}) 与缓存值 ({cached_max_tokens}) 一致。跳过页面交互。")
                else:
                    logger.info(f"[{req_id}] 请求最大输出 Tokens ({clamped_max_tokens}) 与缓存值 ({cached_max_tokens}) 不一致或缓存中无值。需要与页面交互。")
                    max_tokens_input_locator = page.locator(MAX_OUTPUT_TOKENS_SELECTOR)
                    try:
                        await expect_async(max_tokens_input_locator).to_be_visible(timeout=5000)
                        check_client_disconnected("最大输出Token调整 - 输入框可见后: ")
                        current_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
                        check_client_disconnected("最大输出Token调整 - 读取输入框值后: ")
                        current_max_tokens_int = int(current_max_tokens_str)
                        logger.info(f"[{req_id}] 页面当前最大输出 Tokens: {current_max_tokens_int}, 请求调整后最大输出 Tokens: {clamped_max_tokens}")
                        if current_max_tokens_int == clamped_max_tokens:
                            logger.info(f"[{req_id}] 页面当前最大输出 Tokens ({current_max_tokens_int}) 与请求值 ({clamped_max_tokens}) 一致。更新缓存并跳过写入。")
                            page_params_cache["max_output_tokens"] = current_max_tokens_int
                        else:
                            logger.info(f"[{req_id}] 页面最大输出 Tokens ({current_max_tokens_int}) 与请求值 ({clamped_max_tokens}) 不同，正在更新...")
                            await max_tokens_input_locator.fill(str(clamped_max_tokens), timeout=5000)
                            check_client_disconnected("最大输出Token调整 - 填充输入框后: ")
                            await asyncio.sleep(0.1)
                            new_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
                            new_max_tokens_int = int(new_max_tokens_str)
                            if new_max_tokens_int == clamped_max_tokens:
                                logger.info(f"[{req_id}] ✅ 最大输出 Tokens 已成功更新为: {new_max_tokens_int}。更新缓存。")
                                page_params_cache["max_output_tokens"] = new_max_tokens_int
                            else:
                                logger.warning(f"[{req_id}] ⚠️ 最大输出 Tokens 更新后验证失败。页面显示: {new_max_tokens_int}, 期望: {clamped_max_tokens}。清除缓存中的此参数。")
                                page_params_cache.pop("max_output_tokens", None)
                                await save_error_snapshot(f"max_tokens_verify_fail_{req_id}")
                    except ValueError as ve:
                        logger.error(f"[{req_id}] 转换最大输出 Tokens 值为整数时出错: '{current_max_tokens_str if 'current_max_tokens_str' in locals() else '未知值'}'. 错误: {ve}。清除缓存中的此参数。")
                        page_params_cache.pop("max_output_tokens", None)
                        await save_error_snapshot(f"max_tokens_value_error_{req_id}")
                    except PlaywrightAsyncError as pw_err:
                        logger.error(f"[{req_id}] ❌ 操作最大输出 Tokens 输入框时发生Playwright错误: {pw_err}。清除缓存中的此参数。")
                        page_params_cache.pop("max_output_tokens", None)
                        await save_error_snapshot(f"max_tokens_playwright_error_{req_id}")
                    except ClientDisconnectedError:
                        logger.info(f"[{req_id}] 客户端在调整最大输出 Tokens 时断开连接。")
                        raise
                    except Exception as e_max_tokens:
                        logger.exception(f"[{req_id}] ❌ 调整最大输出 Tokens 时发生未知错误。清除缓存中的此参数。")
                        page_params_cache.pop("max_output_tokens", None)
                        await save_error_snapshot(f"max_tokens_unknown_error_{req_id}")
            check_client_disconnected("最大输出Token调整 - 逻辑完成后: ")

            # 设置停止序列 (现有逻辑)
            if request.stop is not None and page and not page.is_closed():
                logger.info(f"[{req_id}] (Refactored Process) 检查并设置停止序列...")
                requested_stop_sequences_raw = []
                if isinstance(request.stop, str):
                    requested_stop_sequences_raw = [request.stop]
                elif isinstance(request.stop, list):
                    requested_stop_sequences_raw = [s for s in request.stop if isinstance(s, str) and s.strip()]
                normalized_requested_stops = set(s.strip() for s in requested_stop_sequences_raw if s.strip())
                cached_stops_set = page_params_cache.get("stop_sequences")
                if cached_stops_set is not None and cached_stops_set == normalized_requested_stops:
                    logger.info(f"[{req_id}] 请求的停止序列 ({normalized_requested_stops}) 与缓存值 ({cached_stops_set}) 一致。跳过页面交互。")
                else:
                    logger.info(f"[{req_id}] 请求停止序列 ({normalized_requested_stops}) 与缓存值 ({cached_stops_set}) 不一致或缓存中无值。需要与页面交互。")
                    stop_input_locator = page.locator(STOP_SEQUENCE_INPUT_SELECTOR)
                    remove_chip_buttons_locator = page.locator(MAT_CHIP_REMOVE_BUTTON_SELECTOR)
                    interaction_successful = False
                    try:
                        logger.info(f"[{req_id}] 尝试清空已有的停止序列...")
                        initial_chip_count = await remove_chip_buttons_locator.count()
                        removed_count = 0
                        max_removals = initial_chip_count + 5
                        while await remove_chip_buttons_locator.count() > 0 and removed_count < max_removals:
                            check_client_disconnected("停止序列清除 - 循环开始: ")
                            try:
                                await remove_chip_buttons_locator.first.click(timeout=2000)
                                removed_count += 1; await asyncio.sleep(0.15)
                            except Exception: break
                        logger.info(f"[{req_id}] 已有停止序列清空尝试完成。移除 {removed_count} 个。")
                        check_client_disconnected("停止序列清除 - 完成后: ")
                        if normalized_requested_stops:
                            logger.info(f"[{req_id}] 添加新的停止序列: {normalized_requested_stops}")
                            await expect_async(stop_input_locator).to_be_visible(timeout=5000)
                            for seq in normalized_requested_stops:
                                await stop_input_locator.fill(seq, timeout=3000)
                                await stop_input_locator.press("Enter", timeout=3000)
                                await asyncio.sleep(0.2)
                                current_input_val = await stop_input_locator.input_value(timeout=1000)
                                if current_input_val:
                                     logger.warning(f"[{req_id}] 添加停止序列 '{seq}' 后输入框未清空 (值为: '{current_input_val}')。")
                            logger.info(f"[{req_id}] ✅ 新停止序列添加操作完成。")
                        else:
                            logger.info(f"[{req_id}] 没有提供新的有效停止序列来添加 (请求清空)。")
                        interaction_successful = True
                        page_params_cache["stop_sequences"] = normalized_requested_stops
                        logger.info(f"[{req_id}] 停止序列缓存已更新为: {normalized_requested_stops}")
                    except PlaywrightAsyncError as pw_err:
                        logger.error(f"[{req_id}] ❌ 操作停止序列时发生Playwright错误: {pw_err}。清除缓存中的此参数。")
                        page_params_cache.pop("stop_sequences", None)
                        await save_error_snapshot(f"stop_sequence_playwright_error_{req_id}")
                    except ClientDisconnectedError:
                        logger.info(f"[{req_id}] 客户端在调整停止序列时断开连接。")
                        raise
                    except Exception as e_stop_seq:
                        logger.exception(f"[{req_id}] ❌ 设置停止序列时发生未知错误。清除缓存中的此参数。")
                        page_params_cache.pop("stop_sequences", None)
                        await save_error_snapshot(f"stop_sequence_unknown_error_{req_id}")
            check_client_disconnected("停止序列调整 - 逻辑完成后: ")

            # 设置 Top P (现有逻辑)
            if request.top_p is not None and page and not page.is_closed():
                logger.info(f"[{req_id}] (Refactored Process) 检查并调整 Top P 设置...")
                requested_top_p = request.top_p
                clamped_top_p = max(0.0, min(1.0, requested_top_p))
                if abs(clamped_top_p - requested_top_p) > 1e-9:
                    logger.warning(f"[{req_id}] 请求的 Top P {requested_top_p} 超出范围 [0, 1]，已调整为 {clamped_top_p}")
                top_p_input_locator = page.locator(TOP_P_INPUT_SELECTOR)
                try:
                    await expect_async(top_p_input_locator).to_be_visible(timeout=5000)
                    check_client_disconnected("Top P 调整 - 输入框可见后: ")
                    current_top_p_str = await top_p_input_locator.input_value(timeout=3000)
                    check_client_disconnected("Top P 调整 - 读取输入框值后: ")
                    current_top_p_float = float(current_top_p_str)
                    logger.info(f"[{req_id}] 页面当前 Top P: {current_top_p_float}, 请求调整后 Top P: {clamped_top_p}")
                    if abs(current_top_p_float - clamped_top_p) > 1e-9:
                        logger.info(f"[{req_id}] 页面 Top P ({current_top_p_float}) 与请求 Top P ({clamped_top_p}) 不同，正在更新...")
                        await top_p_input_locator.fill(str(clamped_top_p), timeout=5000)
                        check_client_disconnected("Top P 调整 - 填充输入框后: ")
                        await asyncio.sleep(0.1)
                        new_top_p_str = await top_p_input_locator.input_value(timeout=3000)
                        new_top_p_float = float(new_top_p_str)
                        if abs(new_top_p_float - clamped_top_p) < 1e-9:
                            logger.info(f"[{req_id}] ✅ Top P 已成功更新为: {new_top_p_float}")
                        else:
                            logger.warning(f"[{req_id}] ⚠️ Top P 更新后验证失败。页面显示: {new_top_p_float}, 期望: {clamped_top_p}")
                    else:
                        logger.info(f"[{req_id}] 页面 Top P ({current_top_p_float}) 与请求 Top P ({clamped_top_p}) 一致或在容差范围内，无需更改。")
                except ValueError as ve:
                    logger.error(f"[{req_id}] 转换 Top P 值为浮点数时出错: '{current_top_p_str if 'current_top_p_str' in locals() else '未知值'}'. 错误: {ve}")
                    await save_error_snapshot(f"top_p_value_error_{req_id}")
                except PlaywrightAsyncError as pw_err:
                    logger.error(f"[{req_id}] ❌ 操作 Top P 输入框时发生Playwright错误: {pw_err}")
                    await save_error_snapshot(f"top_p_playwright_error_{req_id}")
                except ClientDisconnectedError:
                    logger.info(f"[{req_id}] 客户端在调整 Top P 时断开连接。")
                    raise
                except Exception as e_top_p:
                    logger.exception(f"[{req_id}] ❌ 调整 Top P 时发生未知错误")
                    await save_error_snapshot(f"top_p_unknown_error_{req_id}")
                check_client_disconnected("Top P 调整 - 逻辑完成后: ")
        # 参数设置块结束 (async with params_cache_lock)

        logger.info(f"[{req_id}] (Refactored Process) 填充并提交用户提示 ({len(prepared_prompt)} chars)...") # 'prepared_prompt' 现在只包含用户/助手消息
        prompt_textarea_locator = page.locator(PROMPT_TEXTAREA_SELECTOR)
        autosize_wrapper_locator = page.locator('ms-prompt-input-wrapper ms-autosize-textarea')
        try:
            await expect_async(prompt_textarea_locator).to_be_visible(timeout=5000)
            check_client_disconnected("After Input Visible: ")
            logger.info(f"[{req_id}]   - 使用 JavaScript evaluate 填充提示文本...")
            await prompt_textarea_locator.evaluate(
                '''
                (element, text) => {
                    element.value = text;
                    element.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
                }
                ''',
                prepared_prompt
            )
            await autosize_wrapper_locator.evaluate('(element, text) => { element.setAttribute("data-value", text); }', prepared_prompt)
            logger.info(f"[{req_id}]   - JavaScript evaluate 填充完成，data-value 已尝试更新。")
            check_client_disconnected("After Input Fill (evaluate): ")

            logger.info(f"[{req_id}]   - 等待发送按钮启用 (填充提示后)...")
            wait_timeout_ms_submit_enabled = 40000 # 40 seconds
            try:
                # Check disconnect before starting the potentially long wait
                check_client_disconnected("填充提示后等待发送按钮启用 - 前置检查: ")
                await expect_async(submit_button_locator).to_be_enabled(timeout=wait_timeout_ms_submit_enabled)
                logger.info(f"[{req_id}]   - ✅ 发送按钮已启用。")
            except PlaywrightAsyncError as e_pw_enabled:
                logger.error(f"[{req_id}]   - ❌ 等待发送按钮启用超时或错误: {e_pw_enabled}")
                await save_error_snapshot(f"submit_button_enable_timeout_{req_id}")
                raise # Re-raise to be caught by the main try-except block for prompt submission
            except ClientDisconnectedError:
                logger.info(f"[{req_id}] 客户端在等待发送按钮启用时断开连接。")
                raise
            except Exception as e_enable_wait:
                logger.exception(f"[{req_id}]   - ❌ 等待发送按钮启用时发生意外错误。")
                await save_error_snapshot(f"submit_button_enable_unexpected_{req_id}")
                raise

            check_client_disconnected("After Submit Button Enabled (Post-Wait): ")
            await asyncio.sleep(0.3) # Small delay after button is enabled, before pressing shortcut
            check_client_disconnected("After Submit Pre-Shortcut-Delay: ")
            submitted_successfully_via_shortcut = False
            user_prompt_autosize_locator = page.locator('ms-prompt-input-wrapper ms-autosize-textarea').nth(1)
            logger.info(f"[{req_id}]   - 用于快捷键后验证的用户输入区域选择器: nth(1) of 'ms-prompt-input-wrapper ms-autosize-textarea'")
            try:
                host_os_from_launcher = os.environ.get('HOST_OS_FOR_SHORTCUT')
                is_mac_determined = False
                if host_os_from_launcher:
                    logger.info(f"[{req_id}]   - 从启动器环境变量 HOST_OS_FOR_SHORTCUT 获取到操作系统提示: '{host_os_from_launcher}'")
                    if host_os_from_launcher == "Darwin":
                        is_mac_determined = True
                    elif host_os_from_launcher in ["Windows", "Linux"]:
                        is_mac_determined = False
                    else:
                        logger.warning(f"[{req_id}]   - 未知的 HOST_OS_FOR_SHORTCUT 值: '{host_os_from_launcher}'。将回退到浏览器检测。")
                        host_os_from_launcher = None
                if not host_os_from_launcher:
                    if host_os_from_launcher is None:
                        logger.info(f"[{req_id}]   - HOST_OS_FOR_SHORTCUT 未设置或值未知，将进行浏览器内部操作系统检测。")
                    user_agent_data_platform = None
                    try:
                        user_agent_data_platform = await page.evaluate("() => navigator.userAgentData?.platform || ''")
                    except Exception as e_ua_data:
                        logger.warning(f"[{req_id}]   - navigator.userAgentData.platform 读取失败 ({e_ua_data})，尝试 navigator.userAgent。")
                        user_agent_string = await page.evaluate("() => navigator.userAgent || ''")
                        user_agent_string_lower = user_agent_string.lower()
                        if "macintosh" in user_agent_string_lower or "mac os x" in user_agent_string_lower or "macintel" in user_agent_string_lower:
                            user_agent_data_platform = "macOS"
                        elif "windows" in user_agent_string_lower:
                            user_agent_data_platform = "Windows"
                        elif "linux" in user_agent_string_lower:
                            user_agent_data_platform = "Linux"
                        else:
                            user_agent_data_platform = "Other"
                    if user_agent_data_platform and user_agent_data_platform != "Other":
                        user_agent_data_platform_lower = user_agent_data_platform.lower()
                        is_mac_determined = "mac" in user_agent_data_platform_lower or "macos" in user_agent_data_platform_lower or "macintel" in user_agent_data_platform_lower
                        logger.info(f"[{req_id}]   - 浏览器内部检测到平台: '{user_agent_data_platform}', 推断 is_mac: {is_mac_determined}")
                    else:
                        logger.warning(f"[{req_id}]   - 浏览器平台信息获取失败、为空或为'Other' ('{user_agent_data_platform}')。默认使用非Mac快捷键。")
                        is_mac_determined = False
                shortcut_modifier = "Meta" if is_mac_determined else "Control"
                shortcut_key = "Enter"
                logger.info(f"[{req_id}]   - 最终选择快捷键: {shortcut_modifier}+{shortcut_key} (基于 is_mac_determined: {is_mac_determined})")
                logger.info(f"[{req_id}]   - 尝试将焦点设置到输入框...")
                await prompt_textarea_locator.focus(timeout=5000)
                check_client_disconnected("After Input Focus (Shortcut): ")
                await asyncio.sleep(0.1)
                logger.info(f"[{req_id}]   - 焦点设置完成，准备按下快捷键...")
                try:
                    await page.keyboard.press(f'{shortcut_modifier}+{shortcut_key}')
                    logger.info(f"[{req_id}]   - 已使用组合键方式模拟按下: {shortcut_modifier}+{shortcut_key}")
                except Exception as combo_err:
                    logger.warning(f"[{req_id}]   - 组合键方式失败: {combo_err}，尝试分步按键...")
                    try:
                        await page.keyboard.down(shortcut_modifier)
                        await asyncio.sleep(0.05)
                        await page.keyboard.down(shortcut_key)
                        await asyncio.sleep(0.05)
                        await page.keyboard.up(shortcut_key)
                        await asyncio.sleep(0.05)
                        await page.keyboard.up(shortcut_modifier)
                        logger.info(f"[{req_id}]   - 已使用分步按键方式模拟: {shortcut_modifier}+{shortcut_key}")
                    except Exception as step_err:
                        logger.error(f"[{req_id}]   - 分步按键也失败: {step_err}")
                check_client_disconnected("After Keyboard Press: ")
                await asyncio.sleep(0.75) # <--- 新增此行以提供UI反应时间
                check_client_disconnected("After Keyboard Press Post-Delay: ") # <--- 新增此行日志
                user_prompt_actual_textarea_locator = page.locator(
                    'ms-prompt-input-wrapper textarea[aria-label="Start typing a prompt"]'
                )
                selector_string = 'ms-prompt-input-wrapper textarea[aria-label="Start typing a prompt"]'
                logger.info(f"[{req_id}]   - 用于快捷键后验证的用户输入 textarea 选择器: '{selector_string}'")
                validation_attempts = 7
                validation_interval = 0.2
                for i in range(validation_attempts):
                    try:
                        current_value = await user_prompt_actual_textarea_locator.input_value(timeout=500)
                        if current_value == "":
                            submitted_successfully_via_shortcut = True
                            logger.info(f"[{req_id}]   - ✅ 快捷键提交成功确认 (用户输入 textarea value 已清空 after {i+1} attempts)。")
                            break
                        else:
                            if DEBUG_LOGS_ENABLED:
                                logger.debug(f"[{req_id}]   - 用户输入 textarea value 验证尝试 {i+1}/{validation_attempts}: 当前='{current_value}', 期望=''")
                    except PlaywrightAsyncError as e_val:
                        if DEBUG_LOGS_ENABLED:
                            logger.debug(f"[{req_id}]   - 获取用户输入 textarea value 时出错 (尝试 {i+1}): {e_val.message.splitlines()[0]}")
                        if "timeout" in e_val.message.lower():
                            pass
                        else:
                            logger.warning(f"[{req_id}]   - 获取用户输入 textarea value 时 Playwright 错误 (尝试 {i+1}): {e_val.message.splitlines()[0]}")
                            if "strict mode violation" in e_val.message.lower():
                                await save_error_snapshot(f"shortcut_submit_textarea_value_strict_error_{req_id}")
                                break
                            break
                    except Exception as e_gen:
                        logger.warning(f"[{req_id}]   - 获取用户输入 textarea value 时发生其他错误 (尝试 {i+1}): {e_gen}")
                        break
                    if i < validation_attempts - 1:
                        await asyncio.sleep(validation_interval)
                if not submitted_successfully_via_shortcut:
                    final_value_for_log = "(无法获取或未清空)"
                    try:
                        final_value_for_log = await user_prompt_actual_textarea_locator.input_value(timeout=300)
                    except:
                        pass
                    logger.warning(f"[{req_id}]   - ⚠️ 快捷键提交后用户输入 textarea value ('{final_value_for_log}') 未在预期时间内 ({validation_attempts * validation_interval:.1f}s) 清空。")
            except Exception as shortcut_err:
                logger.error(f"[{req_id}]   - ❌ 快捷键提交过程中发生错误: {shortcut_err}", exc_info=True)
                await save_error_snapshot(f"shortcut_submit_error_{req_id}")
                raise PlaywrightAsyncError(f"Failed to submit prompt via keyboard shortcut: {shortcut_err}") from shortcut_err
            if not submitted_successfully_via_shortcut:
                 logger.error(f"[{req_id}] 严重错误: 未能通过快捷键确认提交。")
                 raise PlaywrightAsyncError("Failed to confirm prompt submission via shortcut.")
        except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as submit_err:
            if isinstance(submit_err, ClientDisconnectedError): raise
            logger.error(f"[{req_id}] ❌ 错误: 填充或提交提示时出错: {submit_err}", exc_info=True)
            await save_error_snapshot(f"submit_prompt_error_{req_id}")
            raise HTTPException(status_code=502, detail=f"[{req_id}] Failed to submit prompt to AI Studio: {submit_err}")
        except Exception as submit_exc:
            logger.exception(f"[{req_id}] ❌ 错误: 填充或提交提示时意外错误")
            await save_error_snapshot(f"submit_prompt_unexpected_{req_id}")
            raise HTTPException(status_code=500, detail=f"[{req_id}] Unexpected error during prompt submission: {submit_exc}")
        check_client_disconnected("After Submit Logic: ")

        stream_port = os.environ.get('STREAM_PORT')
        use_stream = stream_port != '0' # 判断是否使用你的辅助流

        if use_stream:
            # 确保 generate_random_string 函数已定义或可访问
            def generate_random_string(length):
                charset = "abcdefghijklmnopqrstuvwxyz0123456789"
                return ''.join(random.choice(charset) for _ in range(length))

            if is_streaming:
                try:
                    completion_event = Event()
                    # 确保 create_stream_generator_from_helper 函数已定义或可访问
                    async def create_stream_generator_from_helper(event_to_set: Event) -> AsyncGenerator[str, None]:
                        last_reason_pos = 0
                        last_body_pos = 0
                        # 使用当前AI Studio模型ID或默认模型名称
                        model_name_for_stream = current_ai_studio_model_id or MODEL_NAME
                        chat_completion_id = f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}-{random.randint(100, 999)}"
                        created_timestamp = int(time.time())

                        async for data in use_stream_response(req_id): # 确保 use_stream_response 是异步生成器
                            # --- 开始处理从 use_stream_response 获取的 data ---
                            # (这里是你现有的解析 data 并生成 SSE 块的逻辑)
                            # 例如:
                            if len(data["reason"]) > last_reason_pos:
                                output = {
                                    "id": chat_completion_id,
                                    "object": "chat.completion.chunk",
                                    "model": model_name_for_stream,
                                    "created": created_timestamp,
                                    "choices":[{
                                        "delta":{
                                            "role": "assistant",
                                            "content": None,
                                            "reasoning_content": data["reason"][last_reason_pos:],
                                        },
                                        "finish_reason": None,
                                        "native_finish_reason": None, # 保持与OpenAI兼容
                                    }]
                                }
                                last_reason_pos = len(data["reason"])
                                yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                            elif len(data["body"]) > last_body_pos:
                                finish_reason_val = None
                                if data["done"]:
                                    finish_reason_val = "stop"
                                
                                delta_content = {"role": "assistant", "content": data["body"][last_body_pos:]}
                                choice_item = {
                                    "delta": delta_content,
                                    "finish_reason": finish_reason_val,
                                    "native_finish_reason": finish_reason_val,
                                }

                                if data["done"] and data.get("function") and len(data["function"]) > 0:
                                    tool_calls_list = []
                                    for func_idx, function_call_data in enumerate(data["function"]):
                                        tool_calls_list.append({
                                            "id": f"call_{generate_random_string(24)}", # 确保ID唯一
                                            "index": func_idx, # 使用实际索引
                                            "type": "function",
                                            "function": {
                                                "name": function_call_data["name"],
                                                "arguments": json.dumps(function_call_data["params"]),
                                            },
                                        })
                                    delta_content["tool_calls"] = tool_calls_list
                                    # 如果有工具调用，finish_reason 应该是 tool_calls
                                    choice_item["finish_reason"] = "tool_calls"
                                    choice_item["native_finish_reason"] = "tool_calls"
                                    # 根据OpenAI规范，当有tool_calls时，content通常为null
                                    delta_content["content"] = None


                                output = {
                                    "id": chat_completion_id,
                                    "object": "chat.completion.chunk",
                                    "model": model_name_for_stream,
                                    "created": created_timestamp,
                                    "choices": [choice_item]
                                }
                                last_body_pos = len(data["body"])
                                yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                            elif data["done"]: # 处理仅 'done' 为 true 的情况，可能包含函数调用但无新内容
                                delta_content = {"role": "assistant"} # 至少需要 role
                                choice_item = {
                                    "delta": delta_content,
                                    "finish_reason": "stop",
                                    "native_finish_reason": "stop",
                                }

                                if data.get("function") and len(data["function"]) > 0:
                                    tool_calls_list = []
                                    for func_idx, function_call_data in enumerate(data["function"]):
                                        tool_calls_list.append({
                                            "id": f"call_{generate_random_string(24)}",
                                            "index": func_idx,
                                            "type": "function",
                                            "function": {
                                                "name": function_call_data["name"],
                                                "arguments": json.dumps(function_call_data["params"]),
                                            },
                                        })
                                    delta_content["tool_calls"] = tool_calls_list
                                    choice_item["finish_reason"] = "tool_calls"
                                    choice_item["native_finish_reason"] = "tool_calls"
                                    delta_content["content"] = None # 有 tool_calls 时 content 为 null

                                output = {
                                    "id": chat_completion_id,
                                    "object": "chat.completion.chunk",
                                    "model": model_name_for_stream,
                                    "created": created_timestamp,
                                    "choices": [choice_item]
                                }
                                yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        # --- 结束处理从 use_stream_response 获取的 data ---
                        
                        yield "data: [DONE]\n\n" # 确保发送最终的 [DONE] 标记

                        if not event_to_set.is_set():
                            event_to_set.set()

                    stream_gen_func = create_stream_generator_from_helper(completion_event)
                    if not result_future.done():
                        result_future.set_result(StreamingResponse(stream_gen_func, media_type="text/event-stream"))
                    else: # 如果 future 已经完成（例如，被取消）
                        if not completion_event.is_set(): completion_event.set() # 确保事件被设置
                    
                    # 修改后的返回语句:
                    return completion_event, submit_button_locator, check_client_disconnected

                except Exception as e:
                    logger.error(f"[{req_id}] (Stream Gen) 从队列获取流式数据时出错: {e}", exc_info=True) # 添加 exc_info
                    # 如果在流生成过程中出错，确保 completion_event 被设置，以防 worker 卡住
                    if completion_event and not completion_event.is_set():
                        completion_event.set()
                    # 此处错误处理：当前代码会将 use_stream 设为 False 并尝试回退到 Playwright 交互。
                    # 如果辅助流是主要方式且失败，可能直接抛出错误更合适，而不是静默回退。
                    # 但根据现有逻辑，我们保持回退。
                    use_stream = False
                    logger.warning(f"[{req_id}] 辅助流处理失败，将尝试回退到 Playwright 页面交互（如果适用）。")


            else: # 非流式辅助路径 (use_stream 为 True, is_streaming 为 False)
                content = None
                reasoning_content = None
                functions = None
                final_data_from_aux_stream = None # 初始化

                # 确保 use_stream_response 是异步迭代器
                async for data in use_stream_response(req_id): # 传递 req_id
                    check_client_disconnected(f"非流式辅助流 - 循环中 ({req_id}): ")

                    final_data_from_aux_stream = data # 存储最后收到的数据
                    if data.get("done"): # 对于非流式，我们期望一个包含所有数据的 "done" 消息
                        content = data.get("body") # 使用 .get() 避免 KeyError
                        reasoning_content = data.get("reason")
                        functions = data.get("function")
                        break # 获取到数据后即中断
                
                if final_data_from_aux_stream and final_data_from_aux_stream.get("reason") == "internal_timeout":
                    logger.error(f"[{req_id}] Non-streaming request via auxiliary stream failed: Internal Timeout from aux stream.")
                    #确保 HTTPException 已导入: from fastapi import HTTPException
                    raise HTTPException(status_code=502, detail=f"[{req_id}] Auxiliary stream processing error for non-streaming request (internal_timeout).")

                if final_data_from_aux_stream and final_data_from_aux_stream.get("done") is True and content is None and final_data_from_aux_stream.get("reason") != "internal_timeout":
                     logger.error(f"[{req_id}] Non-streaming request via auxiliary stream finished but provided no content. Reason: {final_data_from_aux_stream.get('reason')}.")
                     raise HTTPException(status_code=502, detail=f"[{req_id}] Auxiliary stream finished for non-streaming request but provided no content (Reason: {final_data_from_aux_stream.get('reason')}).")

                model_name_for_json = current_ai_studio_model_id or MODEL_NAME
                message_payload = {"role": "assistant", "content": content}
                finish_reason_val = "stop"

                if functions and len(functions) > 0:
                    tool_calls_list = []
                    for func_idx, function_call_data in enumerate(functions):
                        tool_calls_list.append({
                            "id": f"call_{generate_random_string(24)}",
                            "index": func_idx,
                            "type": "function",
                            "function": {
                                "name": function_call_data["name"],
                                "arguments": json.dumps(function_call_data["params"]),
                            },
                        })
                    message_payload["tool_calls"] = tool_calls_list
                    finish_reason_val = "tool_calls"
                    # 当有 tool_calls 时，OpenAI 规范通常将 content 设为 null
                    message_payload["content"] = None
                
                if reasoning_content: # 如果有思考过程内容，也加入到 message 中
                    message_payload["reasoning_content"] = reasoning_content


                response_payload = {
                    "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}",
                    "object": "chat.completion", "created": int(time.time()),
                    "model": model_name_for_json,
                    "choices": [{
                        "index": 0,
                        "message": message_payload,
                        "finish_reason": finish_reason_val,
                        "native_finish_reason": finish_reason_val, # 添加 native_finish_reason
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # 伪使用数据
                }

                if not result_future.done():
                    result_future.set_result(JSONResponse(content=response_payload))
                return None # 非流式请求返回 None


        if not use_stream:
            logger.info(f"[{req_id}] (Refactored Process) 定位响应元素...")
            response_container = page.locator(RESPONSE_CONTAINER_SELECTOR).last
            response_element = response_container.locator(RESPONSE_TEXT_SELECTOR)
            try:
                await expect_async(response_container).to_be_attached(timeout=20000)
                check_client_disconnected("After Response Container Attached: ")
                await expect_async(response_element).to_be_attached(timeout=90000)
                logger.info(f"[{req_id}]   - 响应元素已定位。")
            except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as locate_err:
                if isinstance(locate_err, ClientDisconnectedError): raise
                logger.error(f"[{req_id}] ❌ 错误: 定位响应元素失败或超时: {locate_err}")
                await save_error_snapshot(f"response_locate_error_{req_id}")
                raise HTTPException(status_code=502, detail=f"[{req_id}] Failed to locate AI Studio response element: {locate_err}")
            except Exception as locate_exc:
                logger.exception(f"[{req_id}] ❌ 错误: 定位响应元素时意外错误")
                await save_error_snapshot(f"response_locate_unexpected_{req_id}")
                raise HTTPException(status_code=500, detail=f"[{req_id}] Unexpected error locating response element: {locate_exc}")
            check_client_disconnected("After Locate Response: ")

            # --- MERGED: Helper logic integration ---
            use_helper = False
            helper_endpoint = os.environ.get('HELPER_ENDPOINT')
            helper_sapisid = os.environ.get('HELPER_SAPISID')
            if helper_endpoint and helper_sapisid:
                logger.info(f"[{req_id}] 检测到 Helper 配置，将尝试使用 Helper 服务获取响应。")
                use_helper = True
            else:
                logger.info(f"[{req_id}] 未检测到完整的 Helper 配置，将使用 Playwright 页面交互获取响应。")

            if use_helper and (not use_stream):
                try:
                    if is_streaming:
                        completion_event = Event()
                        async def create_stream_generator_from_helper(event_to_set: Event) -> AsyncGenerator[str, None]:
                            try:
                                async for data_chunk in use_helper_get_response(helper_endpoint, helper_sapisid):
                                    if client_disconnected_event.is_set():
                                        logger.info(f"[{req_id}] (Helper Stream Gen) 客户端断开，停止。")
                                        break
                                    if data_chunk == "[ERROR]": # Helper indicated an error
                                        logger.error(f"[{req_id}] (Helper Stream Gen) Helper 服务返回错误信号。")
                                        yield generate_sse_error_chunk("Helper service reported an error.", req_id, "helper_error")
                                        break
                                    if data_chunk == "[DONE]": # Helper indicated completion
                                        logger.info(f"[{req_id}] (Helper Stream Gen) Helper 服务指示完成。")
                                        break
                                    yield f"data: {data_chunk}\n\n" # Assume helper sends pre-formatted SSE data chunks
                                yield "data: [DONE]\n\n" # Ensure final DONE is sent
                            except Exception as e_helper_stream:
                                logger.error(f"[{req_id}] (Helper Stream Gen) 从 Helper 获取流式数据时出错: {e_helper_stream}", exc_info=True)
                                yield generate_sse_error_chunk(f"Error streaming from helper: {e_helper_stream}", req_id)
                                yield "data: [DONE]\n\n"
                            finally:
                                if not event_to_set.is_set(): event_to_set.set()

                        stream_gen_func = create_stream_generator_from_helper(completion_event)
                        if not result_future.done():
                            result_future.set_result(StreamingResponse(stream_gen_func, media_type="text/event-stream"))
                        else:
                            if not completion_event.is_set(): completion_event.set() # Ensure event is set if future already done
                        return completion_event # Return the event for the worker to wait on
                    else: # Non-streaming with helper
                        full_response_content = ""
                        think_content = ""
                        body_content = ""
                        async for data_chunk in use_helper_get_response(helper_endpoint, helper_sapisid):
                            if data_chunk == "[ERROR]":
                                raise HTTPException(status_code=502, detail=f"[{req_id}] Helper service reported an error during non-streaming fetch.")
                            if data_chunk == "[DONE]":
                                break
                            try:
                                # Assuming helper sends OpenAI-like delta chunks even for non-streaming,
                                # and we need to aggregate them.
                                stream_data = json.loads(data_chunk)
                                if "choices" in stream_data and stream_data["choices"]:
                                    delta = stream_data["choices"][0].get("delta", {})
                                    if "reasoning_content" in delta: # Example for structured content
                                        think_content += delta["reasoning_content"]
                                    elif "content" in delta:
                                        body_content += delta["content"]
                            except json.JSONDecodeError:
                                logger.warning(f"[{req_id}] (Helper Non-Stream) 无法解析来自 Helper 的 JSON 数据块: {data_chunk}")
                                body_content += data_chunk # Fallback: append raw if not JSON

                        if think_content:
                            full_response_content = f"<think>{think_content}</think>\n{body_content}"
                        else:
                            full_response_content = body_content

                        response_payload = {
                            "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}",
                            "object": "chat.completion", "created": int(time.time()), "model": MODEL_NAME,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_response_content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        }
                        if not result_future.done():
                            result_future.set_result(JSONResponse(content=response_payload))
                        return None # No event for non-streaming
                except Exception as e_helper:
                    logger.error(f"[{req_id}] 使用 Helper 服务时发生错误: {e_helper}。将回退到 Playwright 页面交互。", exc_info=True)
                    use_helper = False # Fallback to Playwright

            # --- Fallback to Playwright page interaction if helper is not used or failed ---
            if (not use_helper) and (not use_stream):
                logger.info(f"[{req_id}] (Refactored Process) 等待响应生成完成或检测模型错误...")
                MODEL_ERROR_CONTAINER_SELECTOR = 'ms-chat-turn:last-child div.model-error'
                completion_detected_via_edit_button = False
                page_model_error_message: Optional[str] = None
                completion_detected_via_edit_button = await _wait_for_response_completion(
                    page,
                    input_field_locator,
                    submit_button_locator,
                    page.locator(EDIT_MESSAGE_BUTTON_SELECTOR), # edit_button_locator
                    req_id, # req_id for _wait_for_response_completion
                    check_client_disconnected, # check_client_disconnected_func
                    req_id, # current_chat_id
                )
                check_client_disconnected("After _wait_for_response_completion attempt: ")
                if not completion_detected_via_edit_button:
                    logger.info(f"[{req_id}] _wait_for_response_completion 未通过编辑按钮确认完成，检查是否存在模型错误...")
                    try:
                        error_container_locator = page.locator(MODEL_ERROR_CONTAINER_SELECTOR)
                        await expect_async(error_container_locator).to_be_visible(timeout=2000)
                        specific_error_text_locator = error_container_locator.locator('*:not(mat-icon)')
                        try:
                            page_model_error_message = await specific_error_text_locator.first.text_content(timeout=500)
                            if page_model_error_message: page_model_error_message = page_model_error_message.strip()
                        except PlaywrightAsyncError:
                            page_model_error_message = await error_container_locator.text_content(timeout=500)
                            if page_model_error_message: page_model_error_message = page_model_error_message.strip()
                        if page_model_error_message:
                            logger.error(f"[{req_id}] ❌ 检测到 AI Studio 模型返回的错误信息: {page_model_error_message}")
                            await save_error_snapshot(f"model_returned_error_{req_id}")
                            raise HTTPException(status_code=502, detail=f"[{req_id}] AI Studio Model Error: {page_model_error_message}")
                        else:
                            logger.warning(f"[{req_id}] 检测到 model-error 容器，但未能提取具体错误文本。")
                            await save_error_snapshot(f"model_error_container_no_text_{req_id}")
                            raise HTTPException(status_code=502, detail=f"[{req_id}] AI Studio returned an unspecified model error (error container found).")
                    except (PlaywrightAsyncError, asyncio.TimeoutError) as e_model_err_check:
                        logger.info(f"[{req_id}] 未检测到明确的 model-error 容器 (或检查超时: {type(e_model_err_check).__name__})。继续按原超时逻辑处理。")
                        if not completion_detected_via_edit_button:
                             raise HTTPException(status_code=504, detail=f"[{req_id}] AI Studio response generation timed out (and no specific model error detected).")
                if not completion_detected_via_edit_button:
                    logger.info(f"[{req_id}] (Refactored Process) 检查页面 Toast 错误提示...")
                    page_toast_error = await detect_and_extract_page_error(page, req_id)
                    if page_toast_error:
                        logger.error(f"[{req_id}] ❌ 错误: AI Studio 页面返回 Toast 错误: {page_toast_error}")
                        await save_error_snapshot(f"page_toast_error_detected_{req_id}")
                        raise HTTPException(status_code=502, detail=f"[{req_id}] AI Studio Page Error: {page_toast_error}")
                    check_client_disconnected("After Page Toast Error Check: ")
                else:
                    logger.info(f"[{req_id}] 已通过编辑按钮确认完成，跳过 Toast 错误检查。")
                if not completion_detected_via_edit_button:
                    logger.error(f"[{req_id}] 逻辑异常：响应未完成，也未检测到模型错误，但不应到达此处获取内容。")
                    raise HTTPException(status_code=500, detail=f"[{req_id}] Internal logic error in response processing.")
                logger.info(f"[{req_id}] (Refactored Process) 获取最终响应内容...")
                final_content = await _get_final_response_content(
                    page, req_id, check_client_disconnected
                )
                if final_content is None:
                    try:
                        error_container_locator = page.locator(MODEL_ERROR_CONTAINER_SELECTOR)
                        if await error_container_locator.is_visible(timeout=500):
                            late_error_message = await error_container_locator.text_content(timeout=300) or "Unknown model error after content fetch attempt."
                            logger.error(f"[{req_id}] 获取内容失败后，检测到延迟出现的模型错误: {late_error_message.strip()}")
                            raise HTTPException(status_code=502, detail=f"[{req_id}] AI Studio Model Error (detected after content fetch failure): {late_error_message.strip()}")
                    except:
                        pass
                    raise HTTPException(status_code=500, detail=f"[{req_id}] Failed to extract final response content from AI Studio.")
                check_client_disconnected("After Get Content: ")
                logger.info(f"[{req_id}] (Refactored Process) 格式化并设置结果 (模式: {'流式' if is_streaming else '非流式'})...")
                if is_streaming:
                    completion_event = Event()
                    async def create_stream_generator(event_to_set: Event, content_to_stream: str) -> AsyncGenerator[str, None]:
                        logger.info(f"[{req_id}] (Stream Gen) 开始伪流式输出 ({len(content_to_stream)} chars)...")
                        try:
                            total_chars = len(content_to_stream)
                            chunk_size = 5
                            for i in range(0, total_chars, chunk_size):
                                if client_disconnected_event.is_set():
                                    logger.info(f"[{req_id}] (Stream Gen) 断开连接，停止。")
                                    break
                                chunk = content_to_stream[i:i + chunk_size]
                                if not chunk:
                                    continue
                                yield generate_sse_chunk(chunk, req_id, MODEL_NAME)
                                await asyncio.sleep(PSEUDO_STREAM_DELAY)
                            yield generate_sse_stop_chunk(req_id, MODEL_NAME)
                            yield "data: [DONE]\n\n"
                            logger.info(f"[{req_id}] (Stream Gen) ✅ 伪流式响应发送完毕。")
                        except asyncio.CancelledError:
                            logger.info(f"[{req_id}] (Stream Gen) 流生成器被取消。")
                        except Exception as e:
                            logger.exception(f"[{req_id}] (Stream Gen) ❌ 伪流式生成过程中出错")
                            try: yield generate_sse_error_chunk(f"Stream generation error: {e}", req_id); yield "data: [DONE]\n\n"
                            except: pass
                        finally:
                            logger.info(f"[{req_id}] (Stream Gen) 设置完成事件。")
                            if not event_to_set.is_set(): event_to_set.set()
                    stream_generator_func = create_stream_generator(completion_event, final_content)
                    if not result_future.done():
                        result_future.set_result(StreamingResponse(stream_generator_func, media_type="text/event-stream"))
                        logger.info(f"[{req_id}] (Refactored Process) 流式响应生成器已设置。")
                    else:
                        logger.warning(f"[{req_id}] (Refactored Process) Future 已完成/取消，无法设置流式结果。")
                        if not completion_event.is_set(): completion_event.set()
                    return completion_event
                else:
                    response_payload = {
                        "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}",
                        "object": "chat.completion", "created": int(time.time()), "model": MODEL_NAME,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
                    if not result_future.done():
                        result_future.set_result(JSONResponse(content=response_payload))
                        logger.info(f"[{req_id}] (Refactored Process) 非流式 JSON 响应已设置。")
                    else:
                        logger.warning(f"[{req_id}] (Refactored Process) Future 已完成/取消，无法设置 JSON 结果。")
                    return None
    except ClientDisconnectedError as disco_err:
        logger.info(f"[{req_id}] (Refactored Process) 捕获到客户端断开连接信号: {disco_err}")
        if not result_future.done():
             result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] Client disconnected during processing."))
    except HTTPException as http_err:
        logger.warning(f"[{req_id}] (Refactored Process) 捕获到 HTTP 异常: {http_err.status_code} - {http_err.detail}")
        if not result_future.done(): result_future.set_exception(http_err)
    except PlaywrightAsyncError as pw_err:
        logger.error(f"[{req_id}] (Refactored Process) 捕获到 Playwright 错误: {pw_err}")
        await save_error_snapshot(f"process_playwright_error_{req_id}")
        if not result_future.done(): result_future.set_exception(HTTPException(status_code=502, detail=f"[{req_id}] Playwright interaction failed: {pw_err}"))
    except asyncio.TimeoutError as timeout_err:
        logger.error(f"[{req_id}] (Refactored Process) 捕获到操作超时: {timeout_err}")
        await save_error_snapshot(f"process_timeout_error_{req_id}")
        if not result_future.done(): result_future.set_exception(HTTPException(status_code=504, detail=f"[{req_id}] Operation timed out: {timeout_err}"))
    except asyncio.CancelledError:
        logger.info(f"[{req_id}] (Refactored Process) 任务被取消。")
        if not result_future.done(): result_future.cancel("Processing task cancelled")
    except Exception as e:
        logger.exception(f"[{req_id}] (Refactored Process) 捕获到意外错误")
        await save_error_snapshot(f"process_unexpected_error_{req_id}")
        if not result_future.done(): result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Unexpected server error: {e}"))
    finally:
        if disconnect_check_task and not disconnect_check_task.done():
            disconnect_check_task.cancel()
            try: await disconnect_check_task
            except asyncio.CancelledError: pass
            except Exception as task_clean_err: logger.error(f"[{req_id}] 清理任务时出错: {task_clean_err}")
        logger.info(f"[{req_id}] (Refactored Process) 处理完成。")
        if is_streaming and completion_event and not completion_event.is_set() and (result_future.done() and result_future.exception() is not None):
             logger.warning(f"[{req_id}] (Refactored Process) 流式请求异常，确保完成事件已设置。")
             completion_event.set()
        return completion_event, submit_button_locator, check_client_disconnected

# --- Main Chat Endpoint ---
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    req_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=7))
    logger.info(f"[{req_id}] 收到 /v1/chat/completions 请求 (Stream={request.stream})")
    logger.debug(f"[{req_id}] 完整请求参数: {request.model_dump_json(indent=2)}")
    launch_mode = os.environ.get('LAUNCH_MODE', 'unknown')
    browser_page_critical = launch_mode != "direct_debug_no_browser"
    service_unavailable = is_initializing or \
                          not is_playwright_ready or \
                          (browser_page_critical and (not is_page_ready or not is_browser_connected)) or \
                          not worker_task or worker_task.done()
    if service_unavailable:
        status_code = 503
        error_details = []
        if is_initializing: error_details.append("初始化进行中")
        if not is_playwright_ready: error_details.append("Playwright 未就绪")
        if browser_page_critical:
            if not is_browser_connected: error_details.append("浏览器未连接")
            if not is_page_ready: error_details.append("页面未就绪")
        if not worker_task or worker_task.done(): error_details.append("Worker 未运行")
        detail = f"[{req_id}] 服务当前不可用 ({', '.join(error_details)}). 请稍后重试."
        logger.error(f"[{req_id}] 服务不可用详情: {detail}")
        raise HTTPException(status_code=status_code, detail=detail, headers={"Retry-After": "30"})
    result_future = Future()
    request_item = {
        "req_id": req_id, "request_data": request, "http_request": http_request,
        "result_future": result_future, "enqueue_time": time.time(), "cancelled": False
    }
    await request_queue.put(request_item)
    logger.info(f"[{req_id}] 请求已加入队列 (当前队列长度: {request_queue.qsize()})")
    try:
        timeout_seconds = RESPONSE_COMPLETION_TIMEOUT / 1000 + 120
        result = await asyncio.wait_for(result_future, timeout=timeout_seconds)
        logger.info(f"[{req_id}] Worker 处理完成，返回结果。")
        return result
    except asyncio.TimeoutError:
        logger.error(f"[{req_id}] ❌ 等待 Worker 响应超时 ({timeout_seconds}s)。")
        raise HTTPException(status_code=504, detail=f"[{req_id}] Request processing timed out waiting for worker response.")
    except asyncio.CancelledError:
        logger.info(f"[{req_id}] 请求 Future 被取消 (可能由客户端断开连接触发)。")
        if not result_future.done() or result_future.exception() is None:
             raise HTTPException(status_code=499, detail=f"[{req_id}] Request cancelled by client or server.")
        else:
             raise result_future.exception()
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.exception(f"[{req_id}] ❌ 等待 Worker 响应时发生意外错误")
        raise HTTPException(status_code=500, detail=f"[{req_id}] Unexpected error waiting for worker response: {e}")

# --- Cancel Request Endpoint ---
async def cancel_queued_request(req_id: str) -> bool:
    cancelled = False
    items_to_requeue = []
    found = False
    try:
        while True:
            item = request_queue.get_nowait()
            if item.get("req_id") == req_id and not item.get("cancelled", False):
                logger.info(f"[{req_id}] 在队列中找到请求，标记为已取消。")
                item["cancelled"] = True
                item_future = item.get("result_future")
                if item_future and not item_future.done():
                    item_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] Request cancelled by API call."))
                items_to_requeue.append(item)
                cancelled = True
                found = True
            else:
                items_to_requeue.append(item)
    except asyncio.QueueEmpty:
        pass
    finally:
        for item in items_to_requeue:
            await request_queue.put(item)
    return cancelled

@app.post("/v1/cancel/{req_id}", dependencies=[Depends(verify_api_key)])
async def cancel_request(req_id: str):
    logger.info(f"[{req_id}] 收到取消请求。")
    cancelled = await cancel_queued_request(req_id)
    if cancelled:
        return JSONResponse(content={"success": True, "message": f"Request {req_id} marked as cancelled in queue."})
    else:
        return JSONResponse(
            content={"success": False, "message": f"Request {req_id} not found in queue (it might be processing or already finished)."},
            status_code=404
        )

@app.get("/v1/queue")
async def get_queue_status():
    queue_items = []
    items_to_requeue = []
    try:
        while True:
            item = request_queue.get_nowait()
            items_to_requeue.append(item)
            req_id = item.get("req_id", "unknown")
            timestamp = item.get("enqueue_time", 0)
            is_streaming = item.get("request_data").stream if hasattr(item.get("request_data", {}), "stream") else False
            cancelled = item.get("cancelled", False)
            queue_items.append({
                "req_id": req_id, "enqueue_time": timestamp,
                "wait_time_seconds": round(time.time() - timestamp, 2) if timestamp else None,
                "is_streaming": is_streaming, "cancelled": cancelled
            })
    except asyncio.QueueEmpty:
        pass
    finally:
        for item in items_to_requeue:
            await request_queue.put(item)
    return JSONResponse(content={
        "queue_length": len(queue_items),
        "is_processing_locked": processing_lock.locked(),
        "items": sorted(queue_items, key=lambda x: x.get("enqueue_time", 0))
    })

@app.websocket("/ws/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    if not log_ws_manager:
        try:
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "type": "error", "status": "disconnected",
                "message": "日志服务内部错误 (管理器未初始化)。",
                "timestamp": datetime.datetime.now().isoformat()}))
            await websocket.close(code=1011)
        except Exception: pass
        return
    client_id = str(uuid.uuid4())
    try:
        await log_ws_manager.connect(client_id, websocket)
        while True:
            data = await websocket.receive_text()
            if data.lower() == "ping":
                 await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.datetime.now().isoformat()}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"日志 WebSocket (客户端 {client_id}) 发生异常: {e}", exc_info=True)
    finally:
        if log_ws_manager:
            log_ws_manager.disconnect(client_id)

# --- Main Guard ---
if __name__ == "__main__":
    print("错误: server.py 不应直接作为主脚本运行。", file=sys.stderr)
    print("请使用 launch_camoufox.py (用于调试) 或 start.py (用于后台服务) 来启动。", file=sys.stderr)
    print("\n如果确实需要直接运行 server.py 进行底层测试 (不推荐):", file=sys.stderr)
    print("  1. 确保已设置必要的环境变量，如 CAMOUFOX_WS_ENDPOINT, LAUNCH_MODE, SERVER_REDIRECT_PRINT, SERVER_LOG_LEVEL。", file=sys.stderr)
    print("  2. 然后可以尝试: python -m uvicorn server:app --host 0.0.0.0 --port <端口号>", file=sys.stderr)
    print("     例如: LAUNCH_MODE=direct_debug_no_browser SERVER_REDIRECT_PRINT=false python -m uvicorn server:app --port 8000", file=sys.stderr)
    sys.exit(1)

# --- Model Switching Helper ---
async def switch_ai_studio_model(page: AsyncPage, model_id: str, req_id: str) -> bool:
    logger.info(f"[{req_id}] 开始切换模型到: {model_id}")
    original_prefs_str: Optional[str] = None
    original_prompt_model: Optional[str] = None
    new_chat_url = f"https://{AI_STUDIO_URL_PATTERN}prompts/new_chat"
    try:
        original_prefs_str = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
        if original_prefs_str:
            try:
                original_prefs_obj = json.loads(original_prefs_str)
                original_prompt_model = original_prefs_obj.get("promptModel")
                logger.info(f"[{req_id}] 切换前 localStorage.promptModel 为: {original_prompt_model or '未设置'}")
            except json.JSONDecodeError:
                logger.warning(f"[{req_id}] 无法解析原始的 aiStudioUserPreference JSON 字符串。")
                original_prefs_str = None
        current_prefs_for_modification = json.loads(original_prefs_str) if original_prefs_str else {}
        full_model_path = f"models/{model_id}"
        if current_prefs_for_modification.get("promptModel") == full_model_path:
            logger.info(f"[{req_id}] 模型已经设置为 {model_id} (localStorage 中已是目标值)，无需切换")
            if page.url != new_chat_url:
                 logger.info(f"[{req_id}] 当前 URL 不是 new_chat ({page.url})，导航到 {new_chat_url}")
                 await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=30000)
                 await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=30000)
            return True
        logger.info(f"[{req_id}] 从 {current_prefs_for_modification.get('promptModel', '未知')} 更新 localStorage.promptModel 为 {full_model_path}")
        current_prefs_for_modification["promptModel"] = full_model_path
        await page.evaluate("(prefsStr) => localStorage.setItem('aiStudioUserPreference', prefsStr)", json.dumps(current_prefs_for_modification))
        logger.info(f"[{req_id}] localStorage 已更新，导航到 '{new_chat_url}' 应用新模型...")
        await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=30000)
        input_field = page.locator(INPUT_SELECTOR)
        await expect_async(input_field).to_be_visible(timeout=30000)
        logger.info(f"[{req_id}] 页面已导航到新聊天并加载完成，输入框可见")
        final_prefs_str = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
        final_prompt_model_in_storage: Optional[str] = None
        if final_prefs_str:
            try:
                final_prefs_obj = json.loads(final_prefs_str)
                final_prompt_model_in_storage = final_prefs_obj.get("promptModel")
            except json.JSONDecodeError:
                logger.warning(f"[{req_id}] 无法解析刷新后的 aiStudioUserPreference JSON 字符串。")
        if final_prompt_model_in_storage == full_model_path:
            logger.info(f"[{req_id}] ✅ AI Studio localStorage 中模型已成功设置为: {full_model_path}")
            page_display_match = False
            expected_display_name_for_target_id = None
            actual_displayed_model_name_on_page = "无法读取"
            if parsed_model_list:
                for m_obj in parsed_model_list:
                    if m_obj.get("id") == model_id:
                        expected_display_name_for_target_id = m_obj.get("display_name")
                        break
            if not expected_display_name_for_target_id:
                logger.warning(f"[{req_id}] 无法在parsed_model_list中找到目标ID '{model_id}' 的显示名称，跳过页面显示名称验证。这可能不准确。")
                page_display_match = True
            else:
                try:
                    model_name_locator = page.locator('mat-select[data-test-ms-model-selector] div.model-option-content span.gmat-body-medium')
                    actual_displayed_model_name_on_page_raw = await model_name_locator.first.inner_text(timeout=5000)
                    actual_displayed_model_name_on_page = actual_displayed_model_name_on_page_raw.strip()
                    normalized_actual_display = actual_displayed_model_name_on_page.lower()
                    normalized_expected_display = expected_display_name_for_target_id.strip().lower()
                    if normalized_actual_display == normalized_expected_display:
                        page_display_match = True
                        logger.info(f"[{req_id}] ✅ 页面显示模型 ('{actual_displayed_model_name_on_page}') 与期望 ('{expected_display_name_for_target_id}') 一致。")
                    else:
                        logger.error(f"[{req_id}] ❌ 页面显示模型 ('{actual_displayed_model_name_on_page}') 与期望 ('{expected_display_name_for_target_id}') 不一致。(Raw page: '{actual_displayed_model_name_on_page_raw}')")
                except Exception as e_disp:
                    logger.warning(f"[{req_id}] 读取页面显示的当前模型名称时出错: {e_disp}。将无法验证页面显示。")
            if page_display_match:
                # 模型切换成功，清空页面消息缓存
                async with page_sync_cache_lock:
                    last_api_messages_synced_to_page = None
                    logger.info(f"[{req_id}] 模型切换成功，已清空 last_api_messages_synced_to_page。")
                return True
            else:
                logger.error(f"[{req_id}] ❌ 模型切换失败，因为页面显示的模型与期望不符 (即使localStorage可能已更改)。")
        else:
            logger.error(f"[{req_id}] ❌ AI Studio 未接受模型更改 (localStorage)。期望='{full_model_path}', 实际='{final_prompt_model_in_storage or '未设置或无效'}'.")
        logger.info(f"[{req_id}] 模型切换失败。尝试恢复到页面当前实际显示的模型的状态...")
        current_displayed_name_for_revert_raw = "无法读取"
        current_displayed_name_for_revert_stripped = "无法读取"
        try:
            model_name_locator_revert = page.locator('mat-select[data-test-ms-model-selector] div.model-option-content span.gmat-body-medium')
            current_displayed_name_for_revert_raw = await model_name_locator_revert.first.inner_text(timeout=5000)
            current_displayed_name_for_revert_stripped = current_displayed_name_for_revert_raw.strip()
            logger.info(f"[{req_id}] 恢复：页面当前显示的模型名称 (原始: '{current_displayed_name_for_revert_raw}', 清理后: '{current_displayed_name_for_revert_stripped}')")
        except Exception as e_read_disp_revert:
            logger.warning(f"[{req_id}] 恢复：读取页面当前显示模型名称失败: {e_read_disp_revert}。将尝试回退到原始localStorage。")
            if original_prefs_str:
                logger.info(f"[{req_id}] 恢复：由于无法读取当前页面显示，尝试将 localStorage 恢复到原始状态: '{original_prompt_model or '未设置'}'")
                await page.evaluate("(origPrefs) => localStorage.setItem('aiStudioUserPreference', origPrefs)", original_prefs_str)
                logger.info(f"[{req_id}] 恢复：导航到 '{new_chat_url}' 以应用恢复的原始 localStorage 设置...")
                await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=20000)
                await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=20000)
                logger.info(f"[{req_id}] 恢复：页面已导航到新聊天并加载，已尝试应用原始 localStorage。")
            else:
                logger.warning(f"[{req_id}] 恢复：无有效的原始 localStorage 状态可恢复，也无法读取当前页面显示。")
            return False
        model_id_to_revert_to = None
        if parsed_model_list and current_displayed_name_for_revert_stripped != "无法读取":
            normalized_current_display_for_revert = current_displayed_name_for_revert_stripped.lower()
            for m_obj in parsed_model_list:
                parsed_list_display_name = m_obj.get("display_name", "").strip().lower()
                if parsed_list_display_name == normalized_current_display_for_revert:
                    model_id_to_revert_to = m_obj.get("id")
                    logger.info(f"[{req_id}] 恢复：页面显示名称 '{current_displayed_name_for_revert_stripped}' 对应模型ID: {model_id_to_revert_to}")
                    break
            if not model_id_to_revert_to:
                logger.warning(f"[{req_id}] 恢复：无法在 parsed_model_list 中找到与页面显示名称 '{current_displayed_name_for_revert_stripped}' 匹配的模型ID。")
        else:
            if current_displayed_name_for_revert_stripped == "无法读取":
                 logger.warning(f"[{req_id}] 恢复：因无法读取页面显示名称，故不能从 parsed_model_list 转换ID。")
            else:
                 logger.warning(f"[{req_id}] 恢复：parsed_model_list 为空，无法从显示名称 '{current_displayed_name_for_revert_stripped}' 转换模型ID。")
        if model_id_to_revert_to:
            base_prefs_for_final_revert = {}
            try:
                current_ls_content_str = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
                if current_ls_content_str:
                    base_prefs_for_final_revert = json.loads(current_ls_content_str)
                elif original_prefs_str:
                    base_prefs_for_final_revert = json.loads(original_prefs_str)
            except json.JSONDecodeError:
                logger.warning(f"[{req_id}] 恢复：解析现有 localStorage 以构建恢复偏好失败。")
            path_to_revert_to = f"models/{model_id_to_revert_to}"
            base_prefs_for_final_revert["promptModel"] = path_to_revert_to
            logger.info(f"[{req_id}] 恢复：准备将 localStorage.promptModel 设置回页面实际显示的模型的路径: '{path_to_revert_to}'")
            await page.evaluate("(prefsStr) => localStorage.setItem('aiStudioUserPreference', prefsStr)", json.dumps(base_prefs_for_final_revert))
            logger.info(f"[{req_id}] 恢复：导航到 '{new_chat_url}' 以应用恢复到 '{model_id_to_revert_to}' 的 localStorage 设置...")
            await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=30000)
            await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=30000)
            logger.info(f"[{req_id}] 恢复：页面已导航到新聊天并加载。localStorage 应已设置为反映模型 '{model_id_to_revert_to}'。")
        else:
            logger.error(f"[{req_id}] 恢复：无法将模型恢复到页面显示的状态，因为未能从显示名称 '{current_displayed_name_for_revert_stripped}' 确定有效模型ID。")
            if original_prefs_str:
                logger.warning(f"[{req_id}] 恢复：作为最终后备，尝试恢复到原始 localStorage: '{original_prompt_model or '未设置'}'")
                await page.evaluate("(origPrefs) => localStorage.setItem('aiStudioUserPreference', origPrefs)", original_prefs_str)
                logger.info(f"[{req_id}] 恢复：导航到 '{new_chat_url}' 以应用最终后备的原始 localStorage。")
                await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=20000)
                await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=20000)
                logger.info(f"[{req_id}] 恢复：页面已导航到新聊天并加载，已应用最终后备的原始 localStorage。")
            else:
                logger.warning(f"[{req_id}] 恢复：无有效的原始 localStorage 状态可作为最终后备。")
        return False
    except Exception as e:
        logger.exception(f"[{req_id}] ❌ 切换模型过程中发生严重错误")
        await save_error_snapshot(f"model_switch_error_{req_id}")
        try:
            if original_prefs_str:
                logger.info(f"[{req_id}] 发生异常，尝试恢复 localStorage 至: {original_prompt_model or '未设置'}")
                await page.evaluate("(origPrefs) => localStorage.setItem('aiStudioUserPreference', origPrefs)", original_prefs_str)
                logger.info(f"[{req_id}] 异常恢复：导航到 '{new_chat_url}' 以应用恢复的 localStorage。")
                await page.goto(new_chat_url, wait_until="domcontentloaded", timeout=15000)
                await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=15000)
        except Exception as recovery_err:
            logger.error(f"[{req_id}] 异常后恢复 localStorage 失败: {recovery_err}")
        return False

# --- Load Excluded Models ---
def load_excluded_models(filename: str):
    global excluded_model_ids, logger
    excluded_file_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        if os.path.exists(excluded_file_path):
            with open(excluded_file_path, 'r', encoding='utf-8') as f:
                loaded_ids = {line.strip() for line in f if line.strip()}
            if loaded_ids:
                excluded_model_ids.update(loaded_ids)
                logger.info(f"✅ 从 '{filename}' 加载了 {len(loaded_ids)} 个模型到排除列表: {excluded_model_ids}")
            else:
                logger.info(f"'{filename}' 文件为空或不包含有效的模型 ID，排除列表未更改。")
        else:
            logger.info(f"模型排除列表文件 '{filename}' 未找到，排除列表为空。")
    except Exception as e:
        logger.error(f"❌ 从 '{filename}' 加载排除模型列表时出错: {e}", exc_info=True)

# --- Handle Initial Model State and Storage ---
async def _handle_initial_model_state_and_storage(page: AsyncPage):
    global current_ai_studio_model_id, logger, parsed_model_list, model_list_fetch_event, INPUT_SELECTOR
    logger.info("--- (新) 处理初始模型状态, localStorage 和 isAdvancedOpen ---")
    needs_reload_and_storage_update = False
    reason_for_reload = ""
    try:
        initial_prefs_str = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
        if not initial_prefs_str:
            needs_reload_and_storage_update = True
            reason_for_reload = "localStorage.aiStudioUserPreference 未找到。"
            logger.info(f"   判定需要刷新和存储更新: {reason_for_reload}")
        else:
            logger.info("   localStorage 中找到 'aiStudioUserPreference'。正在解析...")
            try:
                pref_obj = json.loads(initial_prefs_str)
                prompt_model_path = pref_obj.get("promptModel")
                is_advanced_open_in_storage = pref_obj.get("isAdvancedOpen")
                is_prompt_model_valid = isinstance(prompt_model_path, str) and prompt_model_path.strip()
                if not is_prompt_model_valid:
                    needs_reload_and_storage_update = True
                    reason_for_reload = "localStorage.promptModel 无效或未设置。"
                    logger.info(f"   判定需要刷新和存储更新: {reason_for_reload}")
                elif is_advanced_open_in_storage is not True:
                    needs_reload_and_storage_update = True
                    reason_for_reload = f"localStorage.isAdvancedOpen ({is_advanced_open_in_storage}) 不为 True。"
                    logger.info(f"   判定需要刷新和存储更新: {reason_for_reload}")
                else:
                    current_ai_studio_model_id = prompt_model_path.split('/')[-1]
                    logger.info(f"   ✅ localStorage 有效且 isAdvancedOpen=true。初始模型 ID 从 localStorage 设置为: {current_ai_studio_model_id}")
            except json.JSONDecodeError:
                needs_reload_and_storage_update = True
                reason_for_reload = "解析 localStorage.aiStudioUserPreference JSON 失败。"
                logger.error(f"   判定需要刷新和存储更新: {reason_for_reload}")
        if needs_reload_and_storage_update:
            logger.info(f"   执行刷新和存储更新流程，原因: {reason_for_reload}")
            logger.info("   步骤 1: 调用 _set_model_from_page_display(set_storage=True) 更新 localStorage 和全局模型 ID...")
            await _set_model_from_page_display(page, set_storage=True)
            current_page_url = page.url
            logger.info(f"   步骤 2: 重新加载页面 ({current_page_url}) 以应用 isAdvancedOpen=true...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"   尝试重新加载页面 (第 {attempt + 1}/{max_retries} 次): {current_page_url}")
                    await page.goto(current_page_url, wait_until="domcontentloaded", timeout=40000)
                    await expect_async(page.locator(INPUT_SELECTOR)).to_be_visible(timeout=30000)
                    logger.info(f"   ✅ 页面已成功重新加载到: {page.url}")
                    break  # 成功则跳出循环
                except Exception as reload_err:
                    logger.warning(f"   ⚠️ 页面重新加载尝试 {attempt + 1}/{max_retries} 失败: {reload_err}")
                    if attempt < max_retries - 1:
                        logger.info(f"   将在5秒后重试...")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"   ❌ 页面重新加载在 {max_retries} 次尝试后最终失败: {reload_err}. 后续模型状态可能不准确。", exc_info=True)
                        await save_error_snapshot(f"initial_storage_reload_fail_attempt_{attempt+1}")
                        # Consider re-raising or handling more gracefully if critical
                        # 例如，如果这是关键步骤，可能需要: raise reload_err
            logger.info("   步骤 3: 重新加载后，再次调用 _set_model_from_page_display(set_storage=False) 以同步全局模型 ID...")
            await _set_model_from_page_display(page, set_storage=False)
            logger.info(f"   ✅ 刷新和存储更新流程完成。最终全局模型 ID: {current_ai_studio_model_id}")
        else:
            logger.info("   localStorage 状态良好 (isAdvancedOpen=true, promptModel有效)，无需刷新页面。")
    except Exception as e:
        logger.error(f"❌ (新) 处理初始模型状态和 localStorage 时发生严重错误: {e}", exc_info=True)
        try:
            logger.warning("   由于发生错误，尝试回退仅从页面显示设置全局模型 ID (不写入localStorage)...")
            await _set_model_from_page_display(page, set_storage=False)
        except Exception as fallback_err:
            logger.error(f"   回退设置模型ID也失败: {fallback_err}")

async def _set_model_from_page_display(page: AsyncPage, set_storage: bool = False):
    global current_ai_studio_model_id, logger, parsed_model_list, model_list_fetch_event
    try:
        logger.info("   尝试从页面显示元素读取当前模型名称...")
        model_name_locator = page.locator('mat-select[data-test-ms-model-selector] div.model-option-content span.gmat-body-medium')
        displayed_model_name_from_page_raw = await model_name_locator.first.inner_text(timeout=7000)
        displayed_model_name = displayed_model_name_from_page_raw.strip()
        logger.info(f"   页面当前显示模型名称 (原始: '{displayed_model_name_from_page_raw}', 清理后: '{displayed_model_name}')")
        found_model_id_from_display = None
        if not model_list_fetch_event.is_set():
            logger.info("   等待模型列表数据 (最多5秒) 以便转换显示名称...")
            try: await asyncio.wait_for(model_list_fetch_event.wait(), timeout=5.0)
            except asyncio.TimeoutError: logger.warning("   等待模型列表超时，可能无法准确转换显示名称为ID。")
        if parsed_model_list:
            for model_obj in parsed_model_list:
                if model_obj.get("display_name") and model_obj.get("display_name").strip() == displayed_model_name:
                    found_model_id_from_display = model_obj.get("id")
                    logger.info(f"   显示名称 '{displayed_model_name}' 对应模型 ID: {found_model_id_from_display}")
                    break
            if not found_model_id_from_display:
                 logger.warning(f"   未在已知模型列表中找到与显示名称 '{displayed_model_name}' 匹配的 ID。")
        else:
            logger.warning("   模型列表尚不可用，无法将显示名称转换为ID。")
        new_model_value = found_model_id_from_display if found_model_id_from_display else displayed_model_name
        if current_ai_studio_model_id != new_model_value:
            current_ai_studio_model_id = new_model_value
            logger.info(f"   全局 current_ai_studio_model_id 已更新为: {current_ai_studio_model_id}")
        else:
            logger.info(f"   全局 current_ai_studio_model_id ('{current_ai_studio_model_id}') 与从页面获取的值一致，未更改。")
        if set_storage:
            logger.info(f"   准备为页面状态设置 localStorage (确保 isAdvancedOpen=true)...")
            existing_prefs_for_update_str = await page.evaluate("() => localStorage.getItem('aiStudioUserPreference')")
            prefs_to_set = {}
            if existing_prefs_for_update_str:
                try:
                    prefs_to_set = json.loads(existing_prefs_for_update_str)
                except json.JSONDecodeError:
                    logger.warning("   解析现有 localStorage.aiStudioUserPreference 失败，将创建新的偏好设置。")
            prefs_to_set["isAdvancedOpen"] = True
            logger.info(f"     强制 isAdvancedOpen: true")
            prefs_to_set["areToolsOpen"] = False
            logger.info(f"     强制 areToolsOpen: false")
            if found_model_id_from_display:
                new_prompt_model_path = f"models/{found_model_id_from_display}"
                prefs_to_set["promptModel"] = new_prompt_model_path
                logger.info(f"     设置 promptModel 为: {new_prompt_model_path} (基于找到的ID)")
            elif "promptModel" not in prefs_to_set:
                logger.warning(f"     无法从页面显示 '{displayed_model_name}' 找到模型ID，且 localStorage 中无现有 promptModel。promptModel 将不会被主动设置以避免潜在问题。")
            default_keys_if_missing = {
                "bidiModel": "models/gemini-1.0-pro-001",
                "isSafetySettingsOpen": False,
                "hasShownSearchGroundingTos": False,
                "autosaveEnabled": True,
                "theme": "system",
                "bidiOutputFormat": 3,
                "isSystemInstructionsOpen": False,
                "warmWelcomeDisplayed": True,
                "getCodeLanguage": "Node.js",
                "getCodeHistoryToggle": False,
                "fileCopyrightAcknowledged": True
            }
            for key, val_default in default_keys_if_missing.items():
                if key not in prefs_to_set:
                    prefs_to_set[key] = val_default
            await page.evaluate("(prefsStr) => localStorage.setItem('aiStudioUserPreference', prefsStr)", json.dumps(prefs_to_set))
            logger.info(f"   ✅ localStorage.aiStudioUserPreference 已更新。isAdvancedOpen: {prefs_to_set.get('isAdvancedOpen')}, areToolsOpen: {prefs_to_set.get('areToolsOpen')}, promptModel: '{prefs_to_set.get('promptModel', '未设置/保留原样')}'。")
    except Exception as e_set_disp:
        logger.error(f"   尝试从页面显示设置模型时出错: {e_set_disp}", exc_info=True)
