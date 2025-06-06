"""
FastAPI应用初始化和生命周期管理
"""

import asyncio
import logging
import multiprocessing
import os
import platform
import sys
from contextlib import asynccontextmanager
from typing import Optional, List
from models import Message

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from playwright.async_api import Browser as AsyncBrowser, Playwright as AsyncPlaywright

# --- 配置模块导入 ---
from config import *

# --- models模块导入 ---
from models import WebSocketConnectionManager, WebSocketLogHandler

# --- logging_utils模块导入 ---
from logging_utils import setup_server_logging, restore_original_streams

# --- browser_utils模块导入 ---
from browser_utils import (
    _initialize_page_logic,
    _close_page_logic,
    signal_camoufox_shutdown,
    _handle_model_list_response,
    load_excluded_models,
    _handle_initial_model_state_and_storage
)

import stream
from asyncio import Queue, Lock, Task, Event

# 全局状态变量（这些将在server.py中被引用）
playwright_manager: Optional[AsyncPlaywright] = None
browser_instance: Optional[AsyncBrowser] = None
page_instance = None
is_playwright_ready = False
is_browser_connected = False
is_page_ready = False
is_initializing = False

global_model_list_raw_json = None
parsed_model_list = []
model_list_fetch_event = None

current_ai_studio_model_id = None
model_switching_lock = None

excluded_model_ids = set()

request_queue = None
processing_lock = None
worker_task = None

page_params_cache = {}
params_cache_lock = None

log_ws_manager = None

STREAM_QUEUE = None
STREAM_PROCESS = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app_param: FastAPI):
    """FastAPI应用生命周期管理"""
    # 导入server.py中的全局变量，以便正确初始化
    import server
    from server import queue_worker
    
    # 存储原始流供恢复使用
    initial_stdout_before_redirect = sys.stdout
    initial_stderr_before_redirect = sys.stderr
    true_original_stdout = sys.__stdout__
    true_original_stderr = sys.__stderr__

    # 设置服务器日志
    log_level_env = os.environ.get('SERVER_LOG_LEVEL', 'INFO')
    redirect_print_env = os.environ.get('SERVER_REDIRECT_PRINT', 'false')
    
    # 初始化日志 WebSocket 管理器
    server.log_ws_manager = WebSocketConnectionManager()
    
    initial_stdout_before_redirect, initial_stderr_before_redirect = setup_server_logging(
        logger_instance=server.logger,
        log_ws_manager=server.log_ws_manager,
        log_level_name=log_level_env,
        redirect_print_str=redirect_print_env
    )

    # WebSocket日志处理器已在setup_server_logging函数中添加，无需重复添加
    
    # 获取logger实例供后续使用
    logger = server.logger
    
    # 初始化全局变量
    server.request_queue = Queue()
    server.processing_lock = Lock()
    server.model_switching_lock = Lock()
    server.params_cache_lock = Lock()
    server.page_sync_cache_lock = Lock() # 修正：正确初始化 server 模块上的锁
    
    # 初始化代理设置 - 移动到server模块中进行全局配置
    PROXY_SERVER_ENV = "http://127.0.0.1:3120/"
    STREAM_PROXY_SERVER_ENV = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')

    STREAM_PORT = os.environ.get('STREAM_PORT')
    if STREAM_PORT == '0':
        PROXY_SERVER_ENV = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    elif STREAM_PORT is not None:
        PROXY_SERVER_ENV = f"http://127.0.0.1:{STREAM_PORT}/"

    # 设置全局代理配置到server模块
    server.PLAYWRIGHT_PROXY_SETTINGS = None
    if PROXY_SERVER_ENV:
        server.PLAYWRIGHT_PROXY_SETTINGS = {'server': PROXY_SERVER_ENV}
        if NO_PROXY_ENV:
            server.PLAYWRIGHT_PROXY_SETTINGS['bypass'] = NO_PROXY_ENV.replace(',', ';')

    if STREAM_PORT != '0':
        logger.info(f"STREAM 代理启动中，端口: {STREAM_PORT}")
        server.STREAM_QUEUE = multiprocessing.Queue()
        if STREAM_PORT is None:
            port = 3120
        else:
            port = int(STREAM_PORT)
        logger.info(f"STREAM 代理使用上游代理服务器：{STREAM_PROXY_SERVER_ENV}")
        server.STREAM_PROCESS = multiprocessing.Process(target=stream.start, args=(server.STREAM_QUEUE, port, STREAM_PROXY_SERVER_ENV))
        server.STREAM_PROCESS.start()
        logger.info("STREAM 代理启动完毕")
    else:
        logger.info("STREAM 代理已禁用")

    logger.info(f"--- 依赖和环境检查 ---")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"运行平台: {platform.platform()}")
    logger.info(f"Playwright 已导入")
    logger.info(f"FastAPI 应用已初始化")
    
    if server.PLAYWRIGHT_PROXY_SETTINGS:
        logger.info(f"--- 代理配置检测到 (由 server.py 的 lifespan 记录) ---")
        logger.info(f"   将使用代理服务器: {server.PLAYWRIGHT_PROXY_SETTINGS['server']}")
        if 'bypass' in server.PLAYWRIGHT_PROXY_SETTINGS:
            logger.info(f"   绕过代理的主机: {server.PLAYWRIGHT_PROXY_SETTINGS['bypass']}")
        logger.info(f"-----------------------")
    else:
        logger.info("--- 未检测到 HTTP_PROXY 或 HTTPS_PROXY 环境变量，不使用代理 (由 server.py 的 lifespan 记录) ---")
    
    load_excluded_models(EXCLUDED_MODELS_FILENAME)
    server.is_initializing = True
    logger.info("\n" + "="*60 + "\n          🚀 AI Studio Proxy Server (FastAPI App Lifespan) 🚀\n" + "="*60)
    logger.info(f"FastAPI 应用生命周期: 启动中...")
    
    try:
        logger.info(f"   启动 Playwright...")
        from playwright.async_api import async_playwright
        server.playwright_manager = await async_playwright().start()
        server.is_playwright_ready = True
        logger.info(f"   ✅ Playwright 已启动。")
        
        ws_endpoint = os.environ.get('CAMOUFOX_WS_ENDPOINT')
        launch_mode = os.environ.get('LAUNCH_MODE', 'unknown')
        
        if not ws_endpoint:
            if launch_mode == "direct_debug_no_browser":
                logger.warning("CAMOUFOX_WS_ENDPOINT 未设置，但 LAUNCH_MODE 表明不需要浏览器。跳过浏览器连接。")
                server.is_browser_connected = False
                server.is_page_ready = False
                server.model_list_fetch_event.set()
            else:
                logger.error("未找到 CAMOUFOX_WS_ENDPOINT 环境变量。Playwright 将无法连接到浏览器。")
                raise ValueError("CAMOUFOX_WS_ENDPOINT 环境变量缺失。")
        else:
            logger.info(f"   连接到 Camoufox 服务器 (浏览器 WebSocket 端点) 于: {ws_endpoint}")
            try:
                server.browser_instance = await server.playwright_manager.firefox.connect(ws_endpoint, timeout=30000)
                server.is_browser_connected = True
                logger.info(f"   ✅ 已连接到浏览器实例: 版本 {server.browser_instance.version}")
                
                temp_page_instance, temp_is_page_ready = await _initialize_page_logic(server.browser_instance)
                if temp_page_instance and temp_is_page_ready:
                    server.page_instance = temp_page_instance
                    server.is_page_ready = temp_is_page_ready
                    await _handle_initial_model_state_and_storage(server.page_instance)
                    async with server.page_sync_cache_lock: # 修正：使用 server 模块上的锁
                        server.last_api_messages_synced_to_page = None # 修正：使用 server 模块上的变量
                        logger.info("页面初始化成功，已清空 last_api_messages_synced_to_page。")
                else:
                    server.is_page_ready = False
                    if not server.model_list_fetch_event.is_set(): 
                        server.model_list_fetch_event.set()
            except Exception as connect_err:
                logger.error(f"未能连接到 Camoufox 服务器 (浏览器) 或初始化页面失败: {connect_err}", exc_info=True)
                if launch_mode != "direct_debug_no_browser":
                    raise RuntimeError(f"未能连接到 Camoufox 或初始化页面: {connect_err}") from connect_err
                else:
                    server.is_browser_connected = False
                    server.is_page_ready = False
                    if not server.model_list_fetch_event.is_set(): 
                        server.model_list_fetch_event.set()

        if server.is_page_ready and server.is_browser_connected and not server.model_list_fetch_event.is_set():
            logger.info("等待模型列表捕获 (最多等待15秒)...")
            try:
                await asyncio.wait_for(server.model_list_fetch_event.wait(), timeout=15.0)
                if server.model_list_fetch_event.is_set():
                    logger.info("模型列表事件已触发。")
                else:
                    logger.warning("模型列表事件等待后仍未设置。")
            except asyncio.TimeoutError:
                logger.warning("等待模型列表捕获超时。将使用默认或空列表。")
            finally:
                if not server.model_list_fetch_event.is_set():
                    server.model_list_fetch_event.set()
        elif not (server.is_page_ready and server.is_browser_connected):
            if not server.model_list_fetch_event.is_set(): 
                server.model_list_fetch_event.set()

        if (server.is_page_ready and server.is_browser_connected) or launch_mode == "direct_debug_no_browser":
            logger.info(f"   启动请求处理 Worker...")
            server.worker_task = asyncio.create_task(queue_worker())
            logger.info(f"   ✅ 请求处理 Worker 已启动。")
        elif launch_mode == "direct_debug_no_browser":
            logger.warning("浏览器和页面未就绪 (direct_debug_no_browser 模式)，请求处理 Worker 未启动。API 可能功能受限。")
        else:
            logger.error("页面或浏览器初始化失败，无法启动 Worker。")
            if not server.model_list_fetch_event.is_set(): 
                server.model_list_fetch_event.set()
            raise RuntimeError("页面或浏览器初始化失败，无法启动 Worker。")
        
        logger.info(f"✅ FastAPI 应用生命周期: 启动完成。服务已就绪。")
        server.is_initializing = False
        yield
        
    except Exception as startup_err:
        logger.critical(f"❌ FastAPI 应用生命周期: 启动期间发生严重错误: {startup_err}", exc_info=True)
        if not server.model_list_fetch_event.is_set(): 
            server.model_list_fetch_event.set()
        if server.worker_task and not server.worker_task.done(): 
            server.worker_task.cancel()
        if server.browser_instance and server.browser_instance.is_connected():
            try: 
                await server.browser_instance.close()
            except: 
                pass
        if server.playwright_manager:
            try: 
                await server.playwright_manager.stop()
            except: 
                pass
        raise RuntimeError(f"应用程序启动失败: {startup_err}") from startup_err
    finally:
        logger.info("STREAM 代理关闭中")
        if server.STREAM_PROCESS:
            server.STREAM_PROCESS.terminate()

        server.is_initializing = False
        logger.info(f"\nFastAPI 应用生命周期: 关闭中...")
        
        if server.worker_task and not server.worker_task.done():
            logger.info(f"   正在取消请求处理 Worker...")
            server.worker_task.cancel()
            try:
                await asyncio.wait_for(server.worker_task, timeout=5.0)
                logger.info(f"   ✅ 请求处理 Worker 已停止/取消。")
            except asyncio.TimeoutError: 
                logger.warning(f"   ⚠️ Worker 等待超时。")
            except asyncio.CancelledError: 
                logger.info(f"   ✅ 请求处理 Worker 已确认取消。")
            except Exception as wt_err: 
                logger.error(f"   ❌ 等待 Worker 停止时出错: {wt_err}", exc_info=True)
        
        if server.page_instance and not server.page_instance.is_closed():
            try:
                logger.info("Lifespan 清理：移除模型列表响应监听器。")
                server.page_instance.remove_listener("response", _handle_model_list_response)
            except Exception as e:
                logger.debug(f"Lifespan 清理：移除监听器时发生非严重错误或监听器本不存在: {e}")
        
        if server.page_instance:
            await _close_page_logic()
        
        if server.browser_instance:
            logger.info(f"   正在关闭与浏览器实例的连接...")
            try:
                if server.browser_instance.is_connected():
                    await server.browser_instance.close()
                    logger.info(f"   ✅ 浏览器连接已关闭。")
                else: 
                    logger.info(f"   ℹ️ 浏览器先前已断开连接。")
            except Exception as close_err: 
                logger.error(f"   ❌ 关闭浏览器连接时出错: {close_err}", exc_info=True)
            finally: 
                server.browser_instance = None
                server.is_browser_connected = False
                server.is_page_ready = False
        
        if server.playwright_manager:
            logger.info(f"   停止 Playwright...")
            try:
                await server.playwright_manager.stop()
                logger.info(f"   ✅ Playwright 已停止。")
            except Exception as stop_err: 
                logger.error(f"   ❌ 停止 Playwright 时出错: {stop_err}", exc_info=True)
            finally: 
                server.playwright_manager = None
                server.is_playwright_ready = False
                
        server.last_api_messages_synced_to_page = None # 修正：使用 server 模块上的变量
        logger.info(f"   重置 last_api_messages_synced_to_page 缓存。")
        restore_original_streams(initial_stdout_before_redirect, initial_stderr_before_redirect)
        restore_original_streams(true_original_stdout, true_original_stderr)
        logger.info(f"✅ FastAPI 应用生命周期: 关闭完成。")
        
# --- 鉴权配置 ---
auth_logger = logging.getLogger("app.auth") 
API_KEY = os.getenv("API_KEY") # 从环境变量获取 API Key
api_key_scheme = HTTPBearer()
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(api_key_scheme)):
    """
    验证 API 密钥。
    - 如果环境变量 API_KEY 未设置，则跳过验证，允许所有请求。
    - 如果已设置，则强制要求提供有效的 Bearer Token。
    """
    # 如果服务器未配置API_KEY，则不进行任何验证
    if not API_KEY:
        auth_logger.info("未获取到API_KEY环境变量，跳过API密钥验证。")
        return credentials

    # 检查凭据是否存在
    if not credentials:
        auth_logger.warning("Auth failed: No credentials provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. API Key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查认证方案是否为 "Bearer"
    if credentials.scheme != "Bearer":
        auth_logger.warning(f"Auth failed: Invalid scheme '{credentials.scheme}'. 'Bearer' required.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. 'Bearer' token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证密钥是否匹配 (使用更安全的方式，虽然对于简单字符串差异不大，但养成好习惯)
    # import hmac
    # is_valid = hmac.compare_digest(credentials.credentials, API_KEY)
    is_valid = credentials.credentials == API_KEY
    
    if not is_valid:
        auth_logger.warning("Auth failed: Invalid API Key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证成功
    return credentials


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    app = FastAPI(
        title="AI Studio Proxy Server (集成模式)",
        description="通过 Playwright与 AI Studio 交互的代理服务器。",
        version="0.6.0-integrated",
        lifespan=lifespan
    )
    
    # 注册路由
    from .routes import (
        read_index, get_css, get_js, get_api_info,
        health_check, list_models, chat_completions,
        cancel_request, get_queue_status, websocket_log_endpoint
    )
    from fastapi.responses import FileResponse
    
    app.get("/", response_class=FileResponse)(read_index)
    app.get("/webui.css")(get_css)
    app.get("/webui.js")(get_js)
    app.get("/api/info")(get_api_info)
    app.get("/health")(health_check)
    app.get("/v1/models", dependencies=[Depends(verify_api_key)])(list_models)
    app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])(chat_completions)
    app.post("/v1/cancel/{req_id}", dependencies=[Depends(verify_api_key)])(cancel_request)
    app.get("/v1/queue", dependencies=[Depends(verify_api_key)])(get_queue_status)
    app.websocket("/ws/logs")(websocket_log_endpoint)
    
    return app