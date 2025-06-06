"""
请求处理器模块
包含核心的请求处理逻辑
"""

import asyncio
import json
import os
import random
import time
from typing import Optional, Tuple, Callable, AsyncGenerator
from asyncio import Event, Future

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from playwright.async_api import Page as AsyncPage, Locator, Error as PlaywrightAsyncError, expect as expect_async, TimeoutError

# --- 配置模块导入 ---
from config import *

# --- models模块导入 ---
from models import ChatCompletionRequest, ClientDisconnectedError

# --- browser_utils模块导入 ---
from browser_utils import (
    switch_ai_studio_model, 
    save_error_snapshot,
    _wait_for_response_completion,
    _get_final_response_content,
    detect_and_extract_page_error,
    get_response_via_edit_button,
    get_response_via_copy_button,
    get_raw_text_content,
    set_system_prompt_in_page,
)

# --- api_utils模块导入 ---
from .utils import (
    validate_chat_request, 
    prepare_combined_prompt,
    generate_sse_chunk,
    generate_sse_stop_chunk,
    generate_sse_error_chunk,
    use_helper_get_response,
    use_stream_response,
    calculate_usage_stats
)


async def _initialize_request_context(req_id: str, request: ChatCompletionRequest) -> dict:
    """初始化请求上下文"""
    from server import (
        logger, page_instance, is_page_ready, parsed_model_list,
        current_ai_studio_model_id, model_switching_lock, page_params_cache,
        params_cache_lock
    )
    
    logger.info(f"[{req_id}] 开始处理请求...")
    logger.info(f"[{req_id}]   请求参数 - Model: {request.model}, Stream: {request.stream}")
    logger.info(f"[{req_id}]   请求参数 - Temperature: {request.temperature}")
    logger.info(f"[{req_id}]   请求参数 - Max Output Tokens: {request.max_output_tokens}")
    logger.info(f"[{req_id}]   请求参数 - Stop Sequences: {request.stop}")
    logger.info(f"[{req_id}]   请求参数 - Top P: {request.top_p}")
    
    context = {
        'logger': logger,
        'page': page_instance,
        'is_page_ready': is_page_ready,
        'parsed_model_list': parsed_model_list,
        'current_ai_studio_model_id': current_ai_studio_model_id,
        'model_switching_lock': model_switching_lock,
        'page_params_cache': page_params_cache,
        'params_cache_lock': params_cache_lock,
        'is_streaming': request.stream,
        'model_actually_switched': False,
        'requested_model': request.model,
        'model_id_to_use': None,
        'needs_model_switching': False
    }
    
    return context


async def _analyze_model_requirements(req_id: str, context: dict, request: ChatCompletionRequest) -> dict:
    """分析模型需求并确定是否需要切换"""
    logger = context['logger']
    current_ai_studio_model_id = context['current_ai_studio_model_id']
    parsed_model_list = context['parsed_model_list']
    requested_model = request.model
    
    if requested_model and requested_model != MODEL_NAME:
        requested_model_parts = requested_model.split('/')
        requested_model_id = requested_model_parts[-1] if len(requested_model_parts) > 1 else requested_model
        logger.info(f"[{req_id}] 请求使用模型: {requested_model_id}")
        
        if parsed_model_list:
            valid_model_ids = [m.get("id") for m in parsed_model_list]
            if requested_model_id not in valid_model_ids:
                logger.error(f"[{req_id}] ❌ 无效的模型ID: {requested_model_id}。可用模型: {valid_model_ids}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"[{req_id}] Invalid model '{requested_model_id}'. Available models: {', '.join(valid_model_ids)}"
                )
        
        context['model_id_to_use'] = requested_model_id
        if current_ai_studio_model_id != requested_model_id:
            context['needs_model_switching'] = True
            logger.info(f"[{req_id}] 需要切换模型: 当前={current_ai_studio_model_id} -> 目标={requested_model_id}")
        else:
            logger.info(f"[{req_id}] 请求模型与当前模型相同 ({requested_model_id})，无需切换")
    else:
        logger.info(f"[{req_id}] 未指定具体模型或使用代理模型名称，将使用当前模型: {current_ai_studio_model_id or '未知'}")
    
    return context


async def _setup_disconnect_monitoring(req_id: str, http_request: Request, result_future: Future) -> Tuple[Event, asyncio.Task, Callable]:
    """设置客户端断开连接监控"""
    from server import logger
    
    client_disconnected_event = Event()
    
    async def check_disconnect_periodically():
        while not client_disconnected_event.is_set():
            try:
                if await http_request.is_disconnected():
                    logger.info(f"[{req_id}] (Disco Check Task) 客户端断开。设置事件并尝试停止。")
                    client_disconnected_event.set()
                    if not result_future.done(): 
                        result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] 客户端在处理期间关闭了请求"))
                    break
                await asyncio.sleep(1.0)
            except asyncio.CancelledError: 
                break
            except Exception as e:
                logger.error(f"[{req_id}] (Disco Check Task) 错误: {e}")
                client_disconnected_event.set()
                if not result_future.done(): 
                    result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Internal disconnect checker error: {e}"))
                break
    
    disconnect_check_task = asyncio.create_task(check_disconnect_periodically())
    
    def check_client_disconnected(*args):
        msg_to_log = ""
        if len(args) == 1 and isinstance(args[0], str):
            msg_to_log = args[0]

        if client_disconnected_event.is_set():
            logger.info(f"[{req_id}] {msg_to_log}检测到客户端断开连接事件。")
            raise ClientDisconnectedError(f"[{req_id}] Client disconnected event set.")
        return False
    
    return client_disconnected_event, disconnect_check_task, check_client_disconnected


async def _validate_page_status(req_id: str, context: dict, check_client_disconnected: Callable) -> None:
    """验证页面状态"""
    page = context['page']
    is_page_ready = context['is_page_ready']
    
    if not page or page.is_closed() or not is_page_ready:
        raise HTTPException(status_code=503, detail=f"[{req_id}] AI Studio 页面丢失或未就绪。", headers={"Retry-After": "30"})
    
    check_client_disconnected("Initial Page Check: ")


async def _handle_model_switching(req_id: str, context: dict, check_client_disconnected: Callable) -> dict:
    """处理模型切换逻辑"""
    if not context['needs_model_switching'] or not context['model_id_to_use']:
        return context
    
    logger = context['logger']
    page = context['page']
    model_switching_lock = context['model_switching_lock']
    model_id_to_use = context['model_id_to_use']
    
    # 从server模块导入全局变量
    import server
    
    async with model_switching_lock:
        model_before_switch_attempt = server.current_ai_studio_model_id
        if server.current_ai_studio_model_id != model_id_to_use:
            logger.info(f"[{req_id}] 获取锁后准备切换: 当前内存中模型={server.current_ai_studio_model_id}, 目标={model_id_to_use}")
            switch_success = await switch_ai_studio_model(page, model_id_to_use, req_id)
            if switch_success:
                server.current_ai_studio_model_id = model_id_to_use
                context['model_actually_switched'] = True
                context['current_ai_studio_model_id'] = server.current_ai_studio_model_id
                logger.info(f"[{req_id}] ✅ 模型切换成功。全局模型状态已更新为: {server.current_ai_studio_model_id}")
            else:
                await _handle_model_switch_failure(req_id, page, model_id_to_use, model_before_switch_attempt, logger)
        else:
            logger.info(f"[{req_id}] 获取锁后发现模型已是目标模型 {server.current_ai_studio_model_id}，无需切换")
    
    return context


async def _handle_model_switch_failure(req_id: str, page: AsyncPage, model_id_to_use: str, model_before_switch_attempt: str, logger) -> None:
    """处理模型切换失败的情况"""
    # 从server模块导入全局变量
    import server
    
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
    
    server.current_ai_studio_model_id = active_model_id_after_fail
    logger.info(f"[{req_id}] 全局模型状态在切换失败后设置为 (或保持为): {server.current_ai_studio_model_id}")
    
    actual_displayed_model_name = "未知 (无法读取)"
    try:
        model_wrapper_locator = page.locator('#mat-select-value-0 mat-select-trigger').first
        actual_displayed_model_name = await model_wrapper_locator.inner_text(timeout=3000)
    except Exception:
        pass
    
    raise HTTPException(
        status_code=422,
        detail=f"[{req_id}] AI Studio 未能应用所请求的模型 '{model_id_to_use}' 或该模型不受支持。请选择 AI Studio 网页界面中可用的模型。当前实际生效的模型 ID 为 '{server.current_ai_studio_model_id}', 页面显示为 '{actual_displayed_model_name}'."
    )


async def _handle_parameter_cache(req_id: str, context: dict) -> None:
    """处理参数缓存"""
    logger = context['logger']
    params_cache_lock = context['params_cache_lock']
    page_params_cache = context['page_params_cache']
    current_ai_studio_model_id = context['current_ai_studio_model_id']
    model_actually_switched = context['model_actually_switched']
    
    async with params_cache_lock:
        cached_model_for_params = page_params_cache.get("last_known_model_id_for_params")
        
        # 检查默认值版本，如果默认值改变了，需要清除缓存
        current_defaults_hash = f"{DEFAULT_TEMPERATURE}_{DEFAULT_MAX_OUTPUT_TOKENS}_{DEFAULT_TOP_P}_{len(DEFAULT_STOP_SEQUENCES)}"
        cached_defaults_hash = page_params_cache.get("defaults_version_hash")
        defaults_changed = cached_defaults_hash != current_defaults_hash
        
        if model_actually_switched or \
           (current_ai_studio_model_id is not None and current_ai_studio_model_id != cached_model_for_params) or \
           defaults_changed:
            
            reasons = []
            if model_actually_switched:
                reasons.append(f"model switched this call: {model_actually_switched}")
            if current_ai_studio_model_id != cached_model_for_params:
                reasons.append(f"model changed: {cached_model_for_params} -> {current_ai_studio_model_id}")
            if defaults_changed:
                reasons.append(f"default values changed: {cached_defaults_hash} -> {current_defaults_hash}")
            
            action_taken = "Invalidating" if page_params_cache else "Initializing"
            logger.info(f"[{req_id}] {action_taken} parameter cache. Reasons: {'; '.join(reasons)}")
            
            page_params_cache.clear()
            if current_ai_studio_model_id:
                page_params_cache["last_known_model_id_for_params"] = current_ai_studio_model_id
            page_params_cache["defaults_version_hash"] = current_defaults_hash
        else:
            logger.debug(f"[{req_id}] Parameter cache for model '{cached_model_for_params}' remains valid (current model: '{current_ai_studio_model_id}', switched this call: {model_actually_switched}, defaults unchanged).")


async def _prepare_and_validate_request(req_id: str, request: ChatCompletionRequest, check_client_disconnected: Callable) -> str:
    """准备和验证请求"""
    try: 
        validate_chat_request(request.messages, req_id)
    except ValueError as e: 
        raise HTTPException(status_code=400, detail=f"[{req_id}] 无效请求: {e}")
    
    system_prompt_content, prepared_prompt = prepare_combined_prompt(request.messages, req_id)
    check_client_disconnected("After Prompt Prep: ")
    
    return system_prompt_content, prepared_prompt


async def _clear_chat_history(req_id: str, page: AsyncPage, check_client_disconnected: Callable) -> None:
    """清空聊天记录"""
    from server import logger
    
    logger.info(f"[{req_id}] 开始清空聊天记录...")
    try:
        clear_chat_button_locator = page.locator(CLEAR_CHAT_BUTTON_SELECTOR)
        confirm_button_locator = page.locator(CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR)
        overlay_locator = page.locator(OVERLAY_SELECTOR)

        can_attempt_clear = False
        try:
            await expect_async(clear_chat_button_locator).to_be_enabled(timeout=3000)
            can_attempt_clear = True
            logger.info(f"[{req_id}] \"清空聊天\"按钮可用，继续清空流程。")
        except Exception as e_enable:
            is_new_chat_url = '/prompts/new_chat' in page.url.rstrip('/')
            if is_new_chat_url:
                logger.info(f"[{req_id}] \"清空聊天\"按钮不可用 (预期，因为在 new_chat 页面)。跳过清空操作。")
            else:
                logger.warning(f"[{req_id}] 等待\"清空聊天\"按钮可用失败: {e_enable}。清空操作可能无法执行。")
        
        check_client_disconnected("清空聊天 - \"清空聊天\"按钮可用性检查后: ")

        if can_attempt_clear:
            await _execute_chat_clear(req_id, page, clear_chat_button_locator, confirm_button_locator, overlay_locator, check_client_disconnected, logger)
            await _verify_chat_cleared(req_id, page, check_client_disconnected, logger)

    except Exception as e_clear:
        logger.error(f"[{req_id}] 清空聊天过程中发生错误: {e_clear}")
        if not (isinstance(e_clear, ClientDisconnectedError) or (hasattr(e_clear, 'name') and 'Disconnect' in e_clear.name)):
            await save_error_snapshot(f"clear_chat_error_{req_id}")
        raise


async def _execute_chat_clear(req_id: str, page: AsyncPage, clear_chat_button_locator: Locator, 
                             confirm_button_locator: Locator, overlay_locator: Locator, 
                             check_client_disconnected: Callable, logger) -> None:
    """执行清空聊天操作"""
    overlay_initially_visible = False
    try:
        if await overlay_locator.is_visible(timeout=1000):
            overlay_initially_visible = True
            logger.info(f"[{req_id}] 清空聊天确认遮罩层已可见。直接点击\"继续\"。")
    except TimeoutError:
        logger.info(f"[{req_id}] 清空聊天确认遮罩层初始不可见 (检查超时或未找到)。")
        overlay_initially_visible = False
    except Exception as e_vis_check:
        logger.warning(f"[{req_id}] 检查遮罩层可见性时发生错误: {e_vis_check}。假定不可见。")
        overlay_initially_visible = False
    
    check_client_disconnected("清空聊天 - 初始遮罩层检查后: ")

    if overlay_initially_visible:
        logger.info(f"[{req_id}] 点击\"继续\"按钮 (遮罩层已存在): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
        await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
    else:
        logger.info(f"[{req_id}] 点击\"清空聊天\"按钮: {CLEAR_CHAT_BUTTON_SELECTOR}")
        await clear_chat_button_locator.click(timeout=CLICK_TIMEOUT_MS)
        check_client_disconnected("清空聊天 - 点击\"清空聊天\"后: ")
        
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
        logger.info(f"[{req_id}] 点击\"继续\"按钮 (在对话框中): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
        await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
    
    check_client_disconnected("清空聊天 - 点击\"继续\"后: ")

    # 等待对话框消失
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


async def _verify_chat_cleared(req_id: str, page: AsyncPage, check_client_disconnected: Callable, logger) -> None:
    """验证聊天已清空"""
    last_response_container = page.locator(RESPONSE_CONTAINER_SELECTOR).last
    await asyncio.sleep(0.5)
    check_client_disconnected("After Clear Post-Delay: ")
    try:
        await expect_async(last_response_container).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS - 500)
        logger.info(f"[{req_id}] ✅ 聊天已成功清空 (验证通过 - 最后响应容器隐藏)。")
    except Exception as verify_err:
        logger.warning(f"[{req_id}] ⚠️ 警告: 清空聊天验证失败 (最后响应容器未隐藏): {verify_err}")


async def _adjust_request_parameters(req_id: str, page: AsyncPage, request: ChatCompletionRequest,
                                   context: dict, check_client_disconnected: Callable) -> None:
    """调整所有请求参数"""
    if not page or page.is_closed():
        return
    
    from server import logger
    logger.info(f"[{req_id}] 开始调整所有请求参数（包括恢复默认值）...")
    
    # 调整温度 - 总是设置，使用请求值或默认值
    temperature_to_set = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    logger.info(f"[{req_id}] 温度参数 - 请求值: {request.temperature}, 实际使用: {temperature_to_set} ({'请求值' if request.temperature is not None else '默认值'})")
    await _adjust_temperature_parameter_with_value(
        req_id, page, temperature_to_set, context['page_params_cache'], 
        context['params_cache_lock'], check_client_disconnected
    )
    check_client_disconnected("温度调整 - 逻辑完成后: ")
    
    # 调整最大输出Token - 总是设置，使用请求值或默认值
    max_tokens_to_set = request.max_output_tokens if request.max_output_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
    logger.info(f"[{req_id}] 最大输出Token参数 - 请求值: {request.max_output_tokens}, 实际使用: {max_tokens_to_set} ({'请求值' if request.max_output_tokens is not None else '默认值'})")
    await _adjust_max_tokens_parameter_with_value(
        req_id, page, max_tokens_to_set, context['page_params_cache'], 
        context['params_cache_lock'], context['model_id_to_use'], 
        context['parsed_model_list'], check_client_disconnected
    )
    check_client_disconnected("最大输出Token调整 - 逻辑完成后: ")
    
    # 调整停止序列 - 总是设置，使用请求值或默认值
    stop_to_set = request.stop if request.stop is not None else DEFAULT_STOP_SEQUENCES
    logger.info(f"[{req_id}] 停止序列参数 - 请求值: {request.stop}, 实际使用: {stop_to_set} ({'请求值' if request.stop is not None else '默认值'})")
    await _adjust_stop_sequences_parameter_with_value(
        req_id, page, stop_to_set, context['page_params_cache'], 
        context['params_cache_lock'], check_client_disconnected
    )
    check_client_disconnected("停止序列调整 - 逻辑完成后: ")
    
    # 调整Top P - 总是设置，使用请求值或默认值
    top_p_to_set = request.top_p if request.top_p is not None else DEFAULT_TOP_P
    logger.info(f"[{req_id}] Top P参数 - 请求值: {request.top_p}, 实际使用: {top_p_to_set} ({'请求值' if request.top_p is not None else '默认值'})")
    await _adjust_top_p_parameter_with_value(req_id, page, top_p_to_set, check_client_disconnected)


async def _adjust_temperature_parameter_with_value(req_id: str, page: AsyncPage, temperature: float, 
                                                   page_params_cache: dict, params_cache_lock: asyncio.Lock, 
                                                   check_client_disconnected: Callable) -> None:
    """调整温度参数"""
    from server import logger
    
    async with params_cache_lock:
        logger.info(f"[{req_id}] 检查并调整温度设置...")
        clamped_temp = max(0.0, min(2.0, temperature))
        if clamped_temp != temperature:
            logger.warning(f"[{req_id}] 请求的温度 {temperature} 超出范围 [0, 2]，已调整为 {clamped_temp}")
        
        cached_temp = page_params_cache.get("temperature")
        if cached_temp is not None and abs(cached_temp - clamped_temp) < 0.001:
            logger.info(f"[{req_id}] 温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 一致。跳过页面交互。")
            return
        
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
            logger.error(f"[{req_id}] 转换温度值为浮点数时出错. 错误: {ve}。清除缓存中的温度。")
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


async def _adjust_max_tokens_parameter_with_value(req_id: str, page: AsyncPage, max_tokens: int,
                                                   page_params_cache: dict, params_cache_lock: asyncio.Lock,
                                                   model_id_to_use: str, parsed_model_list: list,
                                                   check_client_disconnected: Callable) -> None:
    """调整最大输出Token参数"""
    from server import logger
    
    async with params_cache_lock:
        logger.info(f"[{req_id}] 检查并调整最大输出 Token 设置...")
        min_val_for_tokens = 1
        max_val_for_tokens_from_model = 65536
        
        if model_id_to_use and parsed_model_list:
            current_model_data = next((m for m in parsed_model_list if m.get("id") == model_id_to_use), None)
            if current_model_data and current_model_data.get("supported_max_output_tokens") is not None:
                try:
                    supported_tokens = int(current_model_data["supported_max_output_tokens"])
                    if supported_tokens > 0: 
                        max_val_for_tokens_from_model = supported_tokens
                    else: 
                        logger.warning(f"[{req_id}] 模型 {model_id_to_use} supported_max_output_tokens 无效: {supported_tokens}")
                except (ValueError, TypeError): 
                    logger.warning(f"[{req_id}] 模型 {model_id_to_use} supported_max_output_tokens 解析失败")
        
        clamped_max_tokens = max(min_val_for_tokens, min(max_val_for_tokens_from_model, max_tokens))
        if clamped_max_tokens != max_tokens:
            logger.warning(f"[{req_id}] 请求的最大输出 Tokens {max_tokens} 超出模型范围，已调整为 {clamped_max_tokens}")
        
        cached_max_tokens = page_params_cache.get("max_output_tokens")
        if cached_max_tokens is not None and cached_max_tokens == clamped_max_tokens:
            logger.info(f"[{req_id}] 最大输出 Tokens ({clamped_max_tokens}) 与缓存值一致。跳过页面交互。")
            return
        
        max_tokens_input_locator = page.locator(MAX_OUTPUT_TOKENS_SELECTOR)
        
        try:
            await expect_async(max_tokens_input_locator).to_be_visible(timeout=5000)
            check_client_disconnected("最大输出Token调整 - 输入框可见后: ")
            
            current_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
            current_max_tokens_int = int(current_max_tokens_str)
            
            if current_max_tokens_int == clamped_max_tokens:
                page_params_cache["max_output_tokens"] = current_max_tokens_int
            else:
                await max_tokens_input_locator.fill(str(clamped_max_tokens), timeout=5000)
                check_client_disconnected("最大输出Token调整 - 填充输入框后: ")
                
                await asyncio.sleep(0.1)
                new_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
                new_max_tokens_int = int(new_max_tokens_str)
                
                if new_max_tokens_int == clamped_max_tokens:
                    logger.info(f"[{req_id}] ✅ 最大输出 Tokens 已成功更新为: {new_max_tokens_int}")
                    page_params_cache["max_output_tokens"] = new_max_tokens_int
                else:
                    logger.warning(f"[{req_id}] ⚠️ 最大输出 Tokens 更新后验证失败")
                    page_params_cache.pop("max_output_tokens", None)
                    await save_error_snapshot(f"max_tokens_verify_fail_{req_id}")
                    
        except (ValueError, PlaywrightAsyncError, ClientDisconnectedError, Exception) as e:
            logger.error(f"[{req_id}] 调整最大输出 Tokens 时出错: {e}")
            page_params_cache.pop("max_output_tokens", None)
            if isinstance(e, ClientDisconnectedError):
                raise
            await save_error_snapshot(f"max_tokens_error_{req_id}")


async def _adjust_stop_sequences_parameter_with_value(req_id: str, page: AsyncPage, stop_sequences,
                                         page_params_cache: dict, params_cache_lock: asyncio.Lock,
                                         check_client_disconnected: Callable) -> None:
    """调整停止序列参数"""
    from server import logger
    
    async with params_cache_lock:
        logger.info(f"[{req_id}] 检查并设置停止序列...")
        
        # 处理不同类型的stop_sequences输入
        normalized_requested_stops = set()
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                # 单个字符串
                if stop_sequences.strip():
                    normalized_requested_stops.add(stop_sequences.strip())
            elif isinstance(stop_sequences, list):
                # 字符串列表
                for s in stop_sequences:
                    if isinstance(s, str) and s.strip():
                        normalized_requested_stops.add(s.strip())
        
        cached_stops_set = page_params_cache.get("stop_sequences")
        
        if cached_stops_set is not None and cached_stops_set == normalized_requested_stops:
            logger.info(f"[{req_id}] 请求的停止序列与缓存值一致。跳过页面交互。")
            return
        
        stop_input_locator = page.locator(STOP_SEQUENCE_INPUT_SELECTOR)
        remove_chip_buttons_locator = page.locator(MAT_CHIP_REMOVE_BUTTON_SELECTOR)
        
        try:
            # 清空已有的停止序列
            initial_chip_count = await remove_chip_buttons_locator.count()
            removed_count = 0
            max_removals = initial_chip_count + 5
            
            while await remove_chip_buttons_locator.count() > 0 and removed_count < max_removals:
                check_client_disconnected("停止序列清除 - 循环开始: ")
                try:
                    await remove_chip_buttons_locator.first.click(timeout=2000)
                    removed_count += 1
                    await asyncio.sleep(0.15)
                except Exception: 
                    break
            
            # 添加新的停止序列
            if normalized_requested_stops:
                await expect_async(stop_input_locator).to_be_visible(timeout=5000)
                for seq in normalized_requested_stops:
                    await stop_input_locator.fill(seq, timeout=3000)
                    await stop_input_locator.press("Enter", timeout=3000)
                    await asyncio.sleep(0.2)
            
            page_params_cache["stop_sequences"] = normalized_requested_stops
            logger.info(f"[{req_id}] 停止序列缓存已更新")
            
        except (PlaywrightAsyncError, ClientDisconnectedError, Exception) as e:
            logger.error(f"[{req_id}] 设置停止序列时出错: {e}")
            page_params_cache.pop("stop_sequences", None)
            if isinstance(e, ClientDisconnectedError):
                raise
            await save_error_snapshot(f"stop_sequence_error_{req_id}")


async def _adjust_top_p_parameter_with_value(req_id: str, page: AsyncPage, top_p: float, 
                                                check_client_disconnected: Callable) -> None:
    """调整Top P参数"""
    from server import logger
    
    logger.info(f"[{req_id}] 检查并调整 Top P 设置...")
    clamped_top_p = max(0.0, min(1.0, top_p))
    
    if abs(clamped_top_p - top_p) > 1e-9:
        logger.warning(f"[{req_id}] 请求的 Top P {top_p} 超出范围 [0, 1]，已调整为 {clamped_top_p}")
    
    top_p_input_locator = page.locator(TOP_P_INPUT_SELECTOR)
    try:
        await expect_async(top_p_input_locator).to_be_visible(timeout=5000)
        check_client_disconnected("Top P 调整 - 输入框可见后: ")
        
        current_top_p_str = await top_p_input_locator.input_value(timeout=3000)
        current_top_p_float = float(current_top_p_str)
        
        if abs(current_top_p_float - clamped_top_p) > 1e-9:
            await top_p_input_locator.fill(str(clamped_top_p), timeout=5000)
            check_client_disconnected("Top P 调整 - 填充输入框后: ")
            logger.info(f"[{req_id}] ✅ Top P 已更新为: {clamped_top_p}")
        else:
            logger.info(f"[{req_id}] Top P 值一致，无需更改")
            
    except (ValueError, PlaywrightAsyncError, ClientDisconnectedError, Exception) as e:
        logger.error(f"[{req_id}] 调整 Top P 时出错: {e}")
        if isinstance(e, ClientDisconnectedError):
            raise
        await save_error_snapshot(f"top_p_error_{req_id}")


async def _submit_prompt(req_id: str, page: AsyncPage, prepared_prompt: str, check_client_disconnected: Callable) -> None:
    """提交提示到页面"""
    from server import logger
    
    logger.info(f"[{req_id}] 填充并提交提示 ({len(prepared_prompt)} chars)...")
    prompt_textarea_locator = page.locator(PROMPT_TEXTAREA_SELECTOR)
    autosize_wrapper_locator = page.locator('ms-prompt-input-wrapper ms-autosize-textarea')
    submit_button_locator = page.locator(SUBMIT_BUTTON_SELECTOR)
    
    try:
        await expect_async(prompt_textarea_locator).to_be_visible(timeout=5000)
        check_client_disconnected("After Input Visible: ")
        
        # 使用 JavaScript 填充文本
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
        check_client_disconnected("After Input Fill: ")

        # 等待发送按钮启用
        wait_timeout_ms_submit_enabled = 40000
        try:
            check_client_disconnected("填充提示后等待发送按钮启用 - 前置检查: ")
            await expect_async(submit_button_locator).to_be_enabled(timeout=wait_timeout_ms_submit_enabled)
            logger.info(f"[{req_id}] ✅ 发送按钮已启用。")
        except PlaywrightAsyncError as e_pw_enabled:
            logger.error(f"[{req_id}] ❌ 等待发送按钮启用超时或错误: {e_pw_enabled}")
            await save_error_snapshot(f"submit_button_enable_timeout_{req_id}")
            raise

        check_client_disconnected("After Submit Button Enabled: ")
        await asyncio.sleep(0.3)
        
        # 尝试使用快捷键提交
        submitted_successfully = await _try_shortcut_submit(req_id, page, prompt_textarea_locator, check_client_disconnected, logger)
        
        # 如果快捷键失败，使用按钮点击
        if not submitted_successfully:
            logger.info(f"[{req_id}] 快捷键提交失败，尝试点击提交按钮...")
            try:
                await submit_button_locator.click(timeout=5000)
                logger.info(f"[{req_id}] ✅ 提交按钮点击完成。")
            except Exception as click_err:
                logger.error(f"[{req_id}] ❌ 提交按钮点击失败: {click_err}")
                await save_error_snapshot(f"submit_button_click_fail_{req_id}")
                raise
        
        check_client_disconnected("After Submit: ")
        
    except Exception as e_input_submit:
        logger.error(f"[{req_id}] 输入和提交过程中发生错误: {e_input_submit}")
        if not isinstance(e_input_submit, ClientDisconnectedError):
            await save_error_snapshot(f"input_submit_error_{req_id}")
        raise


async def _try_shortcut_submit(req_id: str, page: AsyncPage, prompt_textarea_locator: Locator, 
                              check_client_disconnected: Callable, logger) -> bool:
    """尝试使用快捷键提交"""
    try:
        # 检测操作系统
        host_os_from_launcher = os.environ.get('HOST_OS_FOR_SHORTCUT')
        is_mac_determined = False
        
        if host_os_from_launcher == "Darwin":
            is_mac_determined = True
        elif host_os_from_launcher in ["Windows", "Linux"]:
            is_mac_determined = False
        else:
            # 使用浏览器检测
            try:
                user_agent_data_platform = await page.evaluate("() => navigator.userAgentData?.platform || ''")
            except Exception:
                user_agent_string = await page.evaluate("() => navigator.userAgent || ''")
                user_agent_string_lower = user_agent_string.lower()
                if "macintosh" in user_agent_string_lower or "mac os x" in user_agent_string_lower:
                    user_agent_data_platform = "macOS"
                else:
                    user_agent_data_platform = "Other"
            
            is_mac_determined = "mac" in user_agent_data_platform.lower()
        
        shortcut_modifier = "Meta" if is_mac_determined else "Control"
        shortcut_key = "Enter"
        
        logger.info(f"[{req_id}] 使用快捷键: {shortcut_modifier}+{shortcut_key}")
        
        await prompt_textarea_locator.focus(timeout=5000)
        check_client_disconnected("After Input Focus: ")
        await asyncio.sleep(0.1)
        
        # 记录提交前的输入框内容，用于验证
        original_content = ""
        try:
            original_content = await prompt_textarea_locator.input_value(timeout=2000) or ""
        except Exception:
            # 如果无法获取原始内容，仍然尝试提交
            pass
        
        try:
            await page.keyboard.press(f'{shortcut_modifier}+{shortcut_key}')
        except Exception:
            # 尝试分步按键
            await page.keyboard.down(shortcut_modifier)
            await asyncio.sleep(0.05)
            await page.keyboard.press(shortcut_key)
            await asyncio.sleep(0.05)
            await page.keyboard.up(shortcut_modifier)
        
        check_client_disconnected("After Shortcut Press: ")
        
        # 等待更长时间让提交完成
        await asyncio.sleep(2.0)
        
        # 多种方式验证提交是否成功
        submission_success = False
        
        try:
            # 方法1: 检查原始输入框是否清空
            current_content = await prompt_textarea_locator.input_value(timeout=2000) or ""
            if original_content and not current_content.strip():
                logger.info(f"[{req_id}] 验证方法1: 输入框已清空，快捷键提交成功")
                submission_success = True
            
            # 方法2: 检查提交按钮状态
            if not submission_success:
                submit_button_locator = page.locator(SUBMIT_BUTTON_SELECTOR)
                try:
                    is_disabled = await submit_button_locator.is_disabled(timeout=2000)
                    if is_disabled:
                        logger.info(f"[{req_id}] 验证方法2: 提交按钮已禁用，快捷键提交成功")
                        submission_success = True
                except Exception:
                    pass
            
            # 方法3: 检查是否有响应容器出现
            if not submission_success:
                try:
                    response_container = page.locator(RESPONSE_CONTAINER_SELECTOR)
                    container_count = await response_container.count()
                    if container_count > 0:
                        # 检查最后一个容器是否是新的
                        last_container = response_container.last
                        if await last_container.is_visible(timeout=1000):
                            logger.info(f"[{req_id}] 验证方法3: 检测到响应容器，快捷键提交成功")
                            submission_success = True
                except Exception:
                    pass
            
        except Exception as verify_err:
            logger.warning(f"[{req_id}] 快捷键提交验证过程出错: {verify_err}")
            # 出错时假定提交成功，让后续流程继续
            submission_success = True
        
        if submission_success:
            logger.info(f"[{req_id}] ✅ 快捷键提交成功")
            return True
        else:
            logger.warning(f"[{req_id}] ⚠️ 快捷键提交验证失败")
            return False
            
    except Exception as shortcut_err:
        logger.warning(f"[{req_id}] 快捷键提交失败: {shortcut_err}")
        return False


async def _handle_response_processing(req_id: str, request: ChatCompletionRequest, page: AsyncPage, 
                                    context: dict, result_future: Future, 
                                    submit_button_locator: Locator, check_client_disconnected: Callable) -> Optional[Tuple[Event, Locator, Callable]]:
    """处理响应生成"""
    from server import logger
    
    is_streaming = request.stream
    current_ai_studio_model_id = context.get('current_ai_studio_model_id')
    
    # 检查是否使用辅助流
    stream_port = os.environ.get('STREAM_PORT')
    use_stream = stream_port != '0'
    
    if use_stream:
        return await _handle_auxiliary_stream_response(req_id, request, context, result_future, submit_button_locator, check_client_disconnected)
    else:
        return await _handle_playwright_response(req_id, request, page, context, result_future, submit_button_locator, check_client_disconnected)


async def _handle_auxiliary_stream_response(req_id: str, request: ChatCompletionRequest, context: dict, 
                                          result_future: Future, submit_button_locator: Locator, 
                                          check_client_disconnected: Callable) -> Optional[Tuple[Event, Locator, Callable]]:
    """使用辅助流处理响应"""
    from server import logger
    
    is_streaming = request.stream
    current_ai_studio_model_id = context.get('current_ai_studio_model_id')
    
    def generate_random_string(length):
        charset = "abcdefghijklmnopqrstuvwxyz0123456789"
        return ''.join(random.choice(charset) for _ in range(length))

    if is_streaming:
        try:
            completion_event = Event()
            
            async def create_stream_generator_from_helper(event_to_set: Event) -> AsyncGenerator[str, None]:
                last_reason_pos = 0
                last_body_pos = 0
                model_name_for_stream = current_ai_studio_model_id or MODEL_NAME
                chat_completion_id = f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}-{random.randint(100, 999)}"
                created_timestamp = int(time.time())
                
                # 用于收集完整内容以计算usage
                full_reasoning_content = ""
                full_body_content = ""

                try:
                    async for raw_data in use_stream_response(req_id):
                        # 检查客户端是否断开连接
                        try:
                            check_client_disconnected(f"流式生成器循环 ({req_id}): ")
                        except ClientDisconnectedError:
                            logger.info(f"[{req_id}] 客户端断开连接，终止流式生成")
                            break
                        
                        # 确保 data 是字典类型
                        if isinstance(raw_data, str):
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                logger.warning(f"[{req_id}] 无法解析流数据JSON: {raw_data}")
                                continue
                        elif isinstance(raw_data, dict):
                            data = raw_data
                        else:
                            logger.warning(f"[{req_id}] 未知的流数据类型: {type(raw_data)}")
                            continue
                        
                        # 确保必要的键存在
                        if not isinstance(data, dict):
                            logger.warning(f"[{req_id}] 数据不是字典类型: {data}")
                            continue
                        
                        reason = data.get("reason", "")
                        body = data.get("body", "")
                        done = data.get("done", False)
                        function = data.get("function", [])
                        
                        # 更新完整内容记录
                        if reason:
                            full_reasoning_content = reason
                        if body:
                            full_body_content = body
                        
                        # 处理推理内容
                        if len(reason) > last_reason_pos:
                            output = {
                                "id": chat_completion_id,
                                "object": "chat.completion.chunk",
                                "model": model_name_for_stream,
                                "created": created_timestamp,
                                "choices":[{
                                    "delta":{
                                        "role": "assistant",
                                        "content": None,
                                        "reasoning_content": reason[last_reason_pos:],
                                    },
                                    "finish_reason": None,
                                    "native_finish_reason": None,
                                }]
                            }
                            last_reason_pos = len(reason)
                            yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        
                        # 处理主体内容
                        if len(body) > last_body_pos:
                            finish_reason_val = None
                            if done:
                                finish_reason_val = "stop"
                            
                            delta_content = {"role": "assistant", "content": body[last_body_pos:]}
                            choice_item = {
                                "delta": delta_content,
                                "finish_reason": finish_reason_val,
                                "native_finish_reason": finish_reason_val,
                            }

                            if done and function and len(function) > 0:
                                tool_calls_list = []
                                for func_idx, function_call_data in enumerate(function):
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
                                delta_content["content"] = None

                            output = {
                                "id": chat_completion_id,
                                "object": "chat.completion.chunk",
                                "model": model_name_for_stream,
                                "created": created_timestamp,
                                "choices": [choice_item]
                            }
                            last_body_pos = len(body)
                            yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        
                        # 处理只有done=True但没有新内容的情况（仅有函数调用或纯结束）
                        elif done:
                            # 如果有函数调用但没有新的body内容
                            if function and len(function) > 0:
                                delta_content = {"role": "assistant", "content": None}
                                tool_calls_list = []
                                for func_idx, function_call_data in enumerate(function):
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
                                choice_item = {
                                    "delta": delta_content,
                                    "finish_reason": "tool_calls",
                                    "native_finish_reason": "tool_calls",
                                }
                            else:
                                # 纯结束，没有新内容和函数调用
                                choice_item = {
                                    "delta": {"role": "assistant"},
                                    "finish_reason": "stop",
                                    "native_finish_reason": "stop",
                                }

                            output = {
                                "id": chat_completion_id,
                                "object": "chat.completion.chunk",
                                "model": model_name_for_stream,
                                "created": created_timestamp,
                                "choices": [choice_item]
                            }
                            yield f"data: {json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n\n"
                
                except ClientDisconnectedError:
                    logger.info(f"[{req_id}] 流式生成器中检测到客户端断开连接")
                except Exception as e:
                    logger.error(f"[{req_id}] 流式生成器处理过程中发生错误: {e}", exc_info=True)
                    # 发送错误信息给客户端
                    try:
                        error_chunk = {
                            "id": chat_completion_id,
                            "object": "chat.completion.chunk",
                            "model": model_name_for_stream,
                            "created": created_timestamp,
                            "choices": [{
                                "delta": {"role": "assistant", "content": f"\n\n[错误: {str(e)}]"},
                                "finish_reason": "stop",
                                "native_finish_reason": "stop",
                            }]
                        }
                        yield f"data: {json.dumps(error_chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    except Exception:
                        pass  # 如果无法发送错误信息，继续处理结束逻辑
                finally:
                    # 计算usage统计
                    try:
                        usage_stats = calculate_usage_stats(
                            [msg.model_dump() for msg in request.messages],
                            full_body_content,
                            full_reasoning_content
                        )
                        logger.info(f"[{req_id}] 计算的token使用统计: {usage_stats}")
                        
                        # 发送带usage的最终chunk
                        final_chunk = {
                            "id": chat_completion_id,
                            "object": "chat.completion.chunk",
                            "model": model_name_for_stream,
                            "created": created_timestamp,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                                "native_finish_reason": "stop"
                            }],
                            "usage": usage_stats
                        }
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        logger.info(f"[{req_id}] 已发送带usage统计的最终chunk")
                        
                    except Exception as usage_err:
                        logger.error(f"[{req_id}] 计算或发送usage统计时出错: {usage_err}")
                    
                    # 确保总是发送 [DONE] 标记
                    try:
                        logger.info(f"[{req_id}] 流式生成器完成，发送 [DONE] 标记")
                        yield "data: [DONE]\n\n"
                    except Exception as done_err:
                        logger.error(f"[{req_id}] 发送 [DONE] 标记时出错: {done_err}")
                    
                    # 确保事件被设置
                    if not event_to_set.is_set():
                        event_to_set.set()
                        logger.info(f"[{req_id}] 流式生成器完成事件已设置")

            stream_gen_func = create_stream_generator_from_helper(completion_event)
            if not result_future.done():
                result_future.set_result(StreamingResponse(stream_gen_func, media_type="text/event-stream"))
            else:
                if not completion_event.is_set():
                    completion_event.set()
            
            return completion_event, submit_button_locator, check_client_disconnected

        except Exception as e:
            logger.error(f"[{req_id}] 从队列获取流式数据时出错: {e}", exc_info=True)
            if completion_event and not completion_event.is_set():
                completion_event.set()
            raise

    else:  # 非流式
        content = None
        reasoning_content = None
        functions = None
        final_data_from_aux_stream = None

        async for raw_data in use_stream_response(req_id):
            check_client_disconnected(f"非流式辅助流 - 循环中 ({req_id}): ")
            
            # 确保 data 是字典类型
            if isinstance(raw_data, str):
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.warning(f"[{req_id}] 无法解析非流式数据JSON: {raw_data}")
                    continue
            elif isinstance(raw_data, dict):
                data = raw_data
            else:
                logger.warning(f"[{req_id}] 非流式未知数据类型: {type(raw_data)}")
                continue
            
            # 确保数据是字典类型
            if not isinstance(data, dict):
                logger.warning(f"[{req_id}] 非流式数据不是字典类型: {data}")
                continue
                
            final_data_from_aux_stream = data
            if data.get("done"):
                content = data.get("body")
                reasoning_content = data.get("reason")
                functions = data.get("function")
                break
        
        if final_data_from_aux_stream and final_data_from_aux_stream.get("reason") == "internal_timeout":
            logger.error(f"[{req_id}] 非流式请求通过辅助流失败: 内部超时")
            raise HTTPException(status_code=502, detail=f"[{req_id}] 辅助流处理错误 (内部超时)")

        if final_data_from_aux_stream and final_data_from_aux_stream.get("done") is True and content is None:
             logger.error(f"[{req_id}] 非流式请求通过辅助流完成但未提供内容")
             raise HTTPException(status_code=502, detail=f"[{req_id}] 辅助流完成但未提供内容")

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
            message_payload["content"] = None
        
        if reasoning_content:
            message_payload["reasoning_content"] = reasoning_content

        # 计算token使用统计
        usage_stats = calculate_usage_stats(
            [msg.model_dump() for msg in request.messages],
            content or "",
            reasoning_content
        )

        response_payload = {
            "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name_for_json,
            "choices": [{
                "index": 0,
                "message": message_payload,
                "finish_reason": finish_reason_val,
                "native_finish_reason": finish_reason_val,
            }],
            "usage": usage_stats
        }

        if not result_future.done():
            result_future.set_result(JSONResponse(content=response_payload))
        return None


async def _handle_playwright_response(req_id: str, request: ChatCompletionRequest, page: AsyncPage, 
                                    context: dict, result_future: Future, submit_button_locator: Locator, 
                                    check_client_disconnected: Callable) -> Optional[Tuple[Event, Locator, Callable]]:
    """使用Playwright处理响应"""
    from server import logger
    
    is_streaming = request.stream
    current_ai_studio_model_id = context.get('current_ai_studio_model_id')
    
    logger.info(f"[{req_id}] 定位响应元素...")
    response_container = page.locator(RESPONSE_CONTAINER_SELECTOR).last
    response_element = response_container.locator(RESPONSE_TEXT_SELECTOR)
    
    try:
        await expect_async(response_container).to_be_attached(timeout=20000)
        check_client_disconnected("After Response Container Attached: ")
        await expect_async(response_element).to_be_attached(timeout=90000)
        logger.info(f"[{req_id}] 响应元素已定位。")
    except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as locate_err:
        if isinstance(locate_err, ClientDisconnectedError):
            raise
        logger.error(f"[{req_id}] ❌ 错误: 定位响应元素失败或超时: {locate_err}")
        await save_error_snapshot(f"response_locate_error_{req_id}")
        raise HTTPException(status_code=502, detail=f"[{req_id}] 定位AI Studio响应元素失败: {locate_err}")
    except Exception as locate_exc:
        logger.exception(f"[{req_id}] ❌ 错误: 定位响应元素时意外错误")
        await save_error_snapshot(f"response_locate_unexpected_{req_id}")
        raise HTTPException(status_code=500, detail=f"[{req_id}] 定位响应元素时意外错误: {locate_exc}")

    check_client_disconnected("After Response Element Located: ")

    if is_streaming:
        completion_event = Event()

        async def create_response_stream_generator():
            try:
                # 获取必需的定位器
                input_field_locator = page.locator(PROMPT_TEXTAREA_SELECTOR)
                edit_button_locator = page.locator(EDIT_MESSAGE_BUTTON_SELECTOR)

                completion_detected = await _wait_for_response_completion(
                    page,
                    input_field_locator,
                    submit_button_locator,
                    edit_button_locator,
                    req_id,
                    check_client_disconnected,
                    req_id  # current_chat_id
                )

                if not completion_detected:
                    logger.warning(f"[{req_id}] 响应完成检测失败，尝试获取当前内容")

                # 获取最终响应内容
                final_content = await _get_final_response_content(page, req_id, check_client_disconnected)
                
                # 生成流式响应
                words = final_content.split()
                for i, word in enumerate(words):
                    # 检查客户端是否断开连接
                    try:
                        check_client_disconnected(f"Playwright流式生成器循环 ({req_id}): ")
                    except ClientDisconnectedError:
                        logger.info(f"[{req_id}] Playwright流式生成器中检测到客户端断开连接")
                        break
                    
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield generate_sse_chunk(chunk_content, req_id, current_ai_studio_model_id or MODEL_NAME)
                    await asyncio.sleep(0.05)
                
                # 计算并发送带usage的完成块
                usage_stats = calculate_usage_stats(
                    [msg.model_dump() for msg in request.messages],
                    final_content,
                    ""  # Playwright模式没有reasoning content
                )
                logger.info(f"[{req_id}] Playwright非流式计算的token使用统计: {usage_stats}")
                
                # 发送带usage的完成块
                yield generate_sse_stop_chunk(req_id, current_ai_studio_model_id or MODEL_NAME, "stop", usage_stats)
                
            except ClientDisconnectedError:
                logger.info(f"[{req_id}] Playwright流式生成器中检测到客户端断开连接")
            except Exception as e:
                logger.error(f"[{req_id}] Playwright流式生成器处理过程中发生错误: {e}", exc_info=True)
                # 发送错误信息给客户端
                try:
                    yield generate_sse_chunk(f"\n\n[错误: {str(e)}]", req_id, current_ai_studio_model_id or MODEL_NAME)
                    yield generate_sse_stop_chunk(req_id, current_ai_studio_model_id or MODEL_NAME)
                except Exception:
                    pass  # 如果无法发送错误信息，继续处理结束逻辑
            finally:
                # 确保事件被设置
                if not completion_event.is_set():
                    completion_event.set()
                    logger.info(f"[{req_id}] Playwright流式生成器完成事件已设置")

        stream_gen_func = create_response_stream_generator()
        if not result_future.done():
            result_future.set_result(StreamingResponse(stream_gen_func, media_type="text/event-stream"))
        
        return completion_event, submit_button_locator, check_client_disconnected
    else:
        # 获取必需的定位器
        input_field_locator = page.locator(PROMPT_TEXTAREA_SELECTOR)
        edit_button_locator = page.locator(EDIT_MESSAGE_BUTTON_SELECTOR)

        # 等待响应完成
        completion_detected = await _wait_for_response_completion(
            page,
            input_field_locator,
            submit_button_locator,
            edit_button_locator,
            req_id,
            check_client_disconnected,
            req_id  # current_chat_id
        )

        if not completion_detected:
            logger.warning(f"[{req_id}] 响应完成检测失败，尝试获取当前内容")

        # 获取最终响应内容
        final_content = await _get_final_response_content(page, req_id, check_client_disconnected)
        
        # 计算token使用统计
        usage_stats = calculate_usage_stats(
            [msg.model_dump() for msg in request.messages],
            final_content,
            ""  # Playwright模式没有reasoning content
        )
        logger.info(f"[{req_id}] Playwright非流式计算的token使用统计: {usage_stats}")
        
        response_payload = {
            "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": current_ai_studio_model_id or MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final_content},
                "finish_reason": "stop"
            }],
            "usage": usage_stats
        }
        
        if not result_future.done():
            result_future.set_result(JSONResponse(content=response_payload))
        
        return None


async def _cleanup_request_resources(req_id: str, disconnect_check_task: Optional[asyncio.Task], 
                                   completion_event: Optional[Event], result_future: Future, 
                                   is_streaming: bool) -> None:
    """清理请求资源"""
    from server import logger
    
    if disconnect_check_task and not disconnect_check_task.done():
        disconnect_check_task.cancel()
        try: 
            await disconnect_check_task
        except asyncio.CancelledError: 
            pass
        except Exception as task_clean_err: 
            logger.error(f"[{req_id}] 清理任务时出错: {task_clean_err}")
    
    logger.info(f"[{req_id}] 处理完成。")
    
    if is_streaming and completion_event and not completion_event.is_set() and (result_future.done() and result_future.exception() is not None):
         logger.warning(f"[{req_id}] 流式请求异常，确保完成事件已设置。")
         completion_event.set()


async def _process_request_refactored(
    req_id: str,
    request: ChatCompletionRequest,
    http_request: Request,
    result_future: Future
) -> Optional[Tuple[Event, Locator, Callable[[str], bool]]]:
    """核心请求处理函数 - 重构版本"""
    
    # 1. 初始化请求上下文
    context = await _initialize_request_context(req_id, request)
    
    # 2. 分析模型需求
    context = await _analyze_model_requirements(req_id, context, request)
    
    # 3. 设置断开连接监控
    client_disconnected_event, disconnect_check_task, check_client_disconnected = await _setup_disconnect_monitoring(
        req_id, http_request, result_future
    )
    
    page = context['page']
    submit_button_locator = page.locator(SUBMIT_BUTTON_SELECTOR) if page else None
    completion_event = None
    
    try:
        # 4. 验证页面状态
        await _validate_page_status(req_id, context, check_client_disconnected)
        
        # 5. 处理模型切换
        context = await _handle_model_switching(req_id, context, check_client_disconnected)
        
        # 6. 处理参数缓存
        await _handle_parameter_cache(req_id, context)
        
        # 7. 准备和验证请求
        system_prompt_content, prepared_prompt = await _prepare_and_validate_request(req_id, request, check_client_disconnected)
        
        # 8. 清空聊天记录
        await _clear_chat_history(req_id, page, check_client_disconnected)
        check_client_disconnected("After Clear Chat: ")
        
        # 9. 设置系统提示 
        await set_system_prompt_in_page(
            page,
            system_prompt_content if system_prompt_content else "",
            req_id,
            check_client_disconnected
        )
        check_client_disconnected("After Set System Prompt: ")
        
        # 10. 调整请求参数
        await _adjust_request_parameters(req_id, page, request, context, check_client_disconnected)
        check_client_disconnected("After Parameters Adjustment: ")
        
        # 11. 提交提示
        await _submit_prompt(req_id, page, prepared_prompt, check_client_disconnected)
        check_client_disconnected("After Input Submit: ")
        
        # 12. 处理响应
        response_result = await _handle_response_processing(
            req_id, request, page, context, result_future, submit_button_locator, check_client_disconnected
        )
        
        if response_result:
            completion_event, submit_button_locator, check_client_disconnected = response_result
        
        return completion_event, submit_button_locator, check_client_disconnected
        
    except ClientDisconnectedError as disco_err:
        context['logger'].info(f"[{req_id}] 捕获到客户端断开连接信号: {disco_err}")
        if not result_future.done():
             result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] Client disconnected during processing."))
    except HTTPException as http_err:
        context['logger'].warning(f"[{req_id}] 捕获到 HTTP 异常: {http_err.status_code} - {http_err.detail}")
        if not result_future.done(): 
            result_future.set_exception(http_err)
    except PlaywrightAsyncError as pw_err:
        context['logger'].error(f"[{req_id}] 捕获到 Playwright 错误: {pw_err}")
        await save_error_snapshot(f"process_playwright_error_{req_id}")
        if not result_future.done(): 
            result_future.set_exception(HTTPException(status_code=502, detail=f"[{req_id}] Playwright interaction failed: {pw_err}"))
    except asyncio.TimeoutError as timeout_err:
        context['logger'].error(f"[{req_id}] 捕获到操作超时: {timeout_err}")
        await save_error_snapshot(f"process_timeout_error_{req_id}")
        if not result_future.done(): 
            result_future.set_exception(HTTPException(status_code=504, detail=f"[{req_id}] Operation timed out: {timeout_err}"))
    except asyncio.CancelledError:
        context['logger'].info(f"[{req_id}] 任务被取消。")
        if not result_future.done(): 
            result_future.cancel("Processing task cancelled")
    except Exception as e:
        context['logger'].exception(f"[{req_id}] 捕获到意外错误")
        await save_error_snapshot(f"process_unexpected_error_{req_id}")
        if not result_future.done(): 
            result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Unexpected server error: {e}"))
    finally:
        # 12. 清理资源
        await _cleanup_request_resources(req_id, disconnect_check_task, completion_event, result_future, request.stream)
