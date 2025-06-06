"""
请求处理器模块
包含核心的请求处理逻辑
"""

import asyncio
import json
import os
import random
import time
from typing import Optional, Tuple, Callable, AsyncGenerator, Union, List
from asyncio import Event, Future

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from playwright.async_api import Page as AsyncPage, Locator, Error as PlaywrightAsyncError, expect as expect_async, TimeoutError

# --- 配置模块导入 ---
from config import *

# --- models模块导入 ---
from models import ChatCompletionRequest, ClientDisconnectedError, MessageContentItem

# --- browser_utils模块导入 ---
from browser_utils import (
    switch_ai_studio_model, 
    save_error_snapshot,
    _wait_for_response_completion,
    _get_final_response_content,
    detect_and_extract_page_error,
    set_system_prompt_in_page,
)

# --- api_utils模块导入 ---
from .utils import (
    validate_chat_request, 
    prepare_combined_prompt,
    generate_sse_chunk,
    generate_sse_stop_chunk,
    use_stream_response,
    calculate_usage_stats,
    is_incremental_messages
)


async def _initialize_request_context(req_id: str, request: ChatCompletionRequest) -> dict:
    """初始化请求上下文"""
    from server import (
        logger, page_instance, is_page_ready, parsed_model_list,
        current_ai_studio_model_id, model_switching_lock, page_params_cache,
        params_cache_lock, page_sync_cache_lock, last_api_messages_synced_to_page
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
        'page_sync_cache_lock': page_sync_cache_lock,
        'last_api_messages_synced_to_page': last_api_messages_synced_to_page,
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
        context['model_id_to_use'] = current_ai_studio_model_id
    
    return context


async def _setup_disconnect_monitoring(req_id: str, http_request: Request, result_future: Future) -> Tuple[Event, asyncio.Task, Callable]:
    """设置客户端断开连接监控"""
    from server import logger
    
    client_disconnected_event = Event()
    
    async def check_disconnect_periodically():
        while not client_disconnected_event.is_set():
            try:
                if await http_request.is_disconnected():
                    logger.info(f"[{req_id}] (Disco Check Task) 客户端断开。设置事件。")
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
        
        if model_actually_switched or \
           (current_ai_studio_model_id is not None and current_ai_studio_model_id != cached_model_for_params):
            
            action_taken = "Invalidating" if page_params_cache else "Initializing"
            logger.info(f"[{req_id}] {action_taken} parameter cache. Reason: Model context changed (switched this call: {model_actually_switched}, current model: {current_ai_studio_model_id}, cache model: {cached_model_for_params}).")
            
            page_params_cache.clear()
            if current_ai_studio_model_id:
                page_params_cache["last_known_model_id_for_params"] = current_ai_studio_model_id
        else:
            logger.debug(f"[{req_id}] Parameter cache for model '{cached_model_for_params}' remains valid (current model: '{current_ai_studio_model_id}', switched this call: {model_actually_switched}).")


async def _prepare_and_validate_request(req_id: str, request: ChatCompletionRequest, check_client_disconnected: Callable) -> Tuple[Optional[str], str]:
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
                logger.warning(f"[{req_id}] 等待\"清空聊天\"按钮可用失败: {e_enable}。")
        
        check_client_disconnected("清空聊天 - 可用性检查后: ")

        if can_attempt_clear:
            await clear_chat_button_locator.click(timeout=CLICK_TIMEOUT_MS)
            await expect_async(overlay_locator).to_be_visible(timeout=WAIT_FOR_ELEMENT_TIMEOUT_MS)
            await expect_async(confirm_button_locator).to_be_visible(timeout=CLICK_TIMEOUT_MS)
            await expect_async(confirm_button_locator).to_be_enabled(timeout=CLICK_TIMEOUT_MS)
            await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
            await expect_async(confirm_button_locator).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS)
            await expect_async(overlay_locator).to_be_hidden(timeout=1000)
            last_response_container = page.locator(RESPONSE_CONTAINER_SELECTOR).last
            await asyncio.sleep(0.5)
            await expect_async(last_response_container).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS - 500)
            logger.info(f"[{req_id}] ✅ 聊天已成功清空。")

    except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as e_clear:
        if isinstance(e_clear, ClientDisconnectedError): raise
        logger.error(f"[{req_id}] 清空聊天过程中发生错误: {e_clear}")
        await save_error_snapshot(f"clear_chat_error_{req_id}")
        raise HTTPException(status_code=502, detail=f"[{req_id}] Failed to clear chat on AI Studio: {e_clear}")
    except Exception as e_clear_unknown:
        logger.exception(f"[{req_id}] 清空聊天时发生未知错误")
        await save_error_snapshot(f"clear_chat_unknown_error_{req_id}")
        raise HTTPException(status_code=500, detail=f"[{req_id}] Unexpected error during clear chat: {e_clear_unknown}")


async def _adjust_request_parameters(req_id: str, page: AsyncPage, request: ChatCompletionRequest,
                                   context: dict, check_client_disconnected: Callable) -> None:
    """调整所有请求参数"""
    if not page or page.is_closed():
        return
    
    from server import logger
    logger.info(f"[{req_id}] 开始调整所有请求参数...")

    async with context['params_cache_lock']:
        # 调整温度
        await _adjust_temperature_parameter_with_value(req_id, page, request.temperature, context, check_client_disconnected)
        check_client_disconnected("温度调整后: ")
        
        # 调整最大输出Token
        await _adjust_max_tokens_parameter_with_value(req_id, page, request.max_output_tokens, context, check_client_disconnected)
        check_client_disconnected("最大Token调整后: ")
        
        # 调整停止序列
        await _adjust_stop_sequences_parameter_with_value(req_id, page, request.stop, context, check_client_disconnected)
        check_client_disconnected("停止序列调整后: ")
        
        # 调整Top P
        await _adjust_top_p_parameter_with_value(req_id, page, request.top_p, context, check_client_disconnected)
        check_client_disconnected("Top P调整后: ")


async def _adjust_temperature_parameter_with_value(req_id: str, page: AsyncPage, temperature: Optional[float], 
                                                   context: dict, check_client_disconnected: Callable) -> None:
    """调整温度参数"""
    from server import logger
    
    page_params_cache = context['page_params_cache']
    temp_to_set = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    logger.info(f"[{req_id}] 温度 - 请求值: {temperature}, 实际使用: {temp_to_set}")
    clamped_temp = max(0.0, min(2.0, temp_to_set))
    if clamped_temp != temp_to_set:
        logger.warning(f"[{req_id}] 温度 {temp_to_set} 超出范围 [0, 2]，已调整为 {clamped_temp}")
    
    cached_temp = page_params_cache.get("temperature")
    if cached_temp is not None and abs(cached_temp - clamped_temp) < 0.001:
        logger.info(f"[{req_id}] 温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 一致。跳过。")
        return
    
    temp_input_locator = page.locator(TEMPERATURE_INPUT_SELECTOR)
    try:
        await expect_async(temp_input_locator).to_be_visible(timeout=5000)
        await temp_input_locator.fill(str(clamped_temp), timeout=5000)
        page_params_cache["temperature"] = clamped_temp
        logger.info(f"[{req_id}] ✅ 温度已更新为: {clamped_temp}")
    except Exception as e:
        logger.error(f"[{req_id}] ❌ 调整温度时出错: {e}")
        page_params_cache.pop("temperature", None)
        await save_error_snapshot(f"temperature_error_{req_id}")


async def _adjust_max_tokens_parameter_with_value(req_id: str, page: AsyncPage, max_tokens: Optional[int],
                                                   context: dict, check_client_disconnected: Callable) -> None:
    """调整最大输出Token参数"""
    from server import logger
    
    page_params_cache = context['page_params_cache']
    model_id_to_use = context['model_id_to_use']
    parsed_model_list = context['parsed_model_list']
    tokens_to_set = max_tokens if max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS

    logger.info(f"[{req_id}] 最大Token - 请求值: {max_tokens}, 实际使用: {tokens_to_set}")
    
    min_val, max_val = 1, 65536
    if model_id_to_use and parsed_model_list:
        model_data = next((m for m in parsed_model_list if m.get("id") == model_id_to_use), None)
        if model_data and model_data.get("supported_max_output_tokens"):
            try: max_val = int(model_data["supported_max_output_tokens"])
            except: pass

    clamped_tokens = max(min_val, min(max_val, tokens_to_set))
    if clamped_tokens != tokens_to_set:
        logger.warning(f"[{req_id}] 最大Token {tokens_to_set} 超出范围 [{min_val}, {max_val}]，调整为 {clamped_tokens}")

    cached_tokens = page_params_cache.get("max_output_tokens")
    if cached_tokens is not None and cached_tokens == clamped_tokens:
        logger.info(f"[{req_id}] 最大Token ({clamped_tokens}) 与缓存值 ({cached_tokens}) 一致。跳过。")
        return

    tokens_input_locator = page.locator(MAX_OUTPUT_TOKENS_SELECTOR)
    try:
        await expect_async(tokens_input_locator).to_be_visible(timeout=5000)
        await tokens_input_locator.fill(str(clamped_tokens), timeout=5000)
        page_params_cache["max_output_tokens"] = clamped_tokens
        logger.info(f"[{req_id}] ✅ 最大Token已更新为: {clamped_tokens}")
    except Exception as e:
        logger.error(f"[{req_id}] ❌ 调整最大Token时出错: {e}")
        page_params_cache.pop("max_output_tokens", None)
        await save_error_snapshot(f"max_tokens_error_{req_id}")


async def _adjust_stop_sequences_parameter_with_value(req_id: str, page: AsyncPage, stop_sequences: Optional[Union[str, List[str]]],
                                         context: dict, check_client_disconnected: Callable) -> None:
    """调整停止序列参数"""
    from server import logger
    
    page_params_cache = context['page_params_cache']
    stops_to_set = stop_sequences if stop_sequences is not None else DEFAULT_STOP_SEQUENCES
    
    logger.info(f"[{req_id}] 停止序列 - 请求值: {stop_sequences}, 实际使用: {stops_to_set}")

    normalized_stops = set()
    if isinstance(stops_to_set, str):
        if stops_to_set.strip(): normalized_stops.add(stops_to_set.strip())
    elif isinstance(stops_to_set, list):
        normalized_stops = {s.strip() for s in stops_to_set if isinstance(s, str) and s.strip()}
    
    cached_stops = page_params_cache.get("stop_sequences")
    if cached_stops is not None and cached_stops == normalized_stops:
        logger.info(f"[{req_id}] 停止序列 ({normalized_stops}) 与缓存值 ({cached_stops}) 一致。跳过。")
        return

    stop_input_locator = page.locator(STOP_SEQUENCE_INPUT_SELECTOR)
    remove_chip_locator = page.locator(MAT_CHIP_REMOVE_BUTTON_SELECTOR)
    try:
        while await remove_chip_locator.count() > 0:
            await remove_chip_locator.first.click(timeout=2000)
            await asyncio.sleep(0.1)
        
        if normalized_stops:
            await expect_async(stop_input_locator).to_be_visible(timeout=5000)
            for seq in normalized_stops:
                await stop_input_locator.fill(seq, timeout=3000)
                await stop_input_locator.press("Enter", timeout=3000)
        
        page_params_cache["stop_sequences"] = normalized_stops
        logger.info(f"[{req_id}] ✅ 停止序列已更新为: {normalized_stops}")
    except Exception as e:
        logger.error(f"[{req_id}] ❌ 调整停止序列时出错: {e}")
        page_params_cache.pop("stop_sequences", None)
        await save_error_snapshot(f"stop_sequence_error_{req_id}")


async def _adjust_top_p_parameter_with_value(req_id: str, page: AsyncPage, top_p: Optional[float], 
                                                context: dict, check_client_disconnected: Callable) -> None:
    """调整Top P参数"""
    from server import logger
    
    top_p_to_set = top_p if top_p is not None else DEFAULT_TOP_P
    logger.info(f"[{req_id}] Top P - 请求值: {top_p}, 实际使用: {top_p_to_set}")
    
    clamped_top_p = max(0.0, min(1.0, top_p_to_set))
    if abs(clamped_top_p - top_p_to_set) > 1e-9:
        logger.warning(f"[{req_id}] Top P {top_p_to_set} 超出范围 [0, 1]，调整为 {clamped_top_p}")
    
    top_p_input_locator = page.locator(TOP_P_INPUT_SELECTOR)
    try:
        await expect_async(top_p_input_locator).to_be_visible(timeout=5000)
        await top_p_input_locator.fill(str(clamped_top_p), timeout=5000)
        logger.info(f"[{req_id}] ✅ Top P 已更新为: {clamped_top_p}")
    except Exception as e:
        logger.error(f"[{req_id}] ❌ 调整 Top P 时出错: {e}")
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
        check_client_disconnected("输入框可见后: ")
        
        await prompt_textarea_locator.evaluate(
            '(element, text) => { element.value = text; element.dispatchEvent(new Event("input", { bubbles: true })); }',
            prepared_prompt
        )
        await autosize_wrapper_locator.evaluate('(element, text) => { element.setAttribute("data-value", text); }', prepared_prompt)
        check_client_disconnected("JS填充后: ")
        
        await expect_async(submit_button_locator).to_be_enabled(timeout=40000)
        
        max_click_attempts = 3
        for attempt in range(max_click_attempts):
            try:
                check_client_disconnected(f"提交按钮点击前 (尝试 {attempt + 1}): ")
                await submit_button_locator.click(timeout=CLICK_TIMEOUT_MS)
                logger.info(f"[{req_id}] ✅ 提示已在尝试 {attempt + 1}/{max_click_attempts} 次后成功提交。")
                break # Success, exit loop
            except (PlaywrightAsyncError, asyncio.TimeoutError) as e:
                logger.warning(f"[{req_id}] 提交提示时点击失败 (尝试 {attempt + 1}/{max_click_attempts}): {type(e).__name__}")
                if attempt < max_click_attempts - 1:
                    await save_error_snapshot(f"submit_prompt_click_retry_attempt_{attempt+1}_{req_id}")
                    await asyncio.sleep(random.uniform(0.5, 1.5)) # Wait before next retry
                else:
                    logger.error(f"[{req_id}] ❌ 经过 {max_click_attempts} 次尝试后，提交提示最终失败。")
                    raise
        
    except (PlaywrightAsyncError, asyncio.TimeoutError, ClientDisconnectedError) as e_submit:
        if isinstance(e_submit, ClientDisconnectedError): raise
        logger.error(f"[{req_id}] ❌ 填充或提交提示时出错: {e_submit}", exc_info=True)
        await save_error_snapshot(f"submit_prompt_error_{req_id}")
        raise HTTPException(status_code=502, detail=f"[{req_id}] Failed to submit prompt to AI Studio: {e_submit}")
    except Exception as e_submit_unknown:
        logger.exception(f"[{req_id}] ❌ 填充或提交提示时意外错误")
        await save_error_snapshot(f"submit_prompt_unexpected_{req_id}")
        raise HTTPException(status_code=500, detail=f"[{req_id}] Unexpected error during prompt submission: {e_submit_unknown}")


async def _handle_response_processing(req_id: str, request: ChatCompletionRequest, page: AsyncPage, 
                                    context: dict, result_future: Future, 
                                    submit_button_locator: Locator, check_client_disconnected: Callable) -> Optional[Tuple[Event, Locator, Callable]]:
    """处理响应生成"""
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
    model_name_for_stream = current_ai_studio_model_id or MODEL_NAME
    
    if is_streaming:
        completion_event = Event()
        async def create_stream_generator_from_helper(event_to_set: Event) -> AsyncGenerator[str, None]:
            last_reason_pos, last_body_pos = 0, 0
            full_reasoning_content, full_body_content = "", ""
            chat_completion_id = f"{CHAT_COMPLETION_ID_PREFIX}{req_id}"
            created_timestamp = int(time.time())

            try:
                async for data in use_stream_response(req_id):
                    check_client_disconnected("辅助流循环中: ")
                    
                    if not isinstance(data, dict):
                        logger.warning(f"[{req_id}] 从辅助流收到非字典类型数据: {data}")
                        continue
                    
                    reason = data.get("reason", "")
                    body = data.get("body", "")
                    done = data.get("done", False)
                    functions = data.get("function", [])
                    
                    delta_payload = {}
                    
                    if len(reason) > last_reason_pos:
                        delta_reason = reason[last_reason_pos:]
                        full_reasoning_content += delta_reason
                        delta_payload["reasoning_content"] = delta_reason
                        last_reason_pos = len(reason)
                    
                    if len(body) > last_body_pos:
                        delta_body = body[last_body_pos:]
                        full_body_content += delta_body
                        delta_payload["content"] = delta_body
                        last_body_pos = len(body)

                    if delta_payload:
                        chunk = {
                            "id": chat_completion_id,
                            "object": "chat.completion.chunk", "created": created_timestamp, "model": model_name_for_stream,
                            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if done:
                        usage_stats = calculate_usage_stats([msg.model_dump() for msg in request.messages], full_body_content, full_reasoning_content)
                        logger.info(f"[{req_id}] 计算的Token使用情况: {usage_stats}")
                        
                        finish_reason = "tool_calls" if functions else "stop"
                        stop_chunk = generate_sse_stop_chunk(req_id, model_name_for_stream, finish_reason, usage_stats)
                        yield stop_chunk
                        break
            except Exception as e:
                logger.error(f"[{req_id}] 辅助流生成器出错: {e}", exc_info=True)
                yield generate_sse_error_chunk(f"Error in auxiliary stream: {e}", req_id)
            finally:
                if not event_to_set.is_set(): event_to_set.set()

        stream_gen_func = create_stream_generator_from_helper(completion_event)
        if not result_future.done():
            result_future.set_result(StreamingResponse(stream_gen_func, media_type="text/event-stream"))
        return completion_event, submit_button_locator, check_client_disconnected
    else: # Non-streaming with auxiliary stream
        body_content, reasoning_content, functions = "", None, None
        async for data in use_stream_response(req_id):
            check_client_disconnected("非流式辅助流循环中: ")
            if isinstance(data, dict) and data.get("done"):
                body_content = data.get("body", "")
                reasoning_content = data.get("reason")
                functions = data.get("function")
                break
        
        if body_content is None: # Check for None explicitly
            raise HTTPException(status_code=502, detail=f"[{req_id}] Auxiliary stream finished but provided no content.")

        # Wrap reasoning content in <think> tags
        final_content = body_content
        if reasoning_content:
            final_content = f"<think>{reasoning_content}</think>\n{body_content}"

        usage_stats = calculate_usage_stats([msg.model_dump() for msg in request.messages], body_content, reasoning_content)
        message_payload = {"role": "assistant", "content": final_content}
        finish_reason = "stop"
        
        if functions:
            message_payload["tool_calls"] = functions
            message_payload["content"] = None # Per OpenAI spec for tool calls
            finish_reason = "tool_calls"

        response_payload = {
            "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}", "object": "chat.completion", "created": int(time.time()),
            "model": current_ai_studio_model_id or MODEL_NAME,
            "choices": [{"index": 0, "message": message_payload, "finish_reason": finish_reason}],
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
    
    logger.info(f"[{req_id}] (Playwright) 等待响应生成完成...")
    input_field_locator = page.locator(INPUT_SELECTOR)
    edit_button_locator = page.locator(EDIT_MESSAGE_BUTTON_SELECTOR)
    
    completion_detected = await _wait_for_response_completion(
        page, input_field_locator, submit_button_locator, edit_button_locator, 
        req_id, check_client_disconnected, req_id
    )
    check_client_disconnected("响应完成后: ")
    
    if not completion_detected:
        page_toast_error = await detect_and_extract_page_error(page, req_id)
        if page_toast_error:
            raise HTTPException(status_code=502, detail=f"[{req_id}] AI Studio Page Error: {page_toast_error}")
        raise HTTPException(status_code=504, detail=f"[{req_id}] AI Studio response generation timed out.")

    final_content = await _get_final_response_content(page, req_id, check_client_disconnected)
    if final_content is None:
        raise HTTPException(status_code=500, detail=f"[{req_id}] Failed to extract final response content from AI Studio.")

    usage_stats = calculate_usage_stats([msg.model_dump() for msg in request.messages], final_content, "")
    logger.info(f"[{req_id}] (Playwright) 计算的Token使用情况: {usage_stats}")

    if is_streaming:
        completion_event = Event()
        async def create_pseudo_stream(event_to_set: Event, content: str) -> AsyncGenerator[str, None]:
            try:
                chunk_size = 5
                for i in range(0, len(content), chunk_size):
                    check_client_disconnected("伪流式循环中: ")
                    chunk = content[i:i+chunk_size]
                    yield generate_sse_chunk(chunk, req_id, current_ai_studio_model_id or MODEL_NAME)
                    await asyncio.sleep(PSEUDO_STREAM_DELAY)
                yield generate_sse_stop_chunk(req_id, current_ai_studio_model_id or MODEL_NAME, "stop", usage_stats)
            finally:
                if not event_to_set.is_set(): event_to_set.set()
        
        stream_gen = create_pseudo_stream(completion_event, final_content)
        if not result_future.done():
            result_future.set_result(StreamingResponse(stream_gen, media_type="text/event-stream"))
        return completion_event, submit_button_locator, check_client_disconnected
    else:
        response_payload = {
            "id": f"{CHAT_COMPLETION_ID_PREFIX}{req_id}", "object": "chat.completion", "created": int(time.time()),
            "model": current_ai_studio_model_id or MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
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
        try: await disconnect_check_task
        except asyncio.CancelledError: pass
    
    logger.info(f"[{req_id}] 处理完成。")
    
    if is_streaming and completion_event and not completion_event.is_set() and (result_future.done() and result_future.exception()):
         logger.warning(f"[{req_id}] 流式请求异常，确保完成事件已设置。")
         completion_event.set()


async def _process_request_refactored(
    req_id: str,
    request: ChatCompletionRequest,
    http_request: Request,
    result_future: Future
) -> Optional[Tuple[Event, Locator, Callable[[str], bool]]]:
    """核心请求处理函数 - 重构版本"""
    
    from server import logger
    context = await _initialize_request_context(req_id, request)
    context = await _analyze_model_requirements(req_id, context, request)
    
    client_disconnected_event, disconnect_check_task, check_client_disconnected = await _setup_disconnect_monitoring(
        req_id, http_request, result_future
    )
    
    page = context['page']
    submit_button_locator = page.locator(SUBMIT_BUTTON_SELECTOR) if page else None
    completion_event = None
    
    try:
        await _validate_page_status(req_id, context, check_client_disconnected)
        context = await _handle_model_switching(req_id, context, check_client_disconnected)
        await _handle_parameter_cache(req_id, context)
        
        system_prompt_content, prepared_prompt = await _prepare_and_validate_request(req_id, request, check_client_disconnected)
        
        # 页面交互逻辑
        stream_port = os.environ.get('STREAM_PORT')
        use_stream = stream_port != '0'

        # --- 统一的增量逻辑判断 ---
        is_incremental_page_update = False
        prompt_to_send_to_page = ""
        
        async with context['page_sync_cache_lock']:
            full_sync_reason = ""
            if context['model_actually_switched']:
                full_sync_reason = "模型已切换"
            elif not is_incremental_messages(context['last_api_messages_synced_to_page'], request.messages, req_id):
                full_sync_reason = "消息非增量"

            if not full_sync_reason:
                is_incremental_page_update = True
                num_old_messages = len(context.get('last_api_messages_synced_to_page', []))
                newly_added_messages = request.messages[num_old_messages:]
                
                if newly_added_messages and newly_added_messages[-1].role == 'user':
                    last_user_message_content = newly_added_messages[-1].content
                    if isinstance(last_user_message_content, str):
                        prompt_to_send_to_page = last_user_message_content
                    elif isinstance(last_user_message_content, list):
                        text_parts = [item.text for item in last_user_message_content if item.type == 'text' and item.text]
                        prompt_to_send_to_page = "\n".join(text_parts)
                    else:
                        prompt_to_send_to_page = str(last_user_message_content or "")
                
                logger.info(f"[{req_id}] (增量) 页面将进行增量更新。")
            else:
                is_incremental_page_update = False
                prompt_to_send_to_page = prepared_prompt
                logger.info(f"[{req_id}] (完全同步) 页面将执行完全同步，原因: {full_sync_reason}。")

        # --- 执行页面操作 ---
        try:
            if not is_incremental_page_update:
                await _clear_chat_history(req_id, page, check_client_disconnected)
                check_client_disconnected("清空聊天后: ")
            
            await set_system_prompt_in_page(page, system_prompt_content or "", req_id, check_client_disconnected)
            check_client_disconnected("设置系统提示后: ")
            
            await _adjust_request_parameters(req_id, page, request, context, check_client_disconnected)
            check_client_disconnected("调整参数后: ")
            
            await _submit_prompt(req_id, page, prompt_to_send_to_page, check_client_disconnected)
            check_client_disconnected("提交提示后: ")
            
            # 成功后更新缓存
            async with context['page_sync_cache_lock']:
                import server
                server.last_api_messages_synced_to_page = [msg.model_copy(deep=True) for msg in request.messages]
                logger.info(f"[{req_id}] last_api_messages_synced_to_page 已更新。")
        except Exception as page_sync_err:
            async with context['page_sync_cache_lock']:
                import server
                server.last_api_messages_synced_to_page = None # 同步失败时清除缓存
            logger.error(f"[{req_id}] 页面同步失败: {page_sync_err}", exc_info=True)
            await save_error_snapshot(f"page_sync_error_{req_id}")
            # 如果使用辅助流，即使页面同步失败也继续；否则，抛出异常
            if not use_stream:
                raise
        
        # 处理响应
        response_result = await _handle_response_processing(
            req_id, request, page, context, result_future, submit_button_locator, check_client_disconnected
        )
        
        if response_result:
            completion_event, submit_button_locator, check_client_disconnected = response_result
        
        return completion_event, submit_button_locator, check_client_disconnected
        
    except (ClientDisconnectedError, HTTPException, PlaywrightAsyncError, asyncio.TimeoutError, asyncio.CancelledError) as e:
        if not result_future.done():
            if isinstance(e, ClientDisconnectedError):
                result_future.set_exception(HTTPException(status_code=499, detail=f"[{req_id}] Client disconnected."))
            elif isinstance(e, HTTPException):
                result_future.set_exception(e)
            elif isinstance(e, (PlaywrightAsyncError, asyncio.TimeoutError)):
                await save_error_snapshot(f"process_error_{req_id}")
                result_future.set_exception(HTTPException(status_code=502, detail=f"[{req_id}] Playwright/Timeout error: {e}"))
            elif isinstance(e, asyncio.CancelledError):
                result_future.cancel()
        context['logger'].info(f"[{req_id}] 捕获到已知异常: {type(e).__name__}")
    except Exception as e:
        context['logger'].exception(f"[{req_id}] 捕获到意外错误")
        await save_error_snapshot(f"process_unexpected_error_{req_id}")
        if not result_future.done(): 
            result_future.set_exception(HTTPException(status_code=500, detail=f"[{req_id}] Unexpected server error: {e}"))
    finally:
        await _cleanup_request_resources(req_id, disconnect_check_task, completion_event, result_future, request.stream)
        # The return value is primarily for the worker to wait on the completion event.
        # In case of exception, it might be None, but the future is already set.
        return completion_event, submit_button_locator, check_client_disconnected