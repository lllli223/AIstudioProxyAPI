"""
API工具函数模块
包含SSE生成、流处理、token统计和请求验证等工具函数
"""

import asyncio
import json
import time
import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
from asyncio import Queue

from models import Message, MessageContentItem, ToolCall



# --- SSE生成函数 ---
def generate_sse_chunk(delta: str, req_id: str, model: str) -> str:
    """生成SSE数据块"""
    chunk_data = {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"


def generate_sse_stop_chunk(req_id: str, model: str, reason: str = "stop", usage: dict = None) -> str:
    """生成SSE停止块"""
    stop_chunk_data = {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
    }
    
    # 添加usage信息（如果提供）
    if usage:
        stop_chunk_data["usage"] = usage
    
    return f"data: {json.dumps(stop_chunk_data)}\n\ndata: [DONE]\n\n"


def generate_sse_error_chunk(message: str, req_id: str, error_type: str = "server_error") -> str:
    """生成SSE错误块"""
    error_chunk = {"error": {"message": message, "type": error_type, "param": None, "code": req_id}}
    return f"data: {json.dumps(error_chunk)}\n\n"


# --- 流处理工具函数 ---
async def use_stream_response(req_id: str) -> AsyncGenerator[Any, None]:
    """使用流响应（从服务器的全局队列获取数据）"""
    from server import STREAM_QUEUE, clear_stream_queue, logger
    import queue
    
    if STREAM_QUEUE is None:
        logger.warning(f"[{req_id}] STREAM_QUEUE is None, 无法使用流响应")
        return
    
    logger.info(f"[{req_id}] 开始使用流响应")
    
    empty_count = 0
    max_empty_retries = 300  # 30秒超时
    data_received = False
    
    try:
        while True:
            try:
                # 从队列中获取数据
                data = STREAM_QUEUE.get_nowait()
                if data is None:  # 结束标志
                    logger.info(f"[{req_id}] 接收到流结束标志")
                    break
                
                # 重置空计数器
                empty_count = 0
                data_received = True
                logger.debug(f"[{req_id}] 接收到流数据: {type(data)} - {str(data)[:200]}...")
                
                # 检查是否是JSON字符串形式的结束标志
                if isinstance(data, str):
                    try:
                        parsed_data = json.loads(data)
                        if parsed_data.get("done") is True:
                            logger.info(f"[{req_id}] 接收到JSON格式的完成标志")
                            yield parsed_data
                            break
                        else:
                            yield parsed_data
                    except json.JSONDecodeError:
                        # 如果不是JSON，直接返回字符串
                        logger.debug(f"[{req_id}] 返回非JSON字符串数据")
                        yield data
                else:
                    # 直接返回数据
                    yield data
                    
                    # 检查字典类型的结束标志
                    if isinstance(data, dict) and data.get("done") is True:
                        logger.info(f"[{req_id}] 接收到字典格式的完成标志")
                        break
                
            except (queue.Empty, asyncio.QueueEmpty):
                empty_count += 1
                if empty_count % 50 == 0:  # 每5秒记录一次等待状态
                    logger.info(f"[{req_id}] 等待流数据... ({empty_count}/{max_empty_retries})")
                
                if empty_count >= max_empty_retries:
                    if not data_received:
                        logger.error(f"[{req_id}] 流响应队列空读取次数达到上限且未收到任何数据，可能是辅助流未启动或出错")
                    else:
                        logger.warning(f"[{req_id}] 流响应队列空读取次数达到上限 ({max_empty_retries})，结束读取")
                    
                    # 返回超时完成信号，而不是简单退出
                    yield {"done": True, "reason": "internal_timeout", "body": "", "function": []}
                    return
                    
                await asyncio.sleep(0.1)  # 100ms等待
                continue
                
    except Exception as e:
        logger.error(f"[{req_id}] 使用流响应时出错: {e}")
        raise
    finally:
        logger.info(f"[{req_id}] 流响应使用完成，数据接收状态: {data_received}")


async def clear_stream_queue():
    """清空流队列"""
    from server import STREAM_QUEUE, logger
    
    if STREAM_QUEUE is None:
        return
    
    try:
        # 清空队列中剩余的数据
        while True:
            try:
                STREAM_QUEUE.get_nowait()
            except:
                break
        logger.debug("流队列已清空")
    except Exception as e:
        logger.error(f"清空流队列时出错: {e}")


# --- Helper response generator ---
async def use_helper_get_response(helper_endpoint: str, helper_sapisid: str) -> AsyncGenerator[str, None]:
    """使用Helper服务获取响应的生成器"""
    from server import logger
    import aiohttp
    
    logger.info(f"正在尝试使用Helper端点: {helper_endpoint}")
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'Cookie': f'SAPISID={helper_sapisid}' if helper_sapisid else ''
            }
            
            async with session.get(helper_endpoint, headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8', errors='ignore')
                else:
                    logger.error(f"Helper端点返回错误状态: {response.status}")
                    
    except Exception as e:
        logger.error(f"使用Helper端点时出错: {e}")


def is_incremental_messages(old_messages: Optional[List['Message']], new_messages: List['Message'], req_id: str) -> bool:
    """
    判断 new_messages 是否是 old_messages 的增量。
    增量定义为：old_messages 非空，new_messages 以 old_messages 开头，且 new_messages 更长。
    """
    from server import logger, TRACE_LOGS_ENABLED
    
    if old_messages is None:
        logger.debug(f"[{req_id}] (is_incremental) 否: 无旧消息。")
        return False
    if len(new_messages) <= len(old_messages):
        logger.debug(f"[{req_id}] (is_incremental) 否: 新消息列表不够长 (新: {len(new_messages)}, 旧: {len(old_messages)})。")
        return False

    # 逐条比较旧消息部分是否完全一致
    for i in range(len(old_messages)):
        # Pydantic 模型可以直接比较，会自动比较所有字段
        if old_messages[i] != new_messages[i]:
            if TRACE_LOGS_ENABLED: # 仅在 TRACE 级别记录详细差异
                logger.trace(f"[{req_id}] (is_incremental) 否: 消息在索引 {i} 处不匹配。")
                try:
                    logger.trace(f"  Old[{i}]: {old_messages[i].model_dump_json(indent=2)}")
                    logger.trace(f"  New[{i}]: {new_messages[i].model_dump_json(indent=2)}")
                except Exception: # 防御性编程，避免日志本身出错
                    logger.trace(f"  (无法序列化消息进行详细比较)")
            else:
                logger.debug(f"[{req_id}] (is_incremental) 否: 消息在索引 {i} 处不匹配。")
            return False
            
    logger.debug(f"[{req_id}] (is_incremental) 是: 新消息列表是旧消息列表的增量扩展。")
    return True


# --- 请求验证函数 ---
def validate_chat_request(messages: List[Message], req_id: str) -> Dict[str, Optional[str]]:
    """验证聊天请求"""
    from server import logger
    
    if not messages:
        raise ValueError(f"[{req_id}] 无效请求: 'messages' 数组缺失或为空。")
    
    if not any(msg.role != 'system' for msg in messages):
        raise ValueError(f"[{req_id}] 无效请求: 所有消息都是系统消息。至少需要一条用户或助手消息。")
    
    # 返回验证结果
    return {
        "error": None,
        "warning": None
    }


# --- 提示准备函数 (已更新) ---
def prepare_combined_prompt(messages: List[Message], req_id: str) -> Tuple[Optional[str], str]:
    """
    准备组合提示。
    该函数会从消息列表中提取第一个系统消息作为独立的系统提示，
    并将其余的用户、助手和工具消息组合成一个单一的对话历史字符串。
    
    Args:
        messages: 消息对象列表。
        req_id: 请求ID，用于日志记录。

    Returns:
        一个元组 (system_prompt, combined_prompt):
        - system_prompt (Optional[str]): 提取的系统提示内容。如果找到但内容为空，则返回空字符串""。
                                         如果没有找到系统消息，则返回 None。
        - combined_prompt (str): 组合后的用户/助手/工具对话历史字符串。
    """
    from server import logger
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


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量
    使用简单的字符计数方法：
    - 英文：大约4个字符 = 1个token
    - 中文：大约1.5个字符 = 1个token  
    - 混合文本：采用加权平均
    """
    if not text:
        return 0
    
    # 统计中文字符数量（包括中文标点）
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f' or '\uff00' <= char <= '\uffef')
    
    # 统计非中文字符数量
    non_chinese_chars = len(text) - chinese_chars
    
    # 计算token估算
    chinese_tokens = chinese_chars / 1.5  # 中文大约1.5字符/token
    english_tokens = non_chinese_chars / 4.0  # 英文大约4字符/token
    
    return max(1, int(chinese_tokens + english_tokens))


def calculate_usage_stats(messages: List[dict], response_content: str, reasoning_content: str = None) -> dict:
    """
    计算token使用统计
    
    Args:
        messages: 请求中的消息列表
        response_content: 响应内容
        reasoning_content: 推理内容（可选）
    
    Returns:
        包含token使用统计的字典
    """
    # 计算输入token（prompt tokens）
    prompt_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prompt_text += f"{role}: {content}\n"
    
    prompt_tokens = estimate_tokens(prompt_text)
    
    # 计算输出token（completion tokens）
    completion_text = response_content or ""
    if reasoning_content:
        completion_text += reasoning_content
    
    completion_tokens = estimate_tokens(completion_text)
    
    # 总token数
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    } 


def generate_sse_stop_chunk_with_usage(req_id: str, model: str, usage_stats: dict, reason: str = "stop") -> str:
    """生成带usage统计的SSE停止块"""
    return generate_sse_stop_chunk(req_id, model, reason, usage_stats)