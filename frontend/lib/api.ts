// 前端 HTTP 边界：统一基础地址、JSON 请求、文件上传、SSE 解帧和错误信息提取。
// 泛型 T 只是编译期类型声明，不会在运行时验证后端 JSON；权限与业务校验仍由服务端负责。
// 普通 API 返回完整 JSON，流式 API 则通过 onEvent 连续交付 delta/tool/final 等事件。
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// no-store 避免研发状态被 fetch 缓存；HTTP 非成功时先抛可读错误，再尝试读取成功 JSON。
export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// FormData 上传不手写 Content-Type，浏览器会自动补带 boundary 的 multipart 请求头。
export async function apiUpload<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 流式 JSON POST 的传输层，负责解码和分帧，不理解 Issue/RAG 等业务字段。
// 一个网络块可能包含多条 SSE，也可能只有半条；buffer 跨 read 调用保留未完整事件。
// TextDecoder(stream:true) 保留跨块 UTF-8 字节，避免中文字符刚好被网络边界拆开而乱码。
// 只按服务端当前 event/data 加空行格式处理；它不是支持所有 SSE 扩展语法的通用库。
// onEvent 回调由页面解释 delta/tool/final，传输结束本身不自动证明已收到了业务 final。
export async function apiPostStream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  options: { signal?: AbortSignal } = {}
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal
  });
  if (!response.ok) await throwApiError(response);
  if (!response.body) throw new Error("Streaming response body is not available.");

  const reader = response.body.getReader();  // 当前 HTTP 响应体的字节流读取器，不是一次性完整 JSON
  // 网络块可能截在一个中文字符中间；用流式解码器保留不完整 UTF-8 字节，避免乱码。
  const decoder = new TextDecoder();
  let buffer = "";  // 保存尚未遇到完整事件分隔符的尾部文本

  // 不断读取网络块直到 done；一个块可能含半条或多条事件，因此外层读块、内层再逐条处理完整 SSE。
  while (true) {
    const { done, value } = await reader.read();
    // done 表示网络响应体结束，必须退出读取循环；是否收到业务 final 仍由调用页面结合事件判断。
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // 网络块不等于 SSE 事件：一次可能来多条，也可能只来半条。按空行拆帧并把尾部留到下一轮。
    const events = buffer.split("\n\n");
    // split 最后一段可能是半个事件，留到下一块再拼；完整事件才交给页面，避免 JSON 读一半。
    buffer = events.pop() ?? "";
    // 只遍历已经由空行分隔完成的事件，未完成尾部仍在 buffer，不能对半个 JSON 提前解析。
    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      const event = lines.find((line) => line.startsWith("event: "))?.slice(7) ?? "message";
      const dataText = lines
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      // 多条 data 行先合并再解析一次 JSON；解析失败会抛给调用页面，不以空对象冒充正常事件。
      if (dataText) onEvent(event, JSON.parse(dataText));
    }
  }
}

export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 先按 Content-Type 尝试解析 JSON 错误，再提取 detail/message；无法解析时退回 HTTP 状态。
// FastAPI 的参数校验 detail 可能是数组，要收集各项 msg，而非显示 [object Object]。
// 抛 Error 让页面 catch 控制交互状态，统一接口行为，不在底层直接弹窗或改页面业务数据。
async function throwApiError(response: Response): Promise<never> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    let message: string | null = null;
    try {
      const payload = await response.json();
      message = readApiErrorMessage(payload);
    } catch {
      message = null;
    }
    throw new Error(message || fallback);
  }
  const text = (await response.text()).trim();
  throw new Error(text || fallback);
}

function readApiErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
        return typeof item === "string" ? item : null;
      })
      .filter(Boolean);
    return messages.length ? messages.join("; ") : null;
  }
  const message = (payload as { message?: unknown }).message;
  return typeof message === "string" ? message : null;
}
