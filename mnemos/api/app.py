"""
Mnemos FastAPI 服务：供 root 安卓手表等客户端调用。

接口：接收 message + member_id，返回 reply 及是否有行为偏离的 bool 标志。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mnemos.core.coordinator import MnemosCoordinator, CoordinatorInput
from mnemos.core.linguist import JarvisLinguist
from mnemos.workers import FactRegistrar, InsightPhilosopher, Cartographer

# 与 Philosopher 约定一致
TAG_BEHAVIOR_DEVIATION = "潜在的行为偏离"

app = FastAPI(title="Mnemos API", description="家庭记忆与反思对话服务")

# 全局协调器（注入 Linguist / Registrar / Philosopher / Cartographer）
coordinator = MnemosCoordinator(
    linguist=JarvisLinguist(),
    registrar=FactRegistrar(),
    philosopher=InsightPhilosopher(),
    cartographer=Cartographer(),
    memos_client=None,
)


class ChatRequest(BaseModel):
    message: str
    member_id: str


class ChatResponse(BaseModel):
    reply: str
    has_deviation: bool


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """接收 message 与 member_id，返回回复内容及是否检测到行为偏离。"""
    try:
        input_data = CoordinatorInput(
            message=request.message.strip(),
            member_id=request.member_id.strip(),
        )
        output = await coordinator.process(input_data)
        insights = output.insights or []
        has_deviation = any(
            isinstance(i, dict) and i.get("tag") == TAG_BEHAVIOR_DEVIATION
            for i in insights
        )
        return ChatResponse(
            reply=output.response,
            has_deviation=has_deviation,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查，便于网关或 adb 探测。"""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 启动与测试
# ---------------------------------------------------------------------------
# 启动（在项目根目录，uv 环境）：
#   uv run uvicorn mnemos.api.app:app --host 0.0.0.0 --port 8000
#
# 安卓 adb shell 下用 curl 测试（将 <HOST> 换成本机 IP，手表与电脑同网段时用电脑 IP）：
#   curl -X POST -d '{"message":"今天没跑步","member_id":"user_001"}' \
#        -H "Content-Type: application/json" \
#        http://<HOST>:8000/chat
# 若 shell 中单引号有问题，可用转义双引号：
#   curl -X POST -d "{\"message\":\"今天没跑步\",\"member_id\":\"user_001\"}" \
#        -H "Content-Type: application/json" \
#        http://<HOST>:8000/chat
