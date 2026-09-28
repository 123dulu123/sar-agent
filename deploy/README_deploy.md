# 部署指南：把 SAR-Agent 变成公网可访问的在线 Demo

> 两种路径：A. HuggingFace Space（免费、最快，推荐）；B. 自有服务器 + Docker。
> 共同前提：**代码里没有任何密钥**——API Key 通过环境变量/Secret 注入（`configs/.env` 永不入库）。

## 路径 A：HuggingFace Space（约 15 分钟）

1. 注册/登录 <https://huggingface.co>，右上角 New → Space；
2. SDK 选择 **Gradio**（Space 会识别 `app.py` 并自动运行），硬件选免费 CPU；
3. 上传仓库文件（或 git push 到 Space 的仓库），注意：
   - 不需要上传 `outputs/`、`store/`、`.venv/`（.gitignore 已排除）；
   - `weights/best.pt`、`data/` 必须包含（在线 Demo 的检测能力来源）；
4. 在 Space → Settings → **Variables and secrets** 添加：
   - `ZHIPUAI_API_KEY` = 你的 Key（Secret 类型，不会泄露到前端）；
5. Space 构建完成后获得 `https://<你的用户名>-sar-agent.hf.space`，这就是可直接发给面试官的链接。

**注意**：
- Space 里 `configs/settings.yaml` 的 `server.host` 需为 `0.0.0.0`（可在仓库中默认改好或用环境变量覆盖）；
- 免费额度是 2 vCPU，推理约 3~5s/图，足够演示；
- GLM API 调用走智谱云，与 Space 无关，不需要额外端口。

## 路径 B：自有服务器 + Docker

```bash
docker build -t sar-agent .
docker run -d -p 7860:7860 -e ZHIPUAI_API_KEY=你的Key --name sar-agent sar-agent
# 访问 http://<服务器IP>:7860
```

如需 HTTPS 与域名，前置一层 Nginx/Caddy 反代即可。`docker-compose.yml` 模板：

```yaml
services:
  sar-agent:
    build: .
    ports: ["7860:7860"]
    environment:
      - ZHIPUAI_API_KEY=${ZHIPUAI_API_KEY}
    restart: unless-stopped
```

## 面试加分点（部署相关可讲的内容）

- 为什么密钥走环境变量而不是配置文件：12-Factor App 的配置分离原则，同一镜像跑所有环境；
- Gradio Space 的"约定优于配置"：SDK 检测 `app.py` 的 `demo.launch()` 自动托管；
- 演示权重进仓库（6MB）换取"克隆即运行"，训练数据与产物走 .gitignore——仓库大小与可复现性的平衡。
