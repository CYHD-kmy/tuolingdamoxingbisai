# 智投未来 — 云服务器部署指南

## 一、服务器准备

### 1.1 选购建议 (中国云厂商)

最低配置即可运行：

| 项目 | 建议 |
|------|------|
| CPU | 1 核 (2 核更佳) |
| 内存 | 2 GB (推荐 4 GB) |
| 系统 | Ubuntu 22.04 LTS |
| 带宽 | 1 Mbps 起步 |

推荐平台:

- **阿里云 ECS** — [ecs.console.aliyun.com](https://ecs.console.aliyun.com)
- **腾讯云 CVM** — [console.cloud.tencent.com/cvm](https://console.cloud.tencent.com/cvm)
- **华为云 ECS** — [console.huaweicloud.com/ecm](https://console.huaweicloud.com/ecm)

### 1.2 安全组 / 防火墙

在云控制台开放以下端口:

| 端口 | 协议 | 用途 |
|------|------|------|
| 22 | TCP | SSH 登录 |
| 80 | TCP | HTTP |
| 443 | TCP | HTTPS |

---

## 二、部署步骤 (Docker)

### 2.1 登录服务器 & 安装 Docker

```bash
ssh root@你的服务器IP

# 安装 Docker (Ubuntu)
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER

# 安装 Docker Compose
sudo apt install docker-compose-plugin -y
```

### 2.2 上传项目

```bash
# 在本地机器上执行
cd "智投未来项目设计"
scp -r . root@你的服务器IP:/opt/zhitou-future/
```

### 2.3 配置环境变量

```bash
# 在服务器上
cd /opt/zhitou-future
cp .env.example .env

# 编辑 .env，填入你的 API Key
vim .env
```

`.env` 最小配置:

```env
LLM_API_KEY=sk-你的DeepSeek密钥
```

### 2.4 启动服务

```bash
docker compose up -d
```

验证:

```bash
curl http://localhost:8000/api/health
# → {"status":"ok"}

docker compose logs -f
```

---

## 三、配置 Nginx + HTTPS

### 3.1 安装 Nginx 和 Certbot

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
```

### 3.2 配置 Nginx

```bash
# 复制配置文件，替换域名
sudo cp /opt/zhitou-future/nginx.conf /etc/nginx/sites-available/zhitou
sudo sed -i 's/your-domain.com/你的真实域名/g' /etc/nginx/sites-available/zhitou
sudo ln -s /etc/nginx/sites-available/zhitou /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 3.3 申请 SSL 证书

```bash
sudo certbot --nginx -d 你的真实域名
```

---

## 四、日常管理

```bash
# 查看日志
docker compose logs -f

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 更新代码后重建
git pull
docker compose up -d --build
```

---

## 五、无域名方案 (仅 IP 访问)

如果还没有域名，暂时用 IP 访问:

```bash
# 直接用 Docker 暴露端口
docker run -d --name zhitou -p 80:8000 \
  -e LLM_API_KEY=sk-xxx \
  --restart always \
  zhitou-future

# 然后访问 http://你的服务器IP
```

---

## 六、监控 & 更新

```bash
# 设置每天凌晨自动重启 (crontab -e)
0 3 * * * cd /opt/zhitou-future && docker compose restart

# 监控健康检查
*/5 * * * * curl -f http://localhost:8000/api/health || docker compose restart
```
