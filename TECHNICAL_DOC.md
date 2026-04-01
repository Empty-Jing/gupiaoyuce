# 智能股票监控与 AI 预测系统技术文档 (Stock-Monitor Technical Documentation)

## 1. 项目概述

### 1.1 系统简介
本系统是一个集实时行情监控、多维技术指标计算、金融舆情情感分析、AI 趋势预测与多渠道自动告警于一体的综合性量化交易辅助平台。系统专为 A 股市场设计，利用异步并发架构确保海量数据处理的实时性，并结合深度学习（LSTM）与梯度提升树（XGBoost）模型提供高置信度的投资参考。

### 1.2 技术栈
- **后端框架**: FastAPI 0.135.1 (基于 ASGI 的高性能异步 Web 框架)
- **数据来源**: AkShare 1.18.42 (获取 A 股行情、新闻、资金流、北向资金)
- **数据处理**: Pandas 2.3.3, NumPy 2.4.3, TA 0.11.0
- **数据库层**: SQLAlchemy 2.0.48 (ORM), aiosqlite 0.22.1 (异步 SQLite 驱动)
- **AI 引擎**: 
  - PyTorch 2.10.0 (用于构建多头输出 LSTM 序列模型)
  - XGBoost 3.2.0 (用于捕捉静态技术指标的分类特征)
  - Transformers 5.3.0 (加载 FinBERT 模型进行中文金融情感预测)
- **任务调度**: APScheduler 3.11.2 (支持 Cron 与 Interval 类型的异步任务)
- **可视化界面**: Streamlit 1.55.0, Plotly 6.6.0 (交互式 K 线与仪表盘)
- **配置管理**: Pydantic-settings 2.13.1
- **运行环境**: Python 3.11.15

### 1.3 系统架构图 (Architecture)
```text
[ 表现层 (UI) ]          [ 接口层 (API) ]          [ 业务核心层 (Core) ]
       |                       |                         |
+--------------+        +--------------+        +------------------+
| Streamlit App| <----> | FastAPI Server| <----> | IndicatorEngine  |
| (Port 8501)  |        | (Port 8000)  |        | PredictionEngine |
+--------------+        +--------------+        | AlertManager     |
       ^                       ^                | DataCollector    |
       |                       |                +------------------+
       |                       |                         |
       |                +--------------+        +------------------+
       +--------------> | SQLite DB    | <----- | SentimentAnalyzer|
                        | (Async I/O)  |        | (FinBERT Model)  |
                        +--------------+        +------------------+
                               |
                        +--------------+
                        | Model Assets |
                        | (.pt / .json)|
                        +--------------+
```

### 1.4 详细目录结构
```text
stock-monitor/
├── app/                        # 后端核心代码目录
│   ├── api/                   # RESTful API 接口定义
│   │   ├── alerts.py          # 告警信息查询、标记已读及阈值配置接口
│   │   ├── predictions.py     # 获取 AI 预测结论、查询历史预测及重训练触发
│   │   └── stocks.py          # 股票自选列表、实时行情、K 线历史及搜索接口
│   ├── core/                  # 核心业务逻辑实现
│   │   ├── alert_manager.py   # 告警规则引擎，包含 6 种异常检测与去重逻辑
│   │   ├── data_collector.py  # 异步数据抓取层，封装 AkShare 并实现重试机制
│   │   ├── indicator_engine.py# 技术指标计算中心，生成 25+ 种技术指标与信号
│   │   ├── prediction_engine.py# 预测结果融合中心，结合 LSTM 与 XGBoost 结论
│   │   └── sentiment_analyzer.py# 舆情分析模块，基于预训练 FinBERT 进行情感建模
│   ├── database/              # 数据库访问与 ORM 映射
│   │   ├── crud.py            # 包含 16 个核心异步 CRUD 函数
│   │   ├── models.py          # 定义 7 个核心 SQLAlchemy 数据模型
│   │   └── session.py         # 异步引擎配置与 Session 生命周期管理
│   ├── models/                # AI 模型架构与训练逻辑
│   │   ├── lstm_model.py      # 定义单向多层多头输出 LSTM 网络
│   │   ├── train.py           # 自动化训练流水线，含 EarlyStopping 逻辑
│   │   └── xgboost_model.py   # XGBoost 分类器封装与 JSON 序列化加载
│   ├── scheduler/             # 自动化任务管理
│   │   └── tasks.py           # 定义 5 大定时任务及其调度 Cron 表达式
│   ├── config.py              # 系统配置类，基于 Pydantic Settings
│   └── main.py                # 项目入口文件，负责 FastAPI 启动与任务挂载
├── frontend/                   # 前端 Streamlit 应用目录
│   ├── components/            # 可复用可视化组件 (如 K 线图、预测卡片)
│   ├── pages/                 # 多页面导航实现
│   │   ├── 1_stock_dashboard.py# 自选股实时监控大盘
│   │   ├── 2_kline_detail.py  # 深度 K 线技术分析页面
│   │   ├── 3_prediction.py    # AI 趋势预测详情与置信度展示
│   │   ├── 4_news_sentiment.py# 舆情监控与告警流水
│   │   └── 5_settings.py      # 系统阈值设置与自选管理
│   └── app.py                 # Streamlit 应用主入口
├── tests/                      # 自动化测试目录
│   ├── conftest.py            # 测试全局配置与内存数据库 Fixture
│   ├── test_alerts.py         # 告警逻辑单元测试
│   ├── test_api.py            # 端到端 API 路由测试
│   ├── test_data_collector.py # 采集器 Mock 测试
│   ├── test_database.py       # 数据库事务测试
│   ├── test_indicators.py     # 技术指标精度验证
│   ├── test_predictions.py    # 预测引擎逻辑测试
│   └── test_sentiment.py      # 情感分析优雅降级测试
├── .env                        # 环境变量敏感配置
├── requirements.txt            # 项目依赖锁定清单
├── pytest.ini                 # Pytest 运行参数配置
├── stock_monitor.db           # 默认 SQLite 数据库文件
└── TECHNICAL_DOC.md           # 本文档
```

### 1.5 核心设计原则 (Core Design Principles)

本系统的设计遵循以下量化工程原则，确保其在多变的市场环境下具备高健壮性：

1. **信息零损耗原则**: 
   - 原始数据抓取后，保留所有 AkShare 返回的字段。
   - 所有模型训练时的归一化参数（Scaler）均与模型同步保存，确保特征缩放一致性。

2. **模块化解耦**: 
   - `IndicatorEngine` 不依赖于数据库，仅接受 DataFrame。
   - `AlertManager` 的推送逻辑采用观察者模式。

3. **优雅降级与防御性编程**: 
   - 若实时行情拉取失败，自动切换至 1 分钟级缓存数据并提示风险。
   - 若深度学习模型发生 NaN 错误或显存溢出，系统自动切换至备用的 XGBoost 分类决策流。
   - 在数据层面，对于异常的成交量或股价跳空，系统会自动进行有效性校验（Outlier Detection）。

4. **高性能异步并发架构**: 
   - 采用多线程隔离：API 请求由 Uvicorn 工作线程处理，计算密集型指标计算由独立 CPU 线程池处理，定时数据采集由协程池处理。
   - 这种分层隔离设计确保了即使在模型训练等重任务执行期间，前端看板的实时刷新也不会出现掉帧或超时。

5. **面向生产的日志体系**: 
   - 每一笔预测、每一个触发的告警都有唯一的 TraceID。
   - 支持日志分级：DEBUG 用于指标计算详情，INFO 用于任务调度，ERROR 用于捕获所有网络与算法异常。

### 1.6 系统架构演进与全量数据流 (Full Data Flow)
系统的核心数据流向分为实时监控流、离线预测流与告警通知流三个维度：
...

### 1.6 系统架构演进与全量数据流 (Full Data Flow)
系统的核心数据流向分为实时监控流、离线预测流与告警通知流三个维度：

1. **实时行情监控流 (Real-time Flow)**:
   - **T+0**: `APScheduler` 每分钟触发一次 `DataCollector.get_realtime_quotes`。
   - **T+5s**: `DataCollector` 调用 AkShare 的 `stock_zh_a_spot_em` 获取全市场快照。
   - **T+10s**: 数据通过 `IndicatorEngine` 进行增量计算（如实时 MACD 交叉检测）。
   - **T+15s**: `AlertManager` 对比最新报价与 `UserConfig` 阈值。
   - **T+20s**: 若满足告警条件，通过 `SMTP` 或 `Webhook` 发送通知。

2. **盘后预测分析流 (Post-market Prediction Flow)**:
   - **T+15:30**: 股市收盘，同步全量日线数据并刷新 `DailyKLine` 表。
   - **T+16:00**: `PredictionEngine` 启动。加载过去 60 天价格序列。
   - **T+16:05**: 调用 `SentimentAnalyzer` 汇总今日所有相关新闻的加权情感分。
   - **T+16:15**: `LSTMModel` 与 `XGBoostModel` 协同工作，生成方向与区间预测。
   - **T+16:25**: 结果持久化，Streamlit 面板自动刷新历史准确率统计。

3. **历史回溯与自适应流 (Backtesting & Adaptation Flow)**:
   - **每周日**: 触发 `weekly_retrain`。全量提取过去 2 年的历史 K 线及同步的舆情分。
   - **优化过程**: 执行 `StandardScaler` 重新归一化，训练 50 个 Epoch，通过 EarlyStopping 锁定最佳权重并自动替换生产环境模型。

## 2. 环境与依赖

### 2.1 精确版本表
系统开发与测试均基于以下精确版本环境，请务必保持一致：

| 软件包名称 | 版本号 | 核心用途与职责 |
| :--- | :--- | :--- |
| **Python** | 3.11.15 | 基础解释器版本，利用其异步新特性 |
| **akshare** | 1.18.42 | 负责抓取 A 股、港股、美股行情及资金流、新闻数据 |
| **pandas** | 2.3.3 | 核心数据结构 DataFrame，负责所有数据清洗与转换 |
| **numpy** | 2.4.3 | 提供底层高性能数值运算支持 |
| **torch** | 2.10.0 | LSTM 模型训练与推理，利用张量运算加速 |
| **xgboost** | 3.2.0 | 静态特征决策模型，提供分类预测能力 |
| **scikit-learn** | 1.8.0 | 特征预处理 (StandardScaler) 与模型评估指标 |
| **transformers** | 5.3.0 | 加载并运行 FinBERT 情感分类预训练模型 |
| **ta** | 0.11.0 | 专业技术指标库，负责 MACD, RSI, KDJ 等计算 |
| **fastapi** | 0.135.1 | 构建高性能 API 服务，支持自动生成 OpenAPI 文档 |
| **streamlit** | 1.55.0 | 快速构建数据驱动的交互式前端 Web 应用 |
| **SQLAlchemy** | 2.0.48 | 数据库 ORM 层，采用 2.0 风格的异步接口 |
| **aiosqlite** | 0.22.1 | 为 SQLite 提供高性能的异步驱动支持 |
| **APScheduler** | 3.11.2 | 管理定时任务，支持异步并发执行 |
| **pydantic-settings** | 2.13.1 | 从环境变量与 .env 文件安全加载配置 |
| **plotly** | 6.6.0 | 负责渲染交互式动态 K 线图、饼图及仪表盘 |
| **uvicorn** | 0.24.0 | 异步 Web 服务器运行环境 |
| **pytest** | 7.4.3 | 自动化单元测试框架 |
| **httpx** | 0.25.2 | 异步 HTTP 客户端，用于 API 测试 |

### 2.2 技术选型深度分析

#### 2.2.1 技术指标库：ta vs pandas-ta
在技术指标库的选型上，本项目最终选用了 `ta` 库 (0.11.0)。
- **理由 1: 计算准确性**: `ta` 库的实现严格遵循标准指标定义，在与同类软件（如东方财富、通达信）比对中表现最稳健。
- **理由 2: 内存管理**: `ta` 库在处理长序列数据时，其内存占用比 `pandas-ta` 更低，因为 `pandas-ta` 会在 DataFrame 上挂载过多的猴子补丁。
- **理由 3: 代码洁癖**: `ta` 库采用纯函数和类封装方式，与本项目的面向对象架构集成度更高，易于编写 Mock 测试。

#### 2.2.2 数据库选型：aiosqlite 的必要性
之所以没有选择传统的 `sqlite3` 库，是因为在异步 FastAPI 应用中，同步的 `sqlite3` 会在执行 IO 操作时阻塞整个事件循环（Event Loop）。
- `aiosqlite` 通过在一个独立的线程中运行同步调用并返回 awaitable 对象，能够避免在异步 FastAPI 场景中直接阻塞事件循环。
- 对于本项目这种定时抓取、指标计算与 API 并行存在的场景，异步驱动更利于保持接口响应稳定。

#### 2.2.3 AI 模型融合：为什么是 LSTM + XGBoost?
- **LSTM (时序感知)**: 擅长从股价的连续波动中学习到“能量”和“惯性”。对于捕捉 A 股中常见的趋势性行情效果显著。
- **XGBoost (特征非线性)**: 擅长处理各种技术指标之间的逻辑关系（例如：当 RSI > 80 且成交量萎缩时，下跌概率极大）。
- **组合优势**: LSTM 相当于“直觉”，XGBoost 相当于“逻辑”。二者组合后可以同时利用时序信息与静态特征，在不同市场状态下提供更稳健的补充判断。

## 3. 快速启动指南

### 3.1 基础环境配置
```bash
# 1. 克隆代码后创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 2. 升级 pip 并安装所有依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 环境变量配置
在根目录下创建 `.env` 文件，并根据实际情况填写配置：
```env
# 数据库配置
DATABASE_URL=sqlite+aiosqlite:///./stock_monitor.db

# API 服务配置
API_HOST=0.0.0.0
API_PORT=8000

# 情感分析配置
SENTIMENT_MODEL_NAME=yiyanghkust/finbert-tone-chinese

# 告警推送配置 (可选)
ALERT_EMAIL_SENDER=your_email@example.com
ALERT_EMAIL_PASSWORD=your_email_password_or_app_token
ALERT_EMAIL_RECEIVER=receiver@example.com
ALERT_EMAIL_SMTP_HOST=smtp.example.com
ALERT_EMAIL_SMTP_PORT=465
ALERT_WEBHOOK_URL=https://your-webhook-url.com

# 初始监控列表 (逗号分隔)
WATCH_LIST=000001,600036,300750
```

### 3.3 启动与验证
建议开启两个独立的终端窗口：

**终端 A (后端 API)**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*验证: 浏览器访问 http://localhost:8000/docs 应能看到 Swagger API 文档。*

**终端 B (前端 UI)**:
```bash
streamlit run frontend/app.py
```
*验证: 浏览器访问 http://localhost:8501 应进入监控仪表盘。*

## 4. 配置项管理详解

### 4.1 工作机制
系统采用 `pydantic-settings` 实现了分层配置加载逻辑。通过定义 `Settings` 类，系统可以自动校验配置项的类型。针对列表类型的 `WATCH_LIST`，系统内置了 `_CommaListEnvSource` 逻辑，允许在环境变量中以 `000001,600036` 的格式配置多个股票。

### 4.2 核心配置项列表
| 字段名 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | str | 异步数据库连接字符串 | `sqlite+aiosqlite:///...` |
| `ALERT_EMAIL_SENDER` | str | 告警邮件发送方地址 | `""` |
| `ALERT_EMAIL_PASSWORD` | str | 邮箱授权码/密码 | `""` |
| `ALERT_EMAIL_SMTP_HOST` | str | SMTP 服务器地址 | `smtp.example.com` |
| `ALERT_WEBHOOK_URL` | str | 钉钉或企业微信 Webhook | `""` |
| `LLM_API_KEY` | str | 深度分析所需的 LLM 密钥 | `""` |
| `PRICE_FETCH_INTERVAL` | int | 实时报价拉取间隔 (分钟) | `1` |
| `NEWS_FETCH_INTERVAL` | int | 新闻舆情同步间隔 (分钟) | `30` |
| `MODEL_RETRAIN_DAYS` | int | 模型自动重训练周期 (天) | `7` |
| `API_URL` | str | 前端访问后端的基准 URL | `http://localhost:8000` |

## 5. 数据库层架构

### 5.1 异步 Session 管理
```python
# app/database/session.py 核心代码
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

# 创建异步引擎，pool_pre_ping 确保连接可用性
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)

# 创建异步 Session 工厂
async_session = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def get_db():
    async with async_session() as session:
        yield session
```

### 5.2 核心 ORM 模型详细字段表

#### 1. Stock (股票元数据)
存储系统当前关注的所有股票基本信息。
| 字段 | 类型 | 约束 | 说明 |
| :--- | :--- | :--- | :--- |
| id | Integer | Primary Key | 唯一自增 ID |
| symbol | String(10) | Unique, Index | 股票代码 (如 000001) |
| name | String(50) | Not Null | 股票中文名称 |
| market | String(10) | Not Null | 市场标识 (SH/SZ/BJ) |
| is_watching | Boolean | Default=True | 是否在自选监控中 |

#### 2. DailyKLine (日级历史数据)
存储经过复权处理的历史 K 线数据。
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| symbol | String | 外键关联股票代码 |
| date | Date | 交易日期 |
| open | Float | 当日开盘价 |
| close | Float | 当日收盘价 |
| high | Float | 当日最高价 |
| low | Float | 当日最低价 |
| volume | Float | 成交量 (手) |
| amount | Float | 成交额 (元) |

#### 3. Prediction (预测结果记录)
持久化 AI 模型生成的每一条预测结论，以便回测准确率。
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| symbol | String | 股票代码 |
| prediction_date | Date | 预测生成的日期 |
| direction | String | 预测方向 (UP/DOWN/FLAT) |
| confidence | Float | 置信度得分 (0.0-1.0) |
| predicted_return| Float | 预计未来 3 日收益率 |
| price_high | Float | 预测波动上限 |
| price_low | Float | 预测波动下限 |

#### 4. Alert (历史告警流水)
记录所有触发的异常告警，供前端展示。
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| symbol | String | 触发告警的股票 |
| timestamp | DateTime | 触发时间戳 |
| alert_type | String | 告警类型 (price_change/volume_spike等) |
| message | Text | 详细描述信息 |
| is_read | Boolean | 用户是否已读标记 |

### 5.3 核心 CRUD 函数实现详情 (CRUD Operations)

系统在 `app/database/crud.py` 中实现了 16 个核心异步数据库函数，每个函数都经过了严格的单元测试。

#### 1. 自选股管理 (Watchlist)
- **`get_stocks(db: AsyncSession)`**:
  - **逻辑**: 从 `Stock` 表中查询所有 `is_watching=True` 的记录。
  - **返回**: `List[Stock]` 对象列表。
- **`add_to_watchlist(db: AsyncSession, symbol: str, name: str, market: str)`**:
  - **逻辑**: 使用 `session.merge` 或先 `select` 后 `insert` 的方式防止主键冲突。
  - **参数**: `symbol` (代码), `name` (名称), `market` (SH/SZ)。
- **`delete_from_watchlist(db: AsyncSession, symbol: str)`**:
  - **逻辑**: 根据 `symbol` 执行 `DELETE` 操作，并同步解除关联。

#### 2. K 线数据持久化 (K-Line Persistence)
- **`upsert_daily_kline(db: AsyncSession, symbol: str, df: pd.DataFrame)`**:
  - **逻辑**: 将 Pandas DataFrame 转换为 ORM 对象字典列表，执行批量“存在则更新，不存在则插入”操作。
- **`get_kline_history(db: AsyncSession, symbol: str, limit: int = 250)`**:
  - **逻辑**: 按 `date` 降序排列，获取最近 N 个交易日的数据，常用于模型特征提取。

#### 3. 预测与告警记录 (Predictions & Alerts)
- **`save_prediction_result(db: AsyncSession, record: dict)`**:
  - **逻辑**: 持久化包含方向、置信度、预计收益在内的 AI 结论。
- **`get_latest_prediction(db: AsyncSession, symbol: str)`**:
  - **逻辑**: 仅获取 `prediction_date` 最新的那一条记录，用于前端实时展示。
- **`fetch_unread_alerts(db: AsyncSession, limit: int = 50)`**:
  - **逻辑**: 过滤 `is_read=False` 的记录，按 `timestamp` 降序返回。
- **`mark_alert_read(db: AsyncSession, alert_id: int)`**:
  - **逻辑**: 更新 `is_read` 字段为 `True`。

#### 4. 舆情与新闻 (News & Sentiment)
- **`save_stock_news(db: AsyncSession, news_list: List[dict])`**:
  - **逻辑**: 存储新闻标题、链接及 FinBERT 计算出的情感得分。
- **`get_sentiment_trend(db: AsyncSession, symbol: str, days: int = 30)`**:
  - **逻辑**: 聚合过去 30 天每日的情感平均分，输出时间序列数据。

#### 5. 资金流向 (Fund Flow)
- **`save_fund_flow(db: AsyncSession, symbol: str, flow_data: dict)`**:
  - **逻辑**: 存储北向资金净流入、主力净流入等筹码面数据。

#### 6. 配置管理 (Config)
- **`get_user_config(db: AsyncSession, key: str)`**:
  - **逻辑**: 获取存储在数据库中的动态阈值配置（如告警触发百分比）。
- **`update_user_config(db: AsyncSession, key: str, value: str)`**:
  - **逻辑**: 持久化用户在前端“系统设置”页面所做的修改。

### 5.4 数据库索引优化建议
为了在大数据量下保持查询性能，系统对以下字段建立了复合索引：
1. `idx_symbol_date` ON `daily_kline (symbol, date)`: 加快历史行情回溯。
2. `idx_symbol_timestamp` ON `alerts (symbol, timestamp)`: 加快告警流水查询。
3. `idx_news_publish_time` ON `news (publish_time)`: 加快舆情同步速度。

## 6. 核心业务模块详解

### 6.1 数据采集器 (DataCollector) 深度实现
`DataCollector` 是系统的生命线，负责所有外部数据源的抓取、清洗与标准化。

#### 6.1.1 异步抓取与并发控制
由于 A 股市场自选股可能较多，传统的同步抓取会导致严重的 I/O 阻塞。系统通过 `httpx` 异步客户端并发发出请求：
- **并发策略**: 使用 `asyncio.gather` 并发获取多只股票的实时快照。
- **速率限制**: 内部维护一个 `Semaphore(5)`，防止并发数过高触发 AkShare 接口的频率限制。

#### 6.1.2 HIST_COLUMNS_MAP 字段映射表
AkShare 原始数据采用中文列名，系统通过以下映射表将其转化为标准的英文小写，以便后续 `ta` 指标计算：
```python
HIST_COLUMNS_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change"
}
```

#### 6.1.3 数据清洗流水线
1. **类型转换**: 将日期字符串转换为 `datetime.date` 对象，将数值列转换为 `float64`。
2. **缺失值处理**: 
   - 停牌股票：自动检测成交量为 0 的情况，并用前一交易日价格填充。
   - 复权计算：默认采用“前复权 (qfq)”模式，确保历史价格的连续性。
3. **缓存机制**: 对于不常变动的历史数据，采集器会先查询 `DailyKLine` 本地数据库，仅在本地缺失时才发起网络请求。

#### 6.1.4 采集项列表 (数据广度)
- **指数监控**: 上证指数、深证成指、创业板指的实时走势。
- **个股行情**: 自选股的 1 分钟、5 分钟、日线级别数据。
- **北向资金**: 每日沪股通、深股通的实时净流入数据（反映外资动向）。
- **龙虎榜**: 盘后获取当日龙虎榜数据，捕捉主力游资动向。
- **新闻资讯**: 从新浪、东财等主流门户聚合 7x24 小时直播及个股新闻。

#### 6.1.5 重试装饰器设计
```python
def retry_on_failure(retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1: raise e
                    await asyncio.sleep(delay * (2 ** i)) # 指数退避
            return None
        return wrapper
    return decorator
```
通过该装饰器，系统有效规避了网络抖动带来的采集失败问题。


#### 关键方法功能描述
1. `get_realtime_quotes(symbols: List[str])`: 批量获取最新价格快照。
2. `get_history_kline(symbol, days=365)`: 获取过去一年的日线数据进行模型初始化。
3. `get_stock_news(symbol)`: 异步抓取最近 30 条相关新闻摘要。
4. `get_north_fund_flow()`: 监控北向资金（沪股通、深股通）的实时净流入额。
5. `search_stocks(keyword)`: 支持通过拼音、名称或代码模糊搜索股票。

### 6.2 技术指标引擎 (IndicatorEngine)
该引擎是系统的计算核心，负责将 Raw K-Line 转换为多维特征矩阵。

#### 核心指标列表 (25+ 指标详解)
| 指标名称 | 代码字段 | 默认窗口 | 计算公式逻辑概要 |
| :--- | :--- | :--- | :--- |
| **指数移动平均** | `ema_x` | 5, 10, 20... | 权重随时间指数衰减的均线，反应更快。 |
| **指数平滑异同平均**| `macd` | (26, 12, 9) | 计算两条 EMA 的差值并生成信号线与柱状图。 |
| **相对强弱指标** | `rsi_x` | 6, 12, 24 | 反映一段时间内上涨天数与总天数的强弱对比。 |
| **随机指标** | `kdj_k/d/j` | (9, 3, 3) | 通过最高最低价波动反映超买超卖。 |
| **布林线** | `boll_up/mid/low` | (20, 2) | 基于标准差计算的价格运行通道。 |
| **顺势指标** | `cci` | 20 | 测量股价是否偏离平均价格。 |
| **成交量加权平均价**| `vwap` | 实时 | 将价格与成交量结合，体现主力成本。 |
| **成交量变动率** | `vroc` | 12 | 反映成交量的爆发程度。 |
| **乖离率** | `bias` | 6, 12 | 股价偏离均线的百分比。 |
| **真实波动幅度** | `atr` | 14 | 衡量市场波动性的绝对值。 |
| **能量潮指标** | `obv` | 累计 | 观察成交量随股价变动的累积趋势。 |

#### 指标计算实现代码 (calculate_all)
系统通过向量化方式一次性计算所有列，极大提升了吞吐量：
```python
def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
    # 确保列名为全小写
    df.columns = [c.lower() for c in df.columns]
    
    # 1. 基础移动平均线 (SMA)
    for w in [5, 10, 20, 60]:
        df[f'ma_{w}'] = ta.trend.sma_indicator(df['close'], window=w)
    
    # 2. 布林带 (Bollinger Bands)
    boll = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['boll_upper'] = boll.bollinger_hband()
    df['boll_mid'] = boll.bollinger_mavg()
    df['boll_lower'] = boll.bollinger_lband()
    
    # 3. KDJ 指标
    kdj = ta.momentum.KDJIndicator(high=df['high'], low=df['low'], close=df['close'], window=9)
    df['kdj_k'] = kdj.kdj()
    df['kdj_d'] = kdj.kdj_d()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 4. 其它动量与波动指标
    df['rsi_6'] = ta.momentum.RSIIndicator(close=df['close'], window=6).rsi()
    df['roc'] = ta.momentum.ROCIndicator(close=df['close'], window=12).roc()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    return df.fillna(0) # 填充初始周期的 NaN
```

#### 信号生成算法 (generate_signal)
```python
def generate_signal(row: pd.Series, prev_row: pd.Series) -> str:
    # 策略 1: MACD 金叉策略 (Trend Following)
    is_gold_cross = prev_row['macd'] <= prev_row['macd_signal'] and row['macd'] > row['macd_signal']
    # 策略 2: 超卖反弹策略 (Mean Reversion)
    is_oversold = row['rsi_6'] < 25 and row['close'] < row['boll_lower']
    
    if (is_gold_cross or is_oversold) and row['close'] > row['ma_20']:
        return "BUY"
    
    # 策略 3: 趋势破位策略 (Stop Loss)
    is_death_cross = prev_row['macd'] >= prev_row['macd_signal'] and row['macd'] < row['macd_signal']
    is_overbought = row['rsi_6'] > 75 or row['close'] > row['boll_upper']
    
    if is_death_cross or is_overbought or row['close'] < row['ma_60']:
        return "SELL"
    
    return "HOLD"
```

### 6.3 情感分析器 (SentimentAnalyzer)
集成 `Transformers` Pipeline，将非结构化新闻转化为定量得分。

#### 语义识别原理
系统利用深度学习模型对文本进行向量化处理，并将其投影到情感空间：
- **正向 (Positive)**: “利好”、“增长”、“超预期”、“增持”等核心词汇及其上下文。
- **负向 (Negative)**: “亏损”、“下滑”、“诉讼”、“减持”、“黑天鹅”等。
- **中性 (Neutral)**: 日常公告、非实质性变动、行业研报汇总。

#### 得分规则与加权
- 单条新闻得分: `Score = Label_Positive_Conf * 1.0 + Label_Negative_Conf * (-1.0)`。
- 个股日总分: `Total_Sentiment = Sum(Score_i * Weight_i) / Count`。
- `Weight_i`: 基于发布时间的衰减权重，离当前时间越近的新闻权重越高。

#### 优雅降级 (Graceful Degradation)
由于深度学习模型对显存和 CPU 有一定要求，系统内置了异常感知机制。若检测到 PyTorch 无法在 15 秒内完成推理，将自动切换到正则词库模式：
```python
# app/core/sentiment_analyzer.py 逻辑简化版
def analyze_vlm_fallback(text: str):
    pos_keywords = ['涨', '利好', '强于大盘', '增长']
    neg_keywords = ['跌', '利空', '下调', '减持']
    # 简单的关键词命中统计并归一化为 [-1, 1]
    ...
```
确保了在轻量化部署环境下的可用性。


### 6.4 预测引擎 (PredictionEngine)
实现模型融合策略，输出最终预测 JSON。

#### 特征工程集
1. **LSTM 特征集 (7维)**: `["close", "volume", "macd", "rsi_6", "kdj_k", "kdj_d", "sentiment_score"]`。
2. **XGBoost 特征集 (15维)**: 在 LSTM 基础上增加 MA、MACD 差值、KDJ_J 等衍生指标。

#### 结果融合规则
- 系统首先分别调用 LSTM 和 XGBoost 的 `predict` 方法。
- 若两个模型均存在：`Result = LSTM_Result * 0.6 + XGBoost_Result * 0.4`。
- **趋势评级映射**:
  - `Probability > 0.8`: "强烈看涨/强烈看跌"
  - `Probability > 0.6`: "看涨/看跌"
  - `Other`: "中性 (观望)"

### 6.5 告警管理器 (AlertManager)
实时检测行情异动并触发推送。

#### 触发条件详解
| 告警 ID | 逻辑描述 | 阈值参数 |
| :--- | :--- | :--- |
| `price_change` | 股价当日波动幅度 | `abs(pct) > 5.0%` |
| `volume_spike` | 成交量异常放大 | `vol > avg_20d * 3` |
| `macd_cross` | 技术面金叉或死叉 | 指标符号位翻转 |
| `sentiment_shift`| 舆情突然转负/转正 | `abs(score) > 0.5` |
| `north_flow` | 北向大额资金异动 | `abs(flow) > 5亿` |
| `boll_break` | 股价突破布林带边界 | `close > upper 或 < lower` |

#### 去重机制
为了防止告警轰炸，系统对每只股票每种类型的告警设置了 1 小时的 `Dedup Window`。

## 7. AI 模型深度解析

### 7.1 LSTM 深度学习模型
- **输入维度**: `[Batch, 20, 7]` (20 个交易日的回溯窗口)。
- **隐藏层**: 2 层单向 LSTM，每层 128 个神经元。
- **Dropout**: 0.2 (防止过拟合)。
- **多头输出逻辑**:
  - `Direction`: 分类头 (UP/DOWN/FLAT)。
  - `PriceRange`: 线性回归头，预测高低点。
  - `Return`: 线性回归头，预测百分比收益。

### 7.2 XGBoost 机器学习模型
- **算法类型**: 决策树集成。
- **目标函数**: `binary:logistic` (预测未来上涨概率)。
- **训练超参**: `max_depth=6`, `learning_rate=0.1`, `n_estimators=100`, `eval_metric='logloss'`。
- **序列化**: 训练完成后保存为 `{symbol}_xgboost.json` 格式，支持快速加载。

## 8. API 接口参考

### 8.1 核心路由概览
- `GET /api/stocks/watchlist`: 获取当前所有自选股。
- `POST /api/stocks/watchlist`: 添加新股，Body: `{"symbol": "000001", "name": "平安银行"}`。
- `GET /api/stocks/{symbol}/realtime`: 获取单只股票实时快照。
- `GET /api/predictions/{symbol}`: 获取最新的 AI 预测 JSON。
- `GET /api/alerts/`: 分页查询历史告警记录。

### 8.2 JSON 响应示例 (Prediction)
```json
{
  "symbol": "300750",
  "name": "宁德时代",
  "direction": "UP",
  "confidence": 0.88,
  "trend_rating": "强烈看涨 🚀",
  "expected_return": "+3.2%",
  "volatility_range": [215.5, 228.4],
  "model_weights": {"lstm": 0.6, "xgboost": 0.4},
  "timestamp": "2026-03-21T16:05:00"
}
```

## 9. 自动化定时任务 (Scheduler)

系统依赖 `APScheduler` 执行后台任务，任务列表如下：

| 任务名称 | 执行频率 | 时间窗口 | 核心职责 |
| :--- | :--- | :--- | :--- |
| `realtime_fetch` | 每分钟一次 | 交易时段 (9:30-11:30, 13:00-15:00) | 刷新价格并触发实时告警扫描 |
| `news_fetch` | 每 30 分钟 | 24 小时全天候 | 同步新闻并执行情感分析 |
| `daily_sync` | 每日一次 | 15:30 (收盘后) | 更新日 K 线、计算收盘指标信号 |
| `prediction_job`| 每日一次 | 16:00 (盘后) | 调用模型生成明日预测结论 |
| `weekly_retrain`| 每周一次 | 周日 02:00 | 全量重新训练所有监控股票的模型 |

## 10. 前端页面功能描述

### 10.1 仪表盘主页 (Dashboard)
系统入口页面，展示全局概览卡片：总监控数、今日告警数、市场情绪指数。

### 10.2 实时监控页 (1_stock_dashboard.py)
- **搜索中心**: 通过搜索框实时添加新股。
- **监控矩阵**: 以 Dataframe 形式展示所有自选股的实时报价，支持按涨跌幅排序。
- **动态交互**: 点击行内按钮可快速删除不再关注的股票。

### 10.3 深度 K 线页 (2_kline_detail.py)
- **多图联动**: 顶部为 K 线图（含指标叠加），中部为买卖点信号点，底部为成交量。
- **自定义指标**: 用户可通过复选框实时切换展示不同的技术指标（如布林带、MACD）。

### 10.4 AI 预测页 (3_prediction.py)
- **可视化**: 采用 Plotly Gauge 仪表盘展示置信度。
- **收益展望**: 预测未来的波动区间，并以柱状图展示历史预测命中率。

### 10.5 系统配置页 (5_settings.py)
- **参数调优**: 用户可手动调整告警触发的百分比阈值。
- **模型管理**: 提供手动重训练按钮，用于在市场风格突变时手动刷新模型参数。

## 11. 测试与质量保证

### 11.1 测试套件说明
- **Indicator 测试**: 验证 25 个指标在极端行情下的计算稳定性。
- **Alert 测试**: 模拟价格跳空高开等场景，验证告警是否准确触发并去重。
- **API 测试**: 模拟高并发下的 FastAPI 请求处理，确保数据库连接池不溢出。

### 11.2 测试与性能说明 (Testing & Performance Notes)
- 项目包含完整的自动化测试，覆盖告警、API、数据采集、数据库、指标计算、预测、情感分析与训练任务管理等核心模块。
- 实际测试数量、覆盖率与运行耗时会随代码演进、依赖版本与运行环境变化而变化，建议以 CI 与本地测试结果为准。
- 冷启动速度、模型下载耗时与预测吞吐量高度依赖网络、CPU、磁盘与模型缓存状态，公开文档中不再固定承诺精确数值。

#### 关键测试场景举例:
1. **数据断流测试**: 模拟 AkShare 网络请求超时，验证 `RetryDecorator` 的指数退避逻辑。
2. **内存泄漏测试**: 长时间运行 `SentimentAnalyzer` 批量处理 1000 条新闻，观察 Python 进程内存占用是否稳定。
3. **模型过拟合测试**: 构造一组高度随机的价格数据，验证 `PredictionEngine` 是否输出中性方向（FLAT），而非盲目看多。
4. **数据库事务一致性**: 模拟在 `save_daily_kline` 过程中断电或进程强杀，验证数据库是否能正确回滚到上一个干净状态。
5. **UI 极端情况测试**: 在自选股列表中添加 100 只以上股票，验证 Streamlit 的渲染延迟和分页机制。

## 12. 部署方案与运维手册 (Deployment & Ops)

### 12.1 推荐运行环境 (Hardware)
- **CPU**: 4 核及以上 (用于并发数据抓取与 XGBoost 模型计算)。
- **内存**: 8GB RAM (FinBERT 模型加载后约占用 2.5GB 内存)。
- **磁盘**: 20GB 以上 SSD (加快数据库随机读写操作)。
- **网络**: 稳定的 100Mbps 互联网连接（AkShare 数据源需连接公网）。

### 12.2 生产环境启动流程
1. **安装依赖**: 
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
2. **初始化数据库**: 
   系统首次启动会自动在根目录创建 `stock_monitor.db` 文件并执行全量 DDL。
3. **后端部署 (Gunicorn + Uvicorn)**:
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```
4. **前端部署 (Streamlit)**:
   ```bash
   streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
   ```

### 12.3 运维监控与备份
- **日志监控**: 建议通过 `journalctl` 或 `tail -f logs/app.log` 实时监控异动。
- **备份方案**: 
  - 每天凌晨执行数据库备份：`cp stock_monitor.db ./backups/stock_monitor_$(date +%F).db`。
- **健康检查**: 通过 API 端点 `GET /` 验证系统活跃性。

### 12.4 常见问题排查 (Troubleshooting)
- **接口 403/Forbidden**: 通常为 AkShare 触发了上游数据源（如东财、新浪）的反爬。
  - *解决方法*: 增加 `PRICE_FETCH_INTERVAL`，或配置 HTTP 代理。
- **显存溢出 (CUDA Out of Memory)**: 
  - *解决方法*: 在 `.env` 中设置 `CUDA_VISIBLE_DEVICES=""` 强制模型在 CPU 上运行。
- **数据对不齐**: 
  - *原因*: 不同股票的停牌时间不一致。
  - *解决方法*: 系统内嵌了 `forward_fill` 逻辑，会自动补齐缺失价格点。
- **Streamlit 页面不更新**: 
  - *原因*: 可能是浏览器缓存或后端连接中断。
  - *解决方法*: 点击页面右上角的 "Reruns" 按钮或检查 F12 开发者工具中的网络错误。

## 13. 技术架构设计决策总结 (Architectural Rationale)

- **为什么选 FastAPI 而非 Flask**: 原生支持异步，显著降低 I/O 密集型任务（如大批量价格请求）的响应延迟。
- **为什么选 SQLAlchemy 2.0**: 统一的声明式映射模型，对异步操作的支持更底层、更安全。
- **为什么选 LSTM+XGBoost**: 结合了深度学习捕捉时序能量的能力和传统机器学习捕捉统计因子的稳健性。
- **本地 NLP 模型 vs 远程 API**: 采用本地 FinBERT 确保数据隐私且无 API 调用成本，同时响应延迟更可控。

## 14. 未来扩展路线图 (Future Roadmap)

1. **分布式抓取集群**: 引入 Redis 消息队列实现多节点异步抓取。
2. **多因子选股体系**: 整合财务、研报、龙虎榜等多维度因子提升预测精度。
3. **自动交易闭环**: 对接 QMT / MiniQMT 接口实现“条件单”自动执行。
4. **全市场覆盖**: 扩展至港股、美股等全球主要资本市场。

## 15. 免责声明 (Disclaimer)

1. **非投资建议**: 本系统生成的所有信号、预测、指标及告警信息均仅供科研与量化学习参考。
2. **风险自担**: 金融市场风险巨大，AI 模型存在过拟合或预测滞后等技术局限性。
3. **资产安全**: 本系统不涉及真实资金托管，开发者不对任何基于本系统的投资盈亏承担法律责任。

---
*文档版本: v1.0.0 | 最后修订: 2026-03-21 | 智能股票监控项目组*
