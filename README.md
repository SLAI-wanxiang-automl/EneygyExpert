# EneygyExpert

面向光伏、风电等**新能源出力预测**的 Agentic AutoML 系统。项目以 LLM 辅助分析和候选模型探索，以可复现的滚动评测作为模型选型依据，统一统计模型、机器学习、深度学习与时间序列基础模型的使用方式。

> **第一次运行可能比较长。** 安装科学计算与深度学习依赖、首次编译计算内核、下载基础模型权重，以及多个滚动窗口中的超参数搜索都可能耗时。建议先用小规模数据和 SeasonalNaive 验证流程，再增加模型、搜索次数和预测窗口。

## 方法与特点

- **新能源场景的数据协议**：统一目标 `y`、历史协变量 `H` 和未来已知协变量 `F`。支持单变量预测、带多变量历史信息的预测，以及使用天气预报/NWP 等未来外生信息的预测；`unique_id` 支持多条序列。这里的“多元”主要指协变量输入，不代表所有适配器都支持联合多目标输出。
- **跨模型家族适配**：StatsForecast、MLForecast、NeuralForecast、Prophet、sktime 与基础模型通过统一窗口接口参与评测。能力按模型显式声明，不能假设每个模型都支持 H/F；可用 `model_catalog()` 查询。
- **训练内调参，验证集选型**：按时间顺序划分训练/验证/测试集（默认 60%/20%/20%）。在训练集内部按统一 horizon、step 和窗口数调参，确定各候选模型的超参数；在验证集上通过 MAE、RMSE 或 MASE 排名，选择 top-k；测试集只用于最终报告。模型差异由指标客观衡量。
- **明确的滚动更新语义**：区分固定参数预测与状态更新，历史观测可以进入后续预测窗口。框架调参和原生 AutoML 均遵守训练内搜索约束，并记录配置与日志。
- **可追踪的集成和产物**：支持模型级中位数集成、预测级中位数集成，以及使用验证集 MAE 确定权重的预测集成，输出排名、逐窗口预测、指标与图表。

这些是本项目的设计特点，不是已证明的学术首创性结论。相对于只封装一种模型家族的工具，本项目强调能源协变量协议、异构模型适配与统一滚动评测的组合。

### LLM 与评测流水线的当前边界

`EneygyExpert.analyze()` 提供基于工具调用的 LLM 分析、交叉验证和预测能力，目前保留上游的单变量 Agent 工作流。`train_validation_test()` 是支持 H/F 的独立实验流水线，按指标确定赢家；目前需要代码配置候选模型与搜索预算。**完整的“LLM 自动编排 H/F 流水线并自动选择全部超参”是项目目标，尚不能视为已经贯通的单一接口。** 原生调参通常由 Optuna/模型后端执行。

验证阶段从训练区间拟合；最终测试阶段会用已固定的配置在训练+验证历史上重新拟合。因此测试阶段的权重并非训练阶段权重的原封不动复用，超参数搜索不会在验证/测试集上重复执行。

## 安装

建议新建 Python 3.11 或 3.12 环境。从私有仓库获取代码前，需要 GitHub 访问权限。仓库名与包名保留 `EneygyExpert` / `eneygyexpert` 的现有拼写。

```bash
git clone git@github.com:humble-gambler/EneygyExpert.git
cd EneygyExpert
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

也可使用 uv 安装锁定依赖与开发工具：

```bash
uv sync --locked --group dev
uv run python examples/quickstart.py
```

`pip install -e .` 会解析 pyproject.toml 中的版本范围；`uv sync --locked` 使用 uv.lock。不要直接复用其他机器复制来的 `.venv`。当前依赖较重，部分基础模型需要网络与模型访问权限；深度模型建议使用适配的 GPU/PyTorch 环境。仅运行确定性流水线不需要 LLM API key。项目尚未承诺已发布到 PyPI，请从源码安装。

## 快速教程：不需要 LLM 的完整滚动实验

```bash
python examples/quickstart.py
```

该示例生成合成日序列，比较两个季节性基线，按验证 MAE 选择模型并保存报告到 `outputs/quickstart/`。它适合首先检查数据划分和运行环境。

```python
import pandas as pd
from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models import create_model

# CSV 至少包含 unique_id、ds、y；ds 为时间，y 为功率。
frame = pd.read_csv("power.csv")
data = ForecastData(frame, freq="15min")
models = [
    create_model("SeasonalNaive", season_length=96),
    create_model("AutoARIMA", season_length=96, num_samples=3),
]
result = train_validation_test(
    data, models, h=96, step_size=96, seasonality=96, metric="mae",
)
print(result.leaderboard)
print(result.selected_model)
print(result.test.summary)
```

保持时间戳规则、无重复、无缺失，且各序列共享同一时间网格。确保各区间足够容纳训练历史和完整预测窗口。`h=96` 对 15 分钟数据表示预测未来一天；`step_size=96` 表示每天推进一次预测起点。

## 使用历史天气与未来天气预报

```python
from eneygyexpert.data import ForecastData
from eneygyexpert.rolling_models import create_model, model_catalog

# history 含 unique_id、ds、y、measured_temperature。
# nwp 含 unique_id、ds、forecast_temperature，预测 horizon 必须完整覆盖。
data = ForecastData(
    history, freq="15min",
    historical_covariate_cols=("measured_temperature",),
    future_covariate_cols=("forecast_temperature",),
    future_covariates=nwp,
)
model = create_model(
    "AutoTFT", h=96, context_length=672, num_samples=3,
    historical_covariate_cols=("measured_temperature",),
    future_covariate_cols=("forecast_temperature",),
)
# 将 data 和 [model] 传入上述 train_validation_test()。
print(model_catalog())
```

未来外生变量必须是**在对应预测起点已经发布、可获得的预报**，不能用事后实测天气替代。当前数据结构不自动核验天气预报的发布时间/版本，需要在数据准备时按预测起点对齐。

PVOD 示例：准备自己的数据后执行（原始数据不随源码发布）：

```bash
python examples/pvod_models_pipeline.py /path/to/station.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

更完整的参数和数据列处理见 `examples/pvod_models_pipeline.py`、`examples/pvod_split_pipeline.py` 与 `docs/rolling_models.md`。

## 使用 LLM Agent

使用支持 OpenAI 协议的本地 vLLM 服务：

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export EE_LLM=openai:qwen3
python agent_local_smoke.py
```

服务端模型名必须与 `EE_LLM` 一致。使用远程供应商时设置相应模型名和真实凭据；凭据通过环境变量传入。该例会调用服务分析合成数据，不使用 PVOD 原始数据。

## 测试与开发

```bash
uv run pytest -o addopts='' tests/test_data.py tests/test_window_models.py \
  tests/test_backtesting.py tests/test_tuning.py tests/test_test_rolling.py \
  tests/test_pipeline.py tests/test_native_automl.py
uv run pytest
```

默认测试配置排除 docs/live/distributed，原生模型测试需要相应后端依赖；完整运行可能耗时较长。模型的真实后端测试与轻量协议测试应分别评估，跳过依赖测试不表示该后端已验证。

`MedianEnsembleAdapter` 的默认模型别名为 `MedianModelEnsemble`；`MedianEnsemble` 与 `WeightedEnsemble` 保留给流水线生成的预测级集成。

## 项目结构

- `eneygyexpert/rolling_models/`：模型能力、适配、集成、报告与绘图。
- `eneygyexpert/data.py`：窗口数据协议与时间划分。
- `eneygyexpert/pipeline.py`：训练内调参、验证选型与测试。
- `eneygyexpert/agent.py`：LLM 工具调用与分析。
- `examples/`、`scripts/`、`tests/`：教程、实验入口与测试。

## 来源与许可证

项目基于 [TimeCopilot](https://github.com/TimeCopilot/timecopilot) 扩展，保留上游 MIT 版权声明。代码采用 [MIT License](LICENSE)，允许使用、修改、分发和商业使用，分发时需保留版权与许可证声明。依赖库、第三方模型权重和数据集遵循其各自许可证；本项目 MIT 不替代它们的授权条件。

GitHub 仓库保持 private；代码许可证和仓库可见性是两个独立设置。
