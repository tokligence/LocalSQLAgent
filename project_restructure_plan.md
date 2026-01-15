# 项目重构计划

## 当前问题

1. **根目录混乱**
   - 大量测试结果JSON文件
   - 测试脚本和文档混在根目录
   - 缺少.gitignore来排除临时文件

2. **scripts目录过载**
   - 混合了基准测试、工具、演示等不同用途的代码
   - 缺少清晰的分类

3. **文档分散**
   - 研究文档在根目录
   - 分析报告在docs目录
   - README内容混乱

## 新的目录结构

```
text2sql2026/
├── README.md                    # 简洁的主文档
├── requirements.txt
├── docker-compose.yml
├── .gitignore
│
├── src/                        # 核心源代码
│   ├── core/                   # 核心功能模块
│   │   ├── ambiguity_detection.py
│   │   ├── intelligent_agent.py
│   │   └── schema_discovery.py
│   ├── agents/                 # Agent实现
│   │   ├── exploratory_sql_agent.py
│   │   └── interactive_sql_agent.py
│   └── mongodb/                # MongoDB专用模块
│       ├── benchmark.py
│       └── schema_discovery.py
│
├── benchmarks/                 # 基准测试脚本
│   ├── sql_benchmark.py
│   ├── mongodb_benchmark.py
│   └── multi_db_benchmark.py
│
├── tests/                      # 测试代码
│   ├── test_agents.py
│   ├── test_ambiguous_queries.py
│   └── integration/
│       └── test_mcp_integration.py
│
├── examples/                   # 示例代码
│   ├── quick_start.py
│   ├── production_usage.py
│   └── demos/
│       └── interactive_demo.py
│
├── results/                    # 测试结果（gitignore）
│   └── .gitkeep
│
├── docs/                       # 文档
│   ├── README_CN.md           # 中文文档
│   ├── research/               # 研究文档
│   └── analysis/               # 分析报告
│
└── config/                     # 配置文件
    └── docker/