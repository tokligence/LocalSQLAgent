#!/usr/bin/env python3
"""
Interactive Text-to-SQL Agent
- 分析用户意图，必要时追问澄清
- 结合Schema生成SQL
- 执行验证，错误时自动修正
- 结果展示，请求用户确认
"""

import json
import requests
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import psycopg2
import pymysql
import clickhouse_connect


class AgentState(Enum):
    ANALYZING = "analyzing"       # 分析用户意图
    CLARIFYING = "clarifying"     # 追问澄清
    GENERATING = "generating"     # 生成SQL
    EXECUTING = "executing"       # 执行SQL
    CORRECTING = "correcting"     # 修正错误
    CONFIRMING = "confirming"     # 确认结果
    DONE = "done"


@dataclass
class SchemaInfo:
    """数据库Schema信息，包含业务含义"""
    tables: Dict[str, Dict[str, str]]  # table -> {column: description}
    relationships: List[str]            # 表关系描述
    business_context: str               # 业务背景


# 示例Schema定义（实际使用时从数据库元数据生成）
DEMO_SCHEMA = SchemaInfo(
    tables={
        "users": {
            "id": "用户唯一标识",
            "name": "用户姓名",
            "email": "用户邮箱地址",
            "age": "用户年龄",
            "department_id": "所属部门ID，关联departments表",
            "salary": "月薪（人民币）",
            "created_at": "账号创建时间"
        },
        "departments": {
            "id": "部门唯一标识",
            "name": "部门名称（如Engineering, Sales, Marketing, HR）",
            "budget": "部门年度预算"
        },
        "products": {
            "id": "产品唯一标识",
            "name": "产品名称",
            "category": "产品类别（Electronics电子产品, Furniture家具）",
            "price": "产品单价（注意：这是标价，不是销售额）",
            "stock": "库存数量"
        },
        "orders": {
            "id": "订单唯一标识",
            "user_id": "下单用户ID",
            "amount": "订单总金额（这是实际销售额）",
            "status": "订单状态：pending待处理, completed已完成, shipped已发货, cancelled已取消",
            "created_at": "下单时间"
        },
        "order_items": {
            "id": "订单明细ID",
            "order_id": "所属订单ID",
            "product_id": "产品ID",
            "quantity": "购买数量",
            "unit_price": "成交单价（可能有折扣，与products.price不同）"
        }
    },
    relationships=[
        "users.department_id -> departments.id (多对一)",
        "orders.user_id -> users.id (多对一)",
        "order_items.order_id -> orders.id (多对一)",
        "order_items.product_id -> products.id (多对一)"
    ],
    business_context="""
这是一个电商公司的内部系统数据库，包含：
- 员工管理（users, departments）
- 产品管理（products）
- 订单管理（orders, order_items）

常见业务概念：
- "销售额" 通常指 orders.amount 或 SUM(order_items.quantity * order_items.unit_price)
- "产品价格" 指 products.price（标价）
- "成交价" 指 order_items.unit_price（可能有折扣）
- "员工" 和 "用户" 在本系统中是同一概念
"""
)


class DatabaseConnector:
    """数据库连接器"""

    def __init__(self, db_type: str, **kwargs):
        self.db_type = db_type
        self.config = kwargs
        self.conn = None

    def connect(self):
        if self.db_type == "postgresql":
            self.conn = psycopg2.connect(**self.config)
        elif self.db_type == "mysql":
            self.conn = pymysql.connect(**self.config)
        elif self.db_type == "clickhouse":
            self.conn = clickhouse_connect.get_client(**self.config)

    def execute(self, sql: str) -> Tuple[bool, Any, Optional[str]]:
        """执行SQL，返回 (成功, 结果, 错误信息)"""
        try:
            if self.db_type == "clickhouse":
                result = self.conn.query(sql)
                columns = result.column_names
                rows = result.result_rows
                return True, {"columns": columns, "rows": rows[:20]}, None
            else:
                cursor = self.conn.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(20)
                return True, {"columns": columns, "rows": rows}, None
        except Exception as e:
            return False, None, str(e)

    def close(self):
        if self.conn:
            if self.db_type == "clickhouse":
                self.conn.close()
            else:
                self.conn.close()


class InteractiveSQLAgent:
    """交互式SQL Agent"""

    def __init__(self,
                 model: str = "deepseek-coder:6.7b",
                 db_type: str = "postgresql",
                 db_config: dict = None,
                 schema: SchemaInfo = None,
                 ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.db_type = db_type
        self.db_config = db_config or {}
        self.schema = schema or DEMO_SCHEMA
        self.ollama_url = ollama_url
        self.state = AgentState.ANALYZING
        self.conversation_history: List[Dict[str, str]] = []
        self.current_sql: Optional[str] = None
        self.correction_attempts = 0
        self.max_corrections = 3

        # 数据库连接
        self.db = DatabaseConnector(db_type, **self.db_config)
        self.db.connect()

    def _build_system_prompt(self) -> str:
        """构建包含Schema信息的System Prompt"""
        schema_desc = "## 数据库Schema\n\n"
        for table, columns in self.schema.tables.items():
            schema_desc += f"### 表: {table}\n"
            for col, desc in columns.items():
                schema_desc += f"- `{col}`: {desc}\n"
            schema_desc += "\n"

        schema_desc += "## 表关系\n"
        for rel in self.schema.relationships:
            schema_desc += f"- {rel}\n"

        schema_desc += f"\n## 业务背景\n{self.schema.business_context}\n"

        return f"""你是一个专业的SQL助手。你的任务是帮助用户将自然语言问题转换为正确的SQL查询。

{schema_desc}

## 你的工作流程

1. **分析用户问题**：判断问题是否足够明确
2. **必要时追问**：如果问题有歧义，礼貌地请求澄清
3. **生成SQL**：基于明确的需求生成{self.db_type.upper()}语法的SQL
4. **解释SQL**：简要说明SQL的逻辑

## 重要规则

- 如果用户的问题可能有多种理解，一定要先澄清
- 区分"价格"(products.price)和"销售额"(orders.amount)
- 注意数据库方言差异（PostgreSQL/MySQL/ClickHouse）
- 只输出单个SQL语句，不要用分号结尾后再加内容

## 输出格式

根据当前阶段，使用以下JSON格式响应：

分析阶段（需要澄清时）：
{{"action": "clarify", "question": "你的澄清问题"}}

生成SQL阶段：
{{"action": "execute", "sql": "SELECT ...", "explanation": "这个SQL做了什么"}}

当前数据库类型: {self.db_type.upper()}
"""

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用Ollama LLM"""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        return response.json()["message"]["content"]

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        # 尝试提取JSON
        try:
            # 查找JSON块
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # 如果没有JSON，尝试提取SQL
        if "SELECT" in response.upper() or "INSERT" in response.upper():
            sql = self._extract_sql(response)
            if sql:
                return {"action": "execute", "sql": sql, "explanation": response}

        # 默认当作需要澄清
        return {"action": "clarify", "question": response}

    def _extract_sql(self, text: str) -> Optional[str]:
        """从文本中提取SQL"""
        import re
        # 尝试提取代码块中的SQL
        code_block = re.search(r'```sql?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if code_block:
            return code_block.group(1).strip()

        # 尝试直接匹配SELECT语句
        select_match = re.search(r'(SELECT\s+.*?)(?:;|\n\n|$)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()

        return None

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入，返回Agent响应"""

        self.conversation_history.append({"role": "user", "content": user_input})

        # 构建消息
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self.conversation_history)

        # 调用LLM
        llm_response = self._call_llm(messages)
        parsed = self._parse_response(llm_response)

        self.conversation_history.append({"role": "assistant", "content": llm_response})

        if parsed["action"] == "clarify":
            self.state = AgentState.CLARIFYING
            return {
                "state": self.state.value,
                "message": parsed.get("question", llm_response),
                "needs_input": True
            }

        elif parsed["action"] == "execute":
            self.current_sql = parsed.get("sql", "")
            self.state = AgentState.EXECUTING

            # 执行SQL
            success, result, error = self.db.execute(self.current_sql)

            if success:
                self.state = AgentState.CONFIRMING
                return {
                    "state": self.state.value,
                    "sql": self.current_sql,
                    "explanation": parsed.get("explanation", ""),
                    "result": result,
                    "message": "SQL执行成功。请确认结果是否符合您的预期？",
                    "needs_input": True
                }
            else:
                # 需要修正
                self.correction_attempts += 1
                if self.correction_attempts >= self.max_corrections:
                    self.state = AgentState.DONE
                    return {
                        "state": self.state.value,
                        "error": f"多次修正失败: {error}",
                        "sql": self.current_sql,
                        "needs_input": False
                    }

                self.state = AgentState.CORRECTING
                # 让LLM修正
                correction_prompt = f"""SQL执行出错，请修正。

原SQL: {self.current_sql}
错误信息: {error}

请分析错误原因并生成修正后的SQL。使用JSON格式响应：
{{"action": "execute", "sql": "修正后的SQL", "explanation": "修正说明"}}
"""
                self.conversation_history.append({"role": "user", "content": correction_prompt})
                return self.process_user_input("")  # 递归处理修正

        return {"state": self.state.value, "message": llm_response, "needs_input": True}

    def confirm_result(self, confirmed: bool, feedback: str = "") -> Dict[str, Any]:
        """用户确认结果"""
        if confirmed:
            self.state = AgentState.DONE
            return {
                "state": self.state.value,
                "message": "查询完成！",
                "sql": self.current_sql,
                "needs_input": False
            }
        else:
            # 用户不满意，需要重新理解需求
            self.state = AgentState.ANALYZING
            clarify_prompt = f"用户反馈结果不符合预期: {feedback}\n请重新理解需求并生成正确的SQL。"
            return self.process_user_input(clarify_prompt)

    def reset(self):
        """重置对话"""
        self.conversation_history = []
        self.current_sql = None
        self.correction_attempts = 0
        self.state = AgentState.ANALYZING

    def close(self):
        """关闭连接"""
        self.db.close()


def interactive_session():
    """交互式命令行会话"""
    print("=" * 60)
    print("Interactive Text-to-SQL Agent")
    print("=" * 60)
    print("\n支持的数据库: postgresql, mysql, clickhouse")
    print("输入 'quit' 退出, 'reset' 重新开始\n")

    # 默认使用PostgreSQL
    db_config = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": "benchmark"
    }

    agent = InteractiveSQLAgent(
        model="deepseek-coder:6.7b",
        db_type="postgresql",
        db_config=db_config
    )

    try:
        while True:
            user_input = input("\n[你] ").strip()

            if user_input.lower() == 'quit':
                print("再见！")
                break
            elif user_input.lower() == 'reset':
                agent.reset()
                print("对话已重置。")
                continue
            elif not user_input:
                continue

            response = agent.process_user_input(user_input)

            print(f"\n[Agent] (状态: {response['state']})")

            if 'sql' in response:
                print(f"\n生成的SQL:\n```sql\n{response['sql']}\n```")

            if 'explanation' in response:
                print(f"\n说明: {response['explanation']}")

            if 'result' in response:
                result = response['result']
                print(f"\n查询结果 (前20行):")
                print(f"列: {result['columns']}")
                for row in result['rows'][:5]:
                    print(f"  {row}")
                if len(result['rows']) > 5:
                    print(f"  ... (共{len(result['rows'])}行)")

            if 'message' in response:
                print(f"\n{response['message']}")

            if 'error' in response:
                print(f"\n错误: {response['error']}")

            # 如果是确认阶段，询问用户
            if response['state'] == 'confirming':
                confirm = input("\n结果正确吗? (y/n/说明问题): ").strip().lower()
                if confirm == 'y':
                    final = agent.confirm_result(True)
                    print(f"\n[Agent] {final['message']}")
                elif confirm == 'n':
                    feedback = input("请说明问题: ").strip()
                    agent.confirm_result(False, feedback)
                else:
                    agent.confirm_result(False, confirm)

    finally:
        agent.close()


if __name__ == "__main__":
    interactive_session()
