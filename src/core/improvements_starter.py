"""
Quick-start improvements for LocalSQLAgent
立即可实施的改进代码示例
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import Levenshtein  # pip install python-Levenshtein


class EnhancedColumnMatcher:
    """增强的列名匹配器"""

    def __init__(self):
        # 同义词映射表
        self.synonyms = {
            "name": ["title", "名称", "名字", "label", "description"],
            "year": ["release_year", "年份", "年度", "yr", "anno"],
            "count": ["number", "quantity", "数量", "total", "sum"],
            "age": ["年龄", "years_old", "edad"],
            "id": ["identifier", "key", "code", "编号"],
            "date": ["time", "datetime", "timestamp", "日期", "时间"],
            "price": ["cost", "amount", "value", "价格", "金额"],
            "status": ["state", "condition", "状态", "情况"],
        }

        # 反向映射
        self.reverse_synonyms = {}
        for key, values in self.synonyms.items():
            for value in values:
                self.reverse_synonyms[value] = key

    def find_best_column(
        self,
        target: str,
        available_columns: List[str],
        table_context: Optional[str] = None
    ) -> Optional[str]:
        """
        找到最佳匹配的列名

        Args:
            target: 目标列名
            available_columns: 可用列名列表
            table_context: 表名上下文（用于消歧）

        Returns:
            最佳匹配的列名，如果没有找到返回None
        """
        target_lower = target.lower()

        # 1. 精确匹配
        for col in available_columns:
            if col.lower() == target_lower:
                return col

        # 2. 考虑表前缀的精确匹配
        if table_context:
            prefixed = f"{table_context}.{target}"
            for col in available_columns:
                if col.lower() == prefixed.lower():
                    return col

        # 3. 同义词匹配
        target_root = self.reverse_synonyms.get(target_lower, target_lower)
        for col in available_columns:
            col_lower = col.lower()
            col_root = self.reverse_synonyms.get(col_lower, col_lower)
            if target_root == col_root:
                return col

        # 4. 部分匹配（包含关系）
        for col in available_columns:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col

        # 5. 编辑距离匹配（模糊匹配）
        best_match = None
        best_distance = float('inf')
        threshold = min(3, len(target) // 3)  # 动态阈值

        for col in available_columns:
            distance = Levenshtein.distance(target_lower, col.lower())
            if distance < best_distance and distance <= threshold:
                best_distance = distance
                best_match = col

        return best_match


class JoinPathOptimizer:
    """JOIN路径优化器"""

    @dataclass
    class TableRelation:
        table1: str
        table2: str
        join_column: str
        confidence: float  # 0-1，关系置信度

    def __init__(self, schema_info):
        self.schema = schema_info
        self.relations = self._build_relations()

    def _build_relations(self) -> List[TableRelation]:
        """构建表关系"""
        relations = []

        # 1. 从外键关系构建（最高置信度）
        if hasattr(self.schema, 'foreign_keys'):
            for fk in self.schema.foreign_keys:
                relations.append(self.TableRelation(
                    table1=fk['table'],
                    table2=fk['referenced_table'],
                    join_column=fk['column'],
                    confidence=1.0
                ))

        # 2. 从列名模式推断（中等置信度）
        # 查找 table_id 模式的列
        for table_name, table_info in self.schema.tables.items():
            for column in table_info.columns:
                col_name = column.name.lower()

                # 检查 xxx_id 模式
                if col_name.endswith('_id'):
                    potential_table = col_name[:-3]  # 去掉_id
                    if potential_table in self.schema.tables:
                        relations.append(self.TableRelation(
                            table1=table_name,
                            table2=potential_table,
                            join_column=column.name,
                            confidence=0.7
                        ))

        return relations

    def find_join_path(self, tables: List[str]) -> List[Tuple[str, str, str]]:
        """
        找到连接多个表的最优路径

        Args:
            tables: 需要连接的表列表

        Returns:
            JOIN路径列表，每个元素是(table1, table2, join_condition)
        """
        if len(tables) <= 1:
            return []

        # 使用图算法找最短路径
        # 这里简化实现，实际可以使用Dijkstra算法
        join_path = []
        connected = {tables[0]}
        remaining = set(tables[1:])

        while remaining:
            best_relation = None
            best_confidence = 0

            for relation in self.relations:
                if relation.table1 in connected and relation.table2 in remaining:
                    if relation.confidence > best_confidence:
                        best_relation = relation
                        best_confidence = relation.confidence
                elif relation.table2 in connected and relation.table1 in remaining:
                    if relation.confidence > best_confidence:
                        best_relation = relation
                        best_confidence = relation.confidence

            if best_relation:
                if best_relation.table1 in connected:
                    join_path.append((
                        best_relation.table1,
                        best_relation.table2,
                        f"{best_relation.table1}.{best_relation.join_column} = {best_relation.table2}.id"
                    ))
                    connected.add(best_relation.table2)
                    remaining.remove(best_relation.table2)
                else:
                    join_path.append((
                        best_relation.table2,
                        best_relation.table1,
                        f"{best_relation.table2}.id = {best_relation.table1}.{best_relation.join_column}"
                    ))
                    connected.add(best_relation.table1)
                    remaining.remove(best_relation.table1)
            else:
                # 没有找到连接路径，使用笛卡尔积（警告）
                next_table = remaining.pop()
                join_path.append((
                    list(connected)[0],
                    next_table,
                    "1=1  -- WARNING: No join condition found"
                ))
                connected.add(next_table)

        return join_path


class SQLAutoRepairer:
    """SQL自动修复器"""

    def __init__(self, schema_info):
        self.schema = schema_info
        self.column_matcher = EnhancedColumnMatcher()

    def repair(self, sql: str, error: str) -> Optional[str]:
        """
        根据错误信息自动修复SQL

        Args:
            sql: 原始SQL
            error: 错误信息

        Returns:
            修复后的SQL，如果无法修复返回None
        """
        error_lower = error.lower()

        # 1. 列名错误
        if "no such column" in error_lower or "unknown column" in error_lower:
            return self._fix_column_error(sql, error)

        # 2. 表名错误
        if "no such table" in error_lower or "table" in error_lower and "doesn't exist" in error_lower:
            return self._fix_table_error(sql, error)

        # 3. 歧义列名
        if "ambiguous column" in error_lower:
            return self._fix_ambiguous_column(sql, error)

        # 4. GROUP BY错误
        if "not in group by" in error_lower or "must appear in the group by" in error_lower:
            return self._fix_group_by_error(sql, error)

        # 5. 语法错误
        if "syntax error" in error_lower:
            return self._fix_syntax_error(sql)

        return None

    def _fix_column_error(self, sql: str, error: str) -> Optional[str]:
        """修复列名错误"""
        # 提取错误的列名
        match = re.search(r"column[: ]+(['\"`]?)(\w+)(['\"`]?)", error, re.IGNORECASE)
        if not match:
            return None

        bad_column = match.group(2)

        # 获取所有可用列
        all_columns = []
        for table_name, table_info in self.schema.tables.items():
            for column in table_info.columns:
                all_columns.append(column.name)

        # 找到最佳匹配
        best_match = self.column_matcher.find_best_column(bad_column, all_columns)

        if best_match:
            # 替换SQL中的列名
            # 使用正则表达式进行智能替换，避免替换字符串内的内容
            pattern = r'\b' + re.escape(bad_column) + r'\b'
            fixed_sql = re.sub(pattern, best_match, sql, flags=re.IGNORECASE)
            return fixed_sql

        return None

    def _fix_ambiguous_column(self, sql: str, error: str) -> Optional[str]:
        """修复歧义列名"""
        # 提取歧义列名
        match = re.search(r"column[: ]+(['\"`]?)(\w+)(['\"`]?)", error, re.IGNORECASE)
        if not match:
            return None

        ambiguous_column = match.group(2)

        # 分析SQL找到涉及的表
        tables = self._extract_tables_from_sql(sql)

        if len(tables) > 0:
            # 添加表前缀到歧义列
            # 简单策略：使用第一个包含该列的表
            for table_name in tables:
                if table_name in self.schema.tables:
                    table_info = self.schema.tables[table_name]
                    for column in table_info.columns:
                        if column.name.lower() == ambiguous_column.lower():
                            # 添加表前缀
                            pattern = r'\b' + re.escape(ambiguous_column) + r'\b'
                            replacement = f"{table_name}.{ambiguous_column}"
                            fixed_sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                            return fixed_sql

        return None

    def _fix_group_by_error(self, sql: str, error: str) -> Optional[str]:
        """修复GROUP BY错误"""
        # 简单策略：将SELECT中的非聚合列添加到GROUP BY
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return None

        select_clause = select_match.group(1)

        # 解析SELECT列（简化版）
        columns = []
        for part in select_clause.split(','):
            part = part.strip()
            # 跳过聚合函数
            if not re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", part, re.IGNORECASE):
                # 提取列名（可能有别名）
                col_match = re.match(r"([\w\.]+)(?:\s+AS\s+\w+)?", part, re.IGNORECASE)
                if col_match:
                    columns.append(col_match.group(1))

        if columns:
            # 检查是否已有GROUP BY
            if re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
                # 添加缺失的列到现有GROUP BY
                group_by_match = re.search(r"(GROUP\s+BY\s+)(.*?)(?:\s+HAVING|\s+ORDER|\s*$)",
                                         sql, re.IGNORECASE)
                if group_by_match:
                    existing_group = group_by_match.group(2)
                    new_columns = [c for c in columns if c not in existing_group]
                    if new_columns:
                        new_group = existing_group + ", " + ", ".join(new_columns)
                        fixed_sql = sql[:group_by_match.start(2)] + new_group + sql[group_by_match.end(2):]
                        return fixed_sql
            else:
                # 添加GROUP BY子句
                # 在ORDER BY或LIMIT之前插入
                insert_point = len(sql)
                for keyword in ["ORDER BY", "LIMIT", ";"]:
                    match = re.search(r"\b" + keyword + r"\b", sql, re.IGNORECASE)
                    if match:
                        insert_point = min(insert_point, match.start())

                group_by_clause = " GROUP BY " + ", ".join(columns)
                fixed_sql = sql[:insert_point] + group_by_clause + " " + sql[insert_point:]
                return fixed_sql

        return None

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """从SQL中提取表名"""
        tables = []

        # FROM子句
        from_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))

        # JOIN子句
        join_matches = re.finditer(r"JOIN\s+(\w+)", sql, re.IGNORECASE)
        for match in join_matches:
            tables.append(match.group(1))

        return tables

    def _fix_syntax_error(self, sql: str) -> Optional[str]:
        """修复常见语法错误"""
        fixed_sql = sql

        # 1. 修复多余的逗号
        fixed_sql = re.sub(r",\s*FROM\b", " FROM", fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r",\s*WHERE\b", " WHERE", fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r",\s*GROUP\s+BY\b", " GROUP BY", fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r",\s*\)", ")", fixed_sql)

        # 2. 修复缺失的空格
        fixed_sql = re.sub(r"(\w)(SELECT|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT)",
                          r"\1 \2", fixed_sql, flags=re.IGNORECASE)

        # 3. 修复引号问题
        # 将智能引号替换为标准引号
        fixed_sql = fixed_sql.replace('"', '"').replace('"', '"')
        fixed_sql = fixed_sql.replace(''', "'").replace(''', "'")

        if fixed_sql != sql:
            return fixed_sql

        return None


# 使用示例
if __name__ == "__main__":
    # 测试列名匹配
    matcher = EnhancedColumnMatcher()
    columns = ["singer_name", "age", "Song_release_year", "concert_date"]

    print("列名匹配测试:")
    print(f"'name' -> {matcher.find_best_column('name', columns)}")
    print(f"'year' -> {matcher.find_best_column('year', columns)}")
    print(f"'date' -> {matcher.find_best_column('date', columns)}")

    # 测试SQL修复
    print("\nSQL修复测试:")
    test_sql = "SELECT singer_nam, age FROM singer WHERE age > 25"
    test_error = "no such column: singer_nam"

    # 模拟schema
    class MockSchema:
        def __init__(self):
            self.tables = {
                "singer": type('Table', (), {
                    'columns': [
                        type('Column', (), {'name': 'singer_name'}),
                        type('Column', (), {'name': 'age'}),
                    ]
                })
            }

    repairer = SQLAutoRepairer(MockSchema())
    fixed = repairer.repair(test_sql, test_error)
    print(f"原SQL: {test_sql}")
    print(f"错误: {test_error}")
    print(f"修复后: {fixed}")