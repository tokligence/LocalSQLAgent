#!/usr/bin/env python3
"""
MongoDB Schema Discovery Tool
动态发现MongoDB数据库的结构和字段信息
"""

import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from datetime import datetime
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import re


class MongoSchemaDiscovery:
    """MongoDB Schema 动态发现工具"""

    def __init__(self, connection_string: str):
        """
        初始化连接
        Args:
            connection_string: MongoDB连接字符串
        """
        self.client = MongoClient(connection_string)
        self.db = None

    def connect(self, database: str):
        """连接到指定数据库"""
        self.db = self.client[database]
        return self.db.list_collection_names()

    def analyze_field_type(self, value: Any) -> str:
        """分析字段类型"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, ObjectId):
            return "ObjectId"
        elif isinstance(value, datetime):
            return "date"
        elif isinstance(value, list):
            if value and len(value) > 0:
                # 分析数组元素类型
                element_type = self.analyze_field_type(value[0])
                return f"array<{element_type}>"
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return str(type(value).__name__)

    def infer_field_meaning(self, field_name: str, sample_values: List[Any]) -> str:
        """
        根据字段名和样本值推断字段含义
        """
        field_lower = field_name.lower()

        # 常见字段名映射
        common_fields = {
            "_id": "主键ID",
            "id": "标识符",
            "name": "名称",
            "email": "邮箱地址",
            "phone": "电话号码",
            "age": "年龄",
            "created_at": "创建时间",
            "updated_at": "更新时间",
            "modified_at": "修改时间",
            "deleted_at": "删除时间",
            "status": "状态",
            "price": "价格",
            "amount": "金额/数量",
            "quantity": "数量",
            "description": "描述",
            "category": "分类",
            "type": "类型",
            "user_id": "用户ID",
            "product_id": "产品ID",
            "order_id": "订单ID",
            "department_id": "部门ID",
            "salary": "薪资",
            "budget": "预算",
            "stock": "库存",
            "items": "条目/项目列表"
        }

        # 直接匹配
        if field_lower in common_fields:
            return common_fields[field_lower]

        # 模糊匹配
        for key, meaning in common_fields.items():
            if key in field_lower:
                return meaning

        # 根据值的模式推断
        if sample_values:
            # 检查是否是邮箱
            if all(isinstance(v, str) and '@' in v for v in sample_values[:3] if v):
                return "邮箱地址"

            # 检查是否是URL
            if all(isinstance(v, str) and ('http' in v or 'www' in v) for v in sample_values[:3] if v):
                return "URL链接"

            # 检查是否是日期
            if all(isinstance(v, datetime) for v in sample_values[:3] if v):
                return "时间戳"

            # 检查是否是布尔标志
            if all(isinstance(v, bool) for v in sample_values[:3] if v is not None):
                if 'is_' in field_lower or 'has_' in field_lower:
                    return "布尔标志"

        return "未知用途"

    def analyze_collection_schema(self, collection_name: str, sample_size: int = 100) -> Dict:
        """
        分析单个集合的schema

        Args:
            collection_name: 集合名称
            sample_size: 采样文档数量

        Returns:
            集合的schema信息
        """
        collection = self.db[collection_name]

        # 获取集合统计信息
        stats = self.db.command("collStats", collection_name)
        doc_count = stats.get('count', 0)

        # 采样文档
        sample_docs = list(collection.find().limit(sample_size))

        # 字段统计
        field_info = defaultdict(lambda: {
            "types": defaultdict(int),
            "values": [],
            "nullable": False,
            "array_element_types": set()
        })

        # 分析每个文档
        for doc in sample_docs:
            self._analyze_document(doc, field_info)

        # 构建schema
        schema = {
            "collection": collection_name,
            "document_count": doc_count,
            "sample_size": len(sample_docs),
            "indexes": self._get_indexes(collection),
            "fields": {}
        }

        # 处理字段信息
        for field_name, info in field_info.items():
            # 找出最常见的类型
            if info["types"]:
                primary_type = max(info["types"].items(), key=lambda x: x[1])[0]
            else:
                primary_type = "unknown"

            # 获取样本值（去重）
            unique_values = []
            seen = set()
            for v in info["values"][:10]:  # 最多10个样本
                try:
                    v_str = str(v) if v is not None else None
                    if v_str not in seen and v is not None:
                        seen.add(v_str)
                        unique_values.append(v)
                except:
                    # 对于不能转换为字符串的值，跳过
                    continue

            schema["fields"][field_name] = {
                "type": primary_type,
                "types_distribution": dict(info["types"]),
                "nullable": info["nullable"],
                "meaning": self.infer_field_meaning(field_name, info["values"]),
                "sample_values": unique_values[:5],
                "unique_count": len(set(str(v) for v in info["values"] if v is not None))
            }

            # 如果是数组，记录元素类型
            if primary_type.startswith("array"):
                schema["fields"][field_name]["element_types"] = list(info["array_element_types"])

        return schema

    def _analyze_document(self, doc: Dict, field_info: Dict, prefix: str = ""):
        """递归分析文档结构"""
        for key, value in doc.items():
            field_path = f"{prefix}.{key}" if prefix else key

            # 记录类型
            field_type = self.analyze_field_type(value)
            field_info[field_path]["types"][field_type] += 1

            # 记录值
            if value is not None:
                field_info[field_path]["values"].append(value)
            else:
                field_info[field_path]["nullable"] = True

            # 如果是数组，分析元素
            if isinstance(value, list) and value:
                for item in value[:5]:  # 采样前5个元素
                    item_type = self.analyze_field_type(item)
                    field_info[field_path]["array_element_types"].add(item_type)

                    # 如果数组元素是对象，递归分析
                    if isinstance(item, dict):
                        self._analyze_document(item, field_info, f"{field_path}[]")

            # 如果是嵌套对象，递归分析
            elif isinstance(value, dict):
                self._analyze_document(value, field_info, field_path)

    def _get_indexes(self, collection) -> List[Dict]:
        """获取集合的索引信息"""
        indexes = []
        for index in collection.list_indexes():
            indexes.append({
                "name": index.get("name"),
                "keys": list(index.get("key").keys()),
                "unique": index.get("unique", False),
                "sparse": index.get("sparse", False)
            })
        return indexes

    def discover_database_schema(self, database: str) -> Dict:
        """
        发现整个数据库的schema

        Args:
            database: 数据库名称

        Returns:
            完整的数据库schema
        """
        collections = self.connect(database)

        db_schema = {
            "database": database,
            "collections": {},
            "relationships": self._infer_relationships(collections)
        }

        print(f"发现 {len(collections)} 个集合")

        for collection_name in collections:
            print(f"分析集合: {collection_name}")
            schema = self.analyze_collection_schema(collection_name)
            db_schema["collections"][collection_name] = schema

        return db_schema

    def _infer_relationships(self, collections: List[str]) -> List[Dict]:
        """推断集合之间的关系（基于字段名）"""
        relationships = []

        for coll in collections:
            # 查找可能的外键字段
            sample = self.db[coll].find_one()
            if sample:
                for field, value in sample.items():
                    # 检查是否是外键模式 (xxx_id)
                    if field.endswith('_id') and field != '_id':
                        possible_ref = field[:-3]  # 去掉_id

                        # 检查是否有对应的集合
                        if possible_ref in collections:
                            relationships.append({
                                "from": coll,
                                "to": possible_ref,
                                "field": field,
                                "type": "reference"
                            })
                        elif possible_ref + 's' in collections:
                            relationships.append({
                                "from": coll,
                                "to": possible_ref + 's',
                                "field": field,
                                "type": "reference"
                            })

                    # 检查数组中的引用
                    elif isinstance(value, list) and value:
                        if isinstance(value[0], dict) and any(k.endswith('_id') for k in value[0].keys()):
                            for k in value[0].keys():
                                if k.endswith('_id'):
                                    possible_ref = k[:-3]
                                    if possible_ref in collections or possible_ref + 's' in collections:
                                        relationships.append({
                                            "from": coll,
                                            "to": possible_ref if possible_ref in collections else possible_ref + 's',
                                            "field": f"{field}.{k}",
                                            "type": "array_reference"
                                        })

        return relationships

    def generate_schema_prompt(self, db_schema: Dict) -> str:
        """
        生成用于LLM的schema描述prompt

        Args:
            db_schema: 数据库schema信息

        Returns:
            格式化的schema描述
        """
        prompt_lines = [f"MongoDB Database: {db_schema['database']}\n"]

        for coll_name, coll_schema in db_schema["collections"].items():
            prompt_lines.append(f"\nCollection: {coll_name} ({coll_schema['document_count']} documents)")
            prompt_lines.append("Fields:")

            for field_name, field_info in coll_schema["fields"].items():
                # 构建字段描述
                field_desc = f"  - {field_name}: {field_info['type']}"

                # 添加含义
                if field_info['meaning'] != "未知用途":
                    field_desc += f" // {field_info['meaning']}"

                # 添加是否可空
                if field_info['nullable']:
                    field_desc += " (nullable)"

                # 添加样本值
                if field_info.get('sample_values'):
                    samples = [str(v)[:20] for v in field_info['sample_values'][:3]]
                    field_desc += f" [示例: {', '.join(samples)}]"

                prompt_lines.append(field_desc)

            # 添加索引信息
            if coll_schema.get("indexes"):
                prompt_lines.append("Indexes:")
                for index in coll_schema["indexes"]:
                    if index["name"] != "_id_":  # 跳过默认主键索引
                        index_desc = f"  - {index['name']}: {', '.join(index['keys'])}"
                        if index.get('unique'):
                            index_desc += " (unique)"
                        prompt_lines.append(index_desc)

        # 添加关系信息
        if db_schema.get("relationships"):
            prompt_lines.append("\n关系:")
            for rel in db_schema["relationships"]:
                prompt_lines.append(f"  - {rel['from']}.{rel['field']} -> {rel['to']} ({rel['type']})")

        return "\n".join(prompt_lines)

    def save_schema(self, db_schema: Dict, filename: str):
        """保存schema到文件"""
        # 转换不可序列化的类型
        def convert(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            return obj

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(db_schema, f, ensure_ascii=False, indent=2, default=convert)

        print(f"Schema保存到: {filename}")


def main():
    """测试schema发现"""
    import argparse

    parser = argparse.ArgumentParser(description='MongoDB Schema Discovery Tool')
    parser.add_argument('--host', default='localhost', help='MongoDB host')
    parser.add_argument('--port', type=int, default=27017, help='MongoDB port')
    parser.add_argument('--database', default='benchmark', help='Database name')
    parser.add_argument('--username', default='text2sql', help='Username')
    parser.add_argument('--password', default='text2sql123', help='Password')
    parser.add_argument('--output', default='mongodb_schema.json', help='Output file')

    args = parser.parse_args()

    # 构建连接字符串
    connection_string = f"mongodb://{args.username}:{args.password}@{args.host}:{args.port}/"

    # 创建发现工具
    discovery = MongoSchemaDiscovery(connection_string)

    try:
        # 发现schema
        print(f"连接到 MongoDB {args.host}:{args.port}/{args.database}")
        db_schema = discovery.discover_database_schema(args.database)

        # 保存到文件
        discovery.save_schema(db_schema, args.output)

        # 生成prompt
        prompt = discovery.generate_schema_prompt(db_schema)
        print("\n" + "="*60)
        print("生成的Schema Prompt:")
        print("="*60)
        print(prompt)

        # 打印统计信息
        print("\n" + "="*60)
        print("Schema统计:")
        print("="*60)
        print(f"集合数: {len(db_schema['collections'])}")
        for coll_name, coll_info in db_schema['collections'].items():
            print(f"  - {coll_name}: {len(coll_info['fields'])} 字段, {coll_info['document_count']} 文档")

        if db_schema.get('relationships'):
            print(f"\n发现 {len(db_schema['relationships'])} 个潜在关系")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()