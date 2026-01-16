"""
Intelligent SQL Agent with Multi-Strategy Execution
Production-ready agent combining ambiguity detection, schema discovery, and adaptive strategies
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .ambiguity_detection import AmbiguityDetector, DetectedAmbiguity, AmbiguityType
from .schema_discovery import (
    SchemaManager,
    SchemaInfo,
    DatabaseIntrospectionProvider,
    MCPSchemaProvider
)


class ExecutionStrategy(Enum):
    """SQL generation strategies"""
    DIRECT = "direct"              # Single attempt, simple queries
    VALIDATED = "validated"        # With validation queries first
    EXPLORATORY = "exploratory"    # Multiple attempts with learning
    CLARIFYING = "clarifying"      # Needs user clarification first


class QueryDifficulty(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"        # Single table, basic conditions
    MODERATE = "moderate"    # 2-3 tables, aggregations
    COMPLEX = "complex"      # Multiple joins, subqueries, window functions
    EXPERT = "expert"        # CTEs, recursive queries, complex business logic


@dataclass
class QueryContext:
    """Context for query execution"""
    original_query: str
    clarified_query: Optional[str] = None
    detected_ambiguities: Optional[List[DetectedAmbiguity]] = None
    difficulty: Optional[QueryDifficulty] = None
    selected_strategy: Optional[ExecutionStrategy] = None
    schema_info: Optional[SchemaInfo] = None
    validation_results: Optional[List[Dict]] = None
    attempts: List[Dict] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
        if self.start_time is None:
            self.start_time = time.time()


@dataclass
class ExecutionResult:
    """Result of query execution"""
    success: bool
    sql: Optional[str] = None
    data: Optional[List] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    error: Optional[str] = None
    execution_time: float = 0
    attempts_count: int = 1
    strategy_used: Optional[ExecutionStrategy] = None
    confidence: float = 0.0
    context: Optional[QueryContext] = None
    results: Optional[List[Dict[str, Any]]] = None
    affected_rows: Optional[int] = None


@dataclass
class ExecutionPolicy:
    """Execution policy for safety and guardrails"""
    read_only: bool = True
    allow_ddl: bool = False
    allow_dml: bool = False
    allow_admin: bool = False
    allow_multi_statement: bool = True
    default_limit: int = 10000
    enforce_default_limit: bool = True


class QueryDifficultyAssessor:
    """Assess query complexity for strategy selection"""

    def __init__(self):
        self.complexity_indicators = {
            # Join indicators
            "multiple_tables": ["join", "from.*,", "关联", "连接"],
            # Aggregation indicators
            "aggregation": ["sum", "avg", "count", "max", "min", "group by",
                          "总计", "平均", "统计", "最大", "最小"],
            # Complex logic indicators
            "subquery": ["not in", "not exists", "exists", "any", "all",
                        "但没有", "除了", "不包括"],
            # Window function indicators
            "window": ["rank", "row_number", "over", "partition by",
                      "排名", "前n", "百分比"],
            # CTE indicators
            "cte": ["with", "recursive", "递归"],
            # Complex conditions
            "complex_conditions": ["case when", "having", "union", "intersect"]
        }

        self.weights = {
            "multiple_tables": 0.25,
            "aggregation": 0.20,
            "subquery": 0.25,
            "window": 0.30,
            "cte": 0.35,
            "complex_conditions": 0.20
        }

    def assess(self, query: str, schema: Optional[SchemaInfo] = None) -> Tuple[QueryDifficulty, float]:
        """
        Assess query difficulty

        Returns:
            Tuple of (difficulty_level, confidence_score)
        """
        query_lower = query.lower()
        complexity_score = 0.0
        detected_features = []

        # Check for complexity indicators
        for feature, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    complexity_score += self.weights[feature]
                    detected_features.append(feature)
                    break  # Only count each feature once

        # Adjust based on schema if available
        if schema:
            # Check if query mentions multiple tables
            mentioned_tables = sum(1 for table in schema.tables if table.lower() in query_lower)
            if mentioned_tables > 1:
                complexity_score += 0.15

        # Determine difficulty level
        if complexity_score < 0.2:
            difficulty = QueryDifficulty.SIMPLE
        elif complexity_score < 0.5:
            difficulty = QueryDifficulty.MODERATE
        elif complexity_score < 0.8:
            difficulty = QueryDifficulty.COMPLEX
        else:
            difficulty = QueryDifficulty.EXPERT

        confidence = min(0.95, 0.6 + complexity_score * 0.4)

        return difficulty, confidence


class StrategySelector:
    """Select optimal execution strategy based on context"""

    def __init__(self, difficulty_assessor: QueryDifficultyAssessor,
                 ambiguity_detector: AmbiguityDetector):
        self.difficulty_assessor = difficulty_assessor
        self.ambiguity_detector = ambiguity_detector

        # Strategy selection rules
        self.strategy_rules = {
            QueryDifficulty.SIMPLE: ExecutionStrategy.DIRECT,
            QueryDifficulty.MODERATE: ExecutionStrategy.VALIDATED,
            QueryDifficulty.COMPLEX: ExecutionStrategy.EXPLORATORY,
            QueryDifficulty.EXPERT: ExecutionStrategy.EXPLORATORY
        }

    def select(self, context: QueryContext) -> ExecutionStrategy:
        """
        Select execution strategy based on context

        Priority:
        1. If ambiguities detected -> CLARIFYING
        2. Based on difficulty -> corresponding strategy
        3. If previous attempts failed -> upgrade strategy
        """
        # Check for ambiguities first
        if context.detected_ambiguities and len(context.detected_ambiguities) > 0:
            # Only require clarification for high-confidence ambiguities
            high_conf_ambiguities = [a for a in context.detected_ambiguities if a.confidence > 0.75]
            if high_conf_ambiguities:
                return ExecutionStrategy.CLARIFYING

        # Assess difficulty if not done
        if not context.difficulty:
            difficulty, _ = self.difficulty_assessor.assess(
                context.clarified_query or context.original_query,
                context.schema_info
            )
            context.difficulty = difficulty

        # Get base strategy from difficulty
        base_strategy = self.strategy_rules[context.difficulty]

        # Upgrade strategy if previous attempts failed
        if context.attempts:
            failed_attempts = [a for a in context.attempts if not a.get("success")]
            if len(failed_attempts) >= 1:
                if base_strategy == ExecutionStrategy.DIRECT:
                    return ExecutionStrategy.VALIDATED
                elif base_strategy == ExecutionStrategy.VALIDATED:
                    return ExecutionStrategy.EXPLORATORY

        return base_strategy


class IntelligentSQLAgent:
    """
    Production-ready SQL agent with adaptive strategies
    """

    def __init__(self,
                 model_name: str,
                 db_config: Dict[str, Any],
                 mcp_server: Optional[str] = None,
                 max_attempts: int = 5,
                 execution_policy: Optional[ExecutionPolicy] = None,
                 schema_allow_fallback: bool = False):
        """
        Initialize intelligent agent

        Args:
            model_name: LLM model to use
            db_config: Database connection configuration
            mcp_server: Optional MCP server URL for schema
            max_attempts: Maximum query attempts
        """
        self.model_name = model_name
        self.db_config = db_config
        self.db_type = db_config.get("type", "postgresql")
        self.max_attempts = max_attempts
        self.execution_policy = execution_policy or ExecutionPolicy()
        policy_overrides = {}
        if isinstance(db_config, dict):
            policy_overrides.update(db_config.get("execution_policy", {}))
            for key in (
                "read_only",
                "allow_ddl",
                "allow_dml",
                "allow_admin",
                "allow_multi_statement",
                "default_limit",
                "enforce_default_limit"
            ):
                if key in db_config:
                    policy_overrides[key] = db_config[key]
        for key, value in policy_overrides.items():
            if hasattr(self.execution_policy, key):
                setattr(self.execution_policy, key, value)
        if self.execution_policy.read_only and (
            self.execution_policy.allow_dml
            or self.execution_policy.allow_ddl
            or self.execution_policy.allow_admin
        ):
            self.logger.warning("Read-only overridden by write permissions")
            self.execution_policy.read_only = False

        # Initialize components
        self.ambiguity_detector = AmbiguityDetector(confidence_threshold=0.7)
        self.difficulty_assessor = QueryDifficultyAssessor()
        self.strategy_selector = StrategySelector(
            self.difficulty_assessor,
            self.ambiguity_detector
        )

        # Setup schema management with logging
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        try:
            from utils.logger import get_logger
            self.logger = get_logger(f"IntelligentSQLAgent.{self.db_type}")
            self.logger.info(f"Initializing agent for {db_config.get('type')} database")
        except ImportError:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Custom logger not available, using basic logging")

        providers = []

        # Primary: Database introspection
        # Extract connection parameters (excluding non-connection keys)
        excluded_keys = {
            "type",
            "execution_policy",
            "read_only",
            "allow_ddl",
            "allow_dml",
            "allow_admin",
            "allow_multi_statement",
            "default_limit",
            "enforce_default_limit",
            "schema_options",
            "schema",
            "schemas",
            "include_samples",
            "include_row_counts",
            "sample_rows",
            "row_count_strategy",
            "max_tables",
            "max_columns",
            "temperature",
            "max_tokens"
        }
        connection_params = {k: v for k, v in db_config.items() if k not in excluded_keys}

        introspection_options: Dict[str, Any] = {}
        if isinstance(db_config, dict):
            if isinstance(db_config.get("schema_options"), dict):
                introspection_options.update(db_config.get("schema_options", {}))
            for key in (
                "schema",
                "schemas",
                "include_samples",
                "include_row_counts",
                "sample_rows",
                "row_count_strategy",
                "max_tables",
                "max_columns"
            ):
                if key in db_config:
                    introspection_options[key] = db_config[key]

        # Mask password for logging
        safe_params = {k: '***' if k == 'password' else v for k, v in connection_params.items()}
        self.logger.debug(f"Creating DatabaseIntrospectionProvider with params: {safe_params}")

        try:
            provider = DatabaseIntrospectionProvider(
                self.db_type,
                connection_params,
                introspection_options=introspection_options
            )
            providers.append(provider)
            self.logger.info("DatabaseIntrospectionProvider created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create DatabaseIntrospectionProvider: {str(e)}")
            raise

        # Fallback: MCP if configured
        if mcp_server:
            providers.append(MCPSchemaProvider(mcp_server))

        self.schema_manager = SchemaManager(
            primary_provider=providers[0],
            fallback_providers=providers[1:] if len(providers) > 1 else None,
            allow_fallback_schema=schema_allow_fallback
        )

        # Cache with size limit to prevent memory leak
        self.query_cache = {}
        self.schema_cache = None
        self.max_cache_size = 100  # Limit cache to 100 queries
        self.llm_temperature = None
        self.llm_max_tokens = None
        if isinstance(db_config, dict):
            for key in ("temperature", "llm_temperature"):
                if key in db_config:
                    try:
                        self.llm_temperature = float(db_config[key])
                    except Exception:
                        pass
            for key in ("max_tokens", "llm_max_tokens"):
                if key in db_config:
                    try:
                        self.llm_max_tokens = int(db_config[key])
                    except Exception:
                        pass

    def execute_query(self, query: str, force_refresh: bool = False) -> ExecutionResult:
        """
        Execute user query with intelligent strategy selection

        Args:
            query: User's natural language query
            force_refresh: Force schema refresh

        Returns:
            ExecutionResult with data or error
        """
        # Check cache first
        cache_key = "|".join([
            query.lower().strip(),
            f"read_only={self.execution_policy.read_only}",
            f"allow_dml={self.execution_policy.allow_dml}",
            f"allow_ddl={self.execution_policy.allow_ddl}",
            f"default_limit={self.execution_policy.default_limit}",
            f"multi={self.execution_policy.allow_multi_statement}"
        ])
        if not force_refresh and cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            cached.confidence = 1.0  # Cached results have high confidence
            return cached

        # Initialize context
        context = QueryContext(original_query=query)

        try:
            # 1. Get schema (dynamic discovery)
            context.schema_info = self._get_schema(force_refresh)
            if not context.schema_info or not context.schema_info.tables:
                return ExecutionResult(
                    success=False,
                    error="Schema discovery failed or returned no tables",
                    context=context
                )

            # 2. Detect ambiguities
            context.detected_ambiguities = self.ambiguity_detector.detect(query)

            # 3. Handle clarification if needed
            if self._needs_clarification(context):
                return self._request_clarification(context)

            # 4. Select and execute strategy
            context.selected_strategy = self.strategy_selector.select(context)

            # 5. Execute with selected strategy
            result = self._execute_with_strategy(context)
            result = self._attempt_fallback(context, result)

            # 6. Cache successful results (with size limit)
            if result.success and result.confidence > 0.7:
                # Check cache size and remove oldest if necessary
                if len(self.query_cache) >= self.max_cache_size:
                    # Remove the first (oldest) item
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[cache_key] = result

            # 7. Record timing
            context.end_time = time.time()
            result.execution_time = context.end_time - context.start_time
            result.context = context

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                context=context
            )

    def _get_schema(self, force_refresh: bool = False) -> SchemaInfo:
        """Get schema with caching"""
        if not force_refresh and self.schema_cache:
            return self.schema_cache

        self.schema_cache = self.schema_manager.get_schema(force_refresh)
        return self.schema_cache

    def get_schema_overview(
        self,
        force_refresh: bool = False,
        include_samples: bool = True,
        max_samples_per_table: int = 3
    ) -> Dict[str, Any]:
        """Return a structured schema overview for UI or API clients."""
        schema = self._get_schema(force_refresh)
        relationships = self.schema_manager.get_table_relationships(schema)

        tables = []
        for table_name, table_info in schema.tables.items():
            columns = []
            for col in table_info.columns:
                col_entry = {
                    "name": col.name,
                    "type": col.data_type,
                    "nullable": col.is_nullable,
                    "primary_key": col.is_primary_key,
                    "foreign_key": col.foreign_key_ref if col.is_foreign_key else None
                }
                if col.description:
                    col_entry["description"] = col.description
                columns.append(col_entry)

            table_entry = {
                "name": table_name,
                "row_count": table_info.row_count,
                "description": table_info.description,
                "columns": columns,
                "relationships": relationships.get(table_name, [])
            }

            if include_samples and table_info.sample_data:
                table_entry["sample_data"] = table_info.sample_data[:max_samples_per_table]

            tables.append(table_entry)

        return {
            "database": schema.database_name,
            "db_type": self.db_type,
            "source": schema.source.value,
            "table_count": len(schema.tables),
            "tables": tables,
            "metadata": schema.metadata or {}
        }

    def _needs_clarification(self, context: QueryContext) -> bool:
        """Check if clarification is needed"""
        if not context.detected_ambiguities:
            return False

        # Only clarify high-confidence ambiguities
        high_conf = [a for a in context.detected_ambiguities if a.confidence > 0.75]
        return len(high_conf) > 0

    def _request_clarification(self, context: QueryContext) -> ExecutionResult:
        """Request user clarification for ambiguous queries"""
        clarifications = []

        for ambiguity in context.detected_ambiguities:
            if ambiguity.confidence > 0.75:
                clarifications.append({
                    "type": ambiguity.type.value,
                    "keyword": ambiguity.keyword,
                    "options": ambiguity.suggested_clarifications,
                    "confidence": ambiguity.confidence
                })

        return ExecutionResult(
            success=False,
            error="Clarification needed",
            strategy_used=ExecutionStrategy.CLARIFYING,
            context=context,
            data=clarifications  # Return clarification options in data field
        )

    def _get_sql_dialect_name(self) -> str:
        """Return a human-friendly SQL dialect name"""
        if self.db_type == "postgresql":
            return "PostgreSQL"
        if self.db_type == "mysql":
            return "MySQL"
        if self.db_type == "clickhouse":
            return "ClickHouse"
        if self.db_type == "sqlite":
            return "SQLite"
        return self.db_type

    def _normalize_clickhouse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params)
        if "user" in normalized and "username" not in normalized:
            normalized["username"] = normalized.pop("user")
        if "database" not in normalized:
            normalized["database"] = "default"
        return normalized

    def _get_connection_params(self) -> Dict[str, Any]:
        """Filter db_config to only include connection parameters."""
        if not isinstance(self.db_config, dict):
            return {}
        excluded = {
            "type",
            "execution_policy",
            "read_only",
            "allow_ddl",
            "allow_dml",
            "allow_admin",
            "allow_multi_statement",
            "default_limit",
            "enforce_default_limit",
            "schema_options",
            "schema",
            "schemas",
            "include_samples",
            "include_row_counts",
            "sample_rows",
            "row_count_strategy",
            "max_tables",
            "max_columns",
            "temperature",
            "llm_temperature",
            "max_tokens",
            "llm_max_tokens",
        }
        return {k: v for k, v in self.db_config.items() if k not in excluded}

    def _get_exploration_examples(self) -> List[str]:
        """Dialect-aware exploration examples"""
        def normalize_schema_list(raw) -> List[str]:
            if raw is None:
                return []
            if isinstance(raw, str):
                value = raw.strip()
                if not value:
                    return []
                if value in ("*", "all"):
                    return []
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(raw, (list, tuple, set)):
                return [str(item).strip() for item in raw if str(item).strip()]
            return []

        if self.db_type == "mysql":
            schemas = normalize_schema_list(self.db_config.get("schemas") or self.db_config.get("schema"))
            db_name = self.db_config.get("database", "")
            if not schemas:
                schemas = [db_name] if db_name else []
            if schemas:
                if len(schemas) == 1:
                    schema_filter = f"table_schema = '{schemas[0]}'"
                else:
                    in_list = ", ".join([f"'{s}'" for s in schemas])
                    schema_filter = f"table_schema IN ({in_list})"
            else:
                schema_filter = "table_schema = DATABASE()"
            return [
                f"SELECT table_name, table_type FROM information_schema.tables WHERE {schema_filter}",
                f"SELECT 'table_name' AS table_name, COUNT(*) AS row_count FROM table_name",
                "SELECT * FROM table_name LIMIT 5"
            ]
        if self.db_type == "clickhouse":
            databases = normalize_schema_list(self.db_config.get("schemas") or self.db_config.get("schema"))
            db_name = self.db_config.get("database", "default")
            if not databases:
                databases = [db_name]
            if len(databases) == 1:
                db_filter = f"database = '{databases[0]}'"
            else:
                in_list = ", ".join([f"'{s}'" for s in databases])
                db_filter = f"database IN ({in_list})"
            return [
                f"SELECT name, engine FROM system.tables WHERE {db_filter}",
                f"SELECT 'table_name' AS table_name, COUNT(*) AS row_count FROM {databases[0]}.table_name",
                "SELECT * FROM table_name LIMIT 5"
            ]
        if self.db_type == "sqlite":
            return [
                "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'",
                "SELECT 'table_name' AS table_name, COUNT(*) AS row_count FROM table_name",
                "SELECT * FROM table_name LIMIT 5"
            ]
        schemas = normalize_schema_list(self.db_config.get("schemas") or self.db_config.get("schema"))
        if not schemas:
            schemas = ["public"]
        if len(schemas) == 1:
            schema_filter = f"table_schema = '{schemas[0]}'"
        else:
            in_list = ", ".join([f"'{s}'" for s in schemas])
            schema_filter = f"table_schema IN ({in_list})"
        return [
            f"SELECT table_name, table_type FROM information_schema.tables WHERE {schema_filter}",
            "SELECT 'table_name' AS table_name, COUNT(*) AS row_count FROM table_name",
            "SELECT * FROM table_name LIMIT 5"
        ]

    def _split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements"""
        import sqlparse

        statements = []
        for statement in sqlparse.split(sql):
            cleaned = statement.strip()
            if cleaned:
                statements.append(cleaned)
        return statements

    def _classify_statement_type(self, sql: str) -> str:
        """Classify statement type using sqlparse tokens"""
        import sqlparse
        from sqlparse import tokens as T

        parsed = sqlparse.parse(sql)
        if not parsed:
            return "UNKNOWN"

        statement = parsed[0]
        for token in statement.flatten():
            if token.is_whitespace or token.ttype in (T.Comment.Single, T.Comment.Multiline):
                continue
            value = token.value.upper()
            if token.ttype in T.Keyword.CTE and value == "WITH":
                continue
            if token.ttype in T.Keyword.DML:
                return value
            if token.ttype in T.Keyword.DDL:
                return value
            if token.ttype in T.Keyword and value in ("SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"):
                return value
            if token.ttype in T.Keyword and value in ("SET", "USE", "BEGIN", "COMMIT", "ROLLBACK"):
                return value

        return statement.get_type().upper()

    def _statement_category(self, statement_type: str) -> str:
        """Map statement type to category"""
        if statement_type in ("SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "PRAGMA"):
            return "read"
        if statement_type in ("INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE"):
            return "dml"
        if statement_type in ("CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE", "COMMENT"):
            return "ddl"
        if statement_type in ("SET", "USE", "BEGIN", "COMMIT", "ROLLBACK"):
            return "admin"
        return "unknown"

    def _is_statement_allowed(self, category: str) -> Tuple[bool, str]:
        """Check if a statement category is allowed by policy"""
        policy = self.execution_policy

        if category == "read":
            return True, ""

        if policy.read_only:
            return False, "Read-only mode: write statements are disabled"

        if category == "dml":
            return policy.allow_dml, "DML statements are disabled"
        if category == "ddl":
            return policy.allow_ddl, "DDL statements are disabled"
        if category == "admin":
            return policy.allow_admin, "Administrative statements are disabled"

        return False, "Unrecognized statement type"

    def _apply_default_limit(self, sql: str, category: str) -> str:
        """Apply default LIMIT to read statements when missing"""
        policy = self.execution_policy
        if category != "read" or not policy.enforce_default_limit:
            return sql

        limit_value = policy.default_limit
        if not limit_value or limit_value <= 0:
            return sql

        if re.search(r"\blimit\b", sql, re.IGNORECASE):
            return sql

        statement_type = self._classify_statement_type(sql)
        if statement_type in ("SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"):
            return sql

        stripped = sql.strip().rstrip(";")
        if re.search(r"\boffset\b", stripped, re.IGNORECASE):
            return re.sub(r"\boffset\b", f"LIMIT {limit_value} OFFSET", stripped, flags=re.IGNORECASE)

        return f"{stripped} LIMIT {limit_value}"

    def _execute_with_strategy(self, context: QueryContext) -> ExecutionResult:
        """Execute query with selected strategy"""
        strategy = context.selected_strategy

        if strategy == ExecutionStrategy.DIRECT:
            return self._execute_direct(context)
        elif strategy == ExecutionStrategy.VALIDATED:
            return self._execute_validated(context)
        elif strategy == ExecutionStrategy.EXPLORATORY:
            return self._execute_exploratory(context)
        else:
            return ExecutionResult(
                success=False,
                error=f"Unknown strategy: {strategy}",
                context=context
            )

    def _attempt_fallback(self, context: QueryContext, result: ExecutionResult) -> ExecutionResult:
        """Fallback to stronger strategies when initial execution fails."""
        if result.success:
            return result
        if result.error == "Clarification needed":
            return result

        policy_blockers = (
            "Read-only mode",
            "statements are disabled",
            "Multiple SQL statements are not allowed"
        )
        if result.error and any(blocker in str(result.error) for blocker in policy_blockers):
            return result

        if context.selected_strategy == ExecutionStrategy.DIRECT:
            fallback_strategies = [ExecutionStrategy.VALIDATED, ExecutionStrategy.EXPLORATORY]
        elif context.selected_strategy == ExecutionStrategy.VALIDATED:
            fallback_strategies = [ExecutionStrategy.EXPLORATORY]
        else:
            return result

        last_result = result
        for strategy in fallback_strategies:
            context.selected_strategy = strategy
            last_result = self._execute_with_strategy(context)
            if last_result.success or last_result.error == "Clarification needed":
                return last_result

        return last_result

    def _execute_direct(self, context: QueryContext) -> ExecutionResult:
        """Direct single-attempt execution with clarification handling"""
        sql = self._generate_sql(
            context.clarified_query or context.original_query,
            context.schema_info
        )

        if not sql:
            return ExecutionResult(
                success=False,
                error="No SQL generated by LLM",
                attempts_count=1,
                strategy_used=ExecutionStrategy.DIRECT,
                context=context
            )

        # Check if LLM is requesting clarification
        if sql.startswith("CLARIFICATION_NEEDED:"):
            clarification_msg = sql.replace("CLARIFICATION_NEEDED:", "").strip()
            self.logger.info(f"Clarification requested: {clarification_msg}")

            # Return clarification request as a special result
            return ExecutionResult(
                success=False,
                error="Clarification needed",
                strategy_used=ExecutionStrategy.CLARIFYING,
                context=context,
                data=[{"type": "clarification", "message": clarification_msg}]
            )

        success, result = self._execute_sql(sql)
        context.attempts.append({
            "strategy": "validated",
            "sql": sql,
            "success": success,
            "result": result if success else None,
            "error": result if not success else None
        })

        context.attempts.append({
            "strategy": "direct",
            "sql": sql,
            "success": success,
            "result": result
        })

        if success:
            return ExecutionResult(
                success=True,
                sql=sql,
                data=result.get("data"),
                columns=result.get("columns"),
                row_count=result.get("row_count", 0),
                attempts_count=1,
                strategy_used=ExecutionStrategy.DIRECT,
                confidence=0.8,
                context=context,
                results=result.get("results"),
                affected_rows=result.get("affected_rows")
            )
        else:
            return ExecutionResult(
                success=False,
                sql=sql,
                error=result,
                attempts_count=1,
                strategy_used=ExecutionStrategy.DIRECT,
                context=context
            )

    def _execute_validated(self, context: QueryContext) -> ExecutionResult:
        """Execute with validation queries first"""
        # First run validation queries
        validations = self._generate_validation_queries(
            context.clarified_query or context.original_query,
            context.schema_info
        )

        context.validation_results = []
        for val_query in validations:
            success, result = self._execute_sql(val_query["sql"])
            context.validation_results.append({
                "purpose": val_query["purpose"],
                "success": success,
                "result": result
            })

        # Generate main query based on validations
        sql = self._generate_sql_with_validation(
            context.clarified_query or context.original_query,
            context.schema_info,
            context.validation_results
        )

        if not sql:
            return ExecutionResult(
                success=False,
                error="No SQL generated by LLM",
                attempts_count=len(validations) + 1,
                strategy_used=ExecutionStrategy.VALIDATED,
                context=context
            )

        success, result = self._execute_sql(sql)

        if success:
            return ExecutionResult(
                success=True,
                sql=sql,
                data=result.get("data"),
                columns=result.get("columns"),
                row_count=result.get("row_count", 0),
                attempts_count=len(validations) + 1,
                strategy_used=ExecutionStrategy.VALIDATED,
                confidence=0.85,
                context=context,
                results=result.get("results"),
                affected_rows=result.get("affected_rows")
            )
        else:
            return ExecutionResult(
                success=False,
                sql=sql,
                error=result,
                attempts_count=len(validations) + 1,
                strategy_used=ExecutionStrategy.VALIDATED,
                context=context
            )

    def _execute_exploratory(self, context: QueryContext) -> ExecutionResult:
        """Execute with multiple attempts and learning"""
        best_result = None
        best_confidence = 0

        for attempt_num in range(self.max_attempts):
            # Generate SQL based on previous attempts
            sql = self._generate_sql_with_learning(
                context.clarified_query or context.original_query,
                context.schema_info,
                context.attempts
            )

            if not sql:
                context.attempts.append({
                    "attempt": attempt_num + 1,
                    "sql": "",
                    "success": False,
                    "result": None,
                    "error": "No SQL generated by LLM"
                })
                continue

            success, result = self._execute_sql(sql)

            context.attempts.append({
                "attempt": attempt_num + 1,
                "sql": sql,
                "success": success,
                "result": result if success else None,
                "error": result if not success else None
            })

            if success:
                # Evaluate result quality
                confidence = self._evaluate_result_quality(
                    context.original_query,
                    sql,
                    result
                )

                if confidence > best_confidence:
                    best_result = ExecutionResult(
                        success=True,
                        sql=sql,
                        data=result.get("data"),
                        columns=result.get("columns"),
                        row_count=result.get("row_count", 0),
                        attempts_count=attempt_num + 1,
                        strategy_used=ExecutionStrategy.EXPLORATORY,
                        confidence=confidence,
                        context=context,
                        results=result.get("results"),
                        affected_rows=result.get("affected_rows")
                    )
                    best_confidence = confidence

                # Stop if high confidence achieved or early success for simple queries
                has_failure = any(not a.get("success") for a in context.attempts)
                difficulty = context.difficulty or QueryDifficulty.MODERATE
                if confidence >= 0.9:
                    break
                if not has_failure and difficulty in (QueryDifficulty.SIMPLE, QueryDifficulty.MODERATE):
                    if confidence >= 0.85:
                        break

        return best_result or ExecutionResult(
            success=False,
            error=self._get_last_attempt_error(context) or "All attempts failed",
            attempts_count=len(context.attempts),
            strategy_used=ExecutionStrategy.EXPLORATORY,
            context=context
        )

    def _get_last_attempt_error(self, context: QueryContext) -> Optional[str]:
        """Return the most recent error from attempts for better diagnostics."""
        if not context or not context.attempts:
            return None
        for attempt in reversed(context.attempts):
            error = attempt.get("error")
            if error:
                return str(error)
        return None

    def _generate_sql(self, query: str, schema: SchemaInfo) -> str:
        """Generate SQL from natural language using LLM"""
        import requests
        import json

        # Format schema information for the LLM
        schema_description = self._format_schema_for_llm(schema)

        dialect = self._get_sql_dialect_name()
        exploration_examples = self._get_exploration_examples()

        # Create prompt for LLM with enhanced clarification and exploration guidance
        prompt = f"""You are an intelligent SQL assistant. Your role is to understand user intent and generate appropriate SQL queries or request clarification when needed.

DATABASE SCHEMA:
{schema_description}

SQL DIALECT:
{dialect}

EXECUTION POLICY:
- read_only: {self.execution_policy.read_only}
- allow_dml: {self.execution_policy.allow_dml}
- allow_ddl: {self.execution_policy.allow_ddl}
- allow_admin: {self.execution_policy.allow_admin}
- default_limit: {self.execution_policy.default_limit}

USER REQUEST: {query}

CRITICAL INSTRUCTIONS:

1. FIRST ASSESS THE REQUEST:
   - Check if referenced tables/columns exist in the schema
   - Identify any ambiguous terms or unclear references
   - Detect if the user wants to explore/understand the database

2. CLARIFICATION DETECTION (Respond with "NEEDS_CLARIFICATION:" prefix):
   - If tables/columns mentioned don't exist, suggest alternatives
   - If the request is vague (e.g., "show data"), ask what specific data
   - If multiple interpretations exist, list options
   Example: "NEEDS_CLARIFICATION: No 'customers' table found. Did you mean 'users'? Available tables: users, orders, products"

3. EXPLORATION DETECTION (Generate multiple queries if needed):
   Detect exploration intent from phrases like:
   - "explore", "what's in the database", "show me the structure"
   - "overview", "what data", "what tables", "describe"
   - General curiosity about database contents

   For exploration, generate:
   a) Schema overview: {exploration_examples[0]};
   b) Table row counts: {exploration_examples[1]};
   c) Sample data for key tables: {exploration_examples[2]};
   d) Key relationships and constraints

4. SPECIFIC QUERIES (Single optimized SQL):
   - Use appropriate JOINs for multi-table queries
   - Add LIMIT 10-20 unless aggregating
   - Include ORDER BY for meaningful results
   - Add comments explaining complex logic

5. INTELLIGENT INTERPRETATION:
   - Understand common aliases (customers→users, products→items)
   - Infer time ranges ("recent" → last 30 days)
   - Handle typos and variations gracefully
   - Default to showing overview if request is too vague

RESPONSE FORMAT:
- If clarification needed: "NEEDS_CLARIFICATION: [question and suggestions]"
- For exploration: Multiple queries separated by semicolons with comments
- For specific queries: Single SQL with explanatory comments

Generate your response:
"""

        try:
            # Call LLM to generate SQL
            response = self._call_llm(prompt)

            # Check if LLM is requesting clarification
            if response.strip().startswith("NEEDS_CLARIFICATION:"):
                # Extract the clarification message
                clarification_msg = response.replace("NEEDS_CLARIFICATION:", "").strip()
                self.logger.info(f"LLM requesting clarification: {clarification_msg}")
                # Return a special marker that can be detected by the caller
                return f"CLARIFICATION_NEEDED: {clarification_msg}"

            # Clean up the SQL using the enhanced cleaning method
            sql = self._clean_sql_response(response)

            if not sql:
                raise Exception("LLM did not return valid SQL")

            # Add comment for tracking if not already present
            if "--" not in sql and not sql.startswith("CLARIFICATION_NEEDED:"):
                sql += f" -- Generated for: {query[:50]}"

            return sql

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            return ""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API using configured provider"""
        from ..config.llm_config import get_llm_config

        try:
            llm_config = get_llm_config()
            provider_override = None
            model_override = None
            if self.model_name in llm_config.PROVIDERS:
                provider_override = self.model_name
            elif self.model_name:
                model_override = self.model_name
            return llm_config.call_llm(
                prompt,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                model=model_override,
                provider=provider_override
            )
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM service error: {str(e)}")

    def _format_schema_for_llm(self, schema: SchemaInfo) -> str:
        """Format comprehensive schema information for LLM prompt"""
        if not schema or not schema.tables:
            return "No schema information available"

        lines = [f"Database: {schema.database_name}" if schema.database_name else ""]
        lines.append(f"Total Tables: {len(schema.tables)}")

        # Add quick statistics summary
        total_rows = 0
        tables_with_data = []

        for table_name, table_info in schema.tables.items():
            if hasattr(table_info, 'row_count') and table_info.row_count is not None:
                total_rows += table_info.row_count
                tables_with_data.append((table_name, table_info.row_count))

        if tables_with_data:
            lines.append(f"Total Rows Across All Tables: {total_rows}")
            lines.append("Tables Overview (sorted by importance):")
            for table_name, count in sorted(tables_with_data, key=lambda x: x[1], reverse=True):
                lines.append(f"  - {table_name}: {count} rows")

        lines.append("\nDetailed Schema:\n")

        for table_name, table_info in schema.tables.items():
            lines.append(f"Table: {table_name}")

            # Add row count if available
            if hasattr(table_info, 'row_count') and table_info.row_count is not None:
                lines.append(f"  Row Count: {table_info.row_count}")

            # Add columns with more detail
            if table_info.columns:
                lines.append("  Columns:")
                for col in table_info.columns:
                    col_desc = f"    - {col.name} ({col.data_type})"
                    if hasattr(col, 'is_primary_key') and col.is_primary_key:
                        col_desc += " [PRIMARY KEY]"
                    if hasattr(col, 'is_nullable') and not col.is_nullable:
                        col_desc += " [NOT NULL]"
                    if getattr(col, 'is_foreign_key', False) and getattr(col, 'foreign_key_ref', None):
                        col_desc += f" [FK -> {col.foreign_key_ref}]"
                    lines.append(col_desc)

            # Add indexes if available
            if hasattr(table_info, 'indexes') and table_info.indexes:
                lines.append("  Indexes:")
                for idx in table_info.indexes:
                    lines.append(f"    - {idx}")

            # Add relationships
            if table_info.relationships:
                lines.append("  Relationships:")
                for rel in table_info.relationships:
                    lines.append(f"    - {rel}")

            # Add sample data if available
            if hasattr(table_info, 'sample_data') and table_info.sample_data:
                lines.append("  Sample Data Preview:")
                for row in table_info.sample_data[:2]:  # Show first 2 rows
                    lines.append(f"    {row}")

            lines.append("")  # Empty line between tables

        return "\n".join(lines)

    def _guess_main_table(self, schema: SchemaInfo) -> str:
        """Guess the main table name from schema"""
        if schema and schema.tables:
            # Prefer common table names
            for name in ["users", "customers", "orders", "products"]:
                for table in schema.tables:
                    if table == name or table.endswith(f".{name}"):
                        return table
            # Return the first table
            return list(schema.tables.keys())[0]
        return "users"  # default

    def _generate_validation_queries(self, query: str, schema: SchemaInfo) -> List[Dict]:
        """Generate validation queries using LLM to understand the query intent"""
        schema_description = self._format_schema_for_llm(schema)

        dialect = self._get_sql_dialect_name()
        prompt = f"""You are a SQL expert. Analyze this user request and generate validation queries to verify data availability.

Database Schema:
{schema_description}

SQL Dialect: {dialect}

User Request: {query}

Generate 2-3 validation queries to:
1. Check if required tables exist and have data
2. Verify key columns are available
3. Test any joins or relationships needed

Return ONLY a JSON array of validation queries in this format:
[
  {{"purpose": "description", "sql": "SELECT query"}},
  {{"purpose": "description", "sql": "SELECT query"}}
]

JSON Array:"""

        try:
            response = self._call_llm(prompt)
            # Parse JSON response
            import json
            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            validation_queries = json.loads(response)
            return validation_queries

        except Exception as e:
            self.logger.warning(f"Failed to generate validation queries via LLM: {str(e)}")
            # Fallback to basic validation
            main_table = self._guess_main_table(schema)
            return [
                {"purpose": "Check table exists", "sql": f"SELECT COUNT(*) FROM {main_table} LIMIT 1"},
                {"purpose": "Check columns", "sql": f"SELECT * FROM {main_table} LIMIT 1"}
            ]

    def _generate_sql_with_validation(self, query: str, schema: SchemaInfo,
                                     validations: List[Dict]) -> str:
        """Generate SQL using validation results to inform the query"""
        schema_description = self._format_schema_for_llm(schema)

        # Format validation results
        validation_info = "\nValidation Results:\n"
        for val in validations:
            status = "Success" if val.get("success") else f"Failed: {val.get('result')}"
            validation_info += f"- {val['purpose']}: {status}\n"

        dialect = self._get_sql_dialect_name()
        prompt = f"""You are a SQL expert. Generate a {dialect} query using the validation results.

Database Schema:
{schema_description}

{validation_info}

Execution Policy:
- read_only: {self.execution_policy.read_only}
- allow_dml: {self.execution_policy.allow_dml}
- allow_ddl: {self.execution_policy.allow_ddl}
- allow_admin: {self.execution_policy.allow_admin}

User Request: {query}

Requirements:
1. Return ONLY the SQL query - no explanations, no natural language
2. Start directly with SELECT/INSERT/UPDATE/DELETE/etc
3. Use the validation results to ensure the query will work
4. Include appropriate JOINs, GROUP BY, and LIMIT clauses as needed
5. DO NOT include any text before or after the SQL

SQL Query (no explanations):"""

        try:
            sql = self._call_llm(prompt)
            sql = self._clean_sql_response(sql)
            return sql + f" -- Validated for: {query[:50]}"
        except Exception as e:
            self.logger.error(f"LLM validation query failed: {str(e)}")
            return self._generate_sql(query, schema)

    def _generate_sql_with_learning(self, query: str, schema: SchemaInfo,
                                   attempts: List[Dict]) -> str:
        """Generate SQL learning from previous attempts and errors"""
        schema_description = self._format_schema_for_llm(schema)

        # Format previous attempts and errors
        attempts_info = "\nPrevious Attempts and Errors:\n"
        for i, attempt in enumerate(attempts, 1):
            sql = attempt.get('sql', 'Unknown')
            error = attempt.get('error')
            error_text = str(error) if error is not None else "None"
            attempts_info += f"Attempt {i}:\n"
            attempts_info += f"  SQL: {str(sql)[:200]}\n"
            attempts_info += f"  Error: {error_text[:200]}\n"

        dialect = self._get_sql_dialect_name()
        prompt = f"""You are a SQL expert. Previous SQL attempts have failed. Learn from the errors and generate a correct {dialect} query.

Database Schema:
{schema_description}

{attempts_info}

Execution Policy:
- read_only: {self.execution_policy.read_only}
- allow_dml: {self.execution_policy.allow_dml}
- allow_ddl: {self.execution_policy.allow_ddl}
- allow_admin: {self.execution_policy.allow_admin}

User Request: {query}

Requirements:
1. Analyze the previous errors carefully
2. Generate a corrected SQL query that avoids these errors
3. Ensure table and column names are correct
4. Return ONLY the SQL query - no explanations, no natural language
5. Start directly with SELECT/INSERT/UPDATE/DELETE/etc
6. DO NOT include any text before or after the SQL

Corrected SQL Query (no explanations):"""

        try:
            sql = self._call_llm(prompt)
            sql = self._clean_sql_response(sql)
            return sql + f" -- Attempt {len(attempts) + 1} for: {query[:50]}"
        except Exception as e:
            self.logger.error(f"LLM learning query failed: {str(e)}")
            # Fallback to basic query
            main_table = self._guess_main_table(schema)
            return f"SELECT * FROM {main_table} LIMIT 10 -- Fallback attempt {len(attempts) + 1}"

    def _clean_sql_response(self, sql: str) -> str:
        """Clean up SQL response from LLM"""
        original_sql = sql
        sql = sql.strip()

        # Remove markdown code blocks (handle multi-line blocks)
        # Remove opening markdown
        if "```sql" in sql:
            parts = sql.split("```sql", 1)
            if len(parts) > 1:
                sql = parts[1]
        elif "```" in sql:
            parts = sql.split("```", 1)
            if len(parts) > 1:
                sql = parts[1]

        # Remove closing markdown
        if "```" in sql:
            sql = sql.split("```")[0]

        sql = sql.strip()

        # Remove any leading natural language explanations
        # If the response contains natural language before SQL, extract just the SQL
        lines = sql.split('\n')
        sql_start_idx = -1

        # Find where SQL actually starts (common SQL keywords)
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
                       'WITH', 'EXPLAIN', 'SHOW', 'DESCRIBE']

        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            if any(line_upper.startswith(keyword) for keyword in sql_keywords):
                sql_start_idx = i
                break

        # If we found SQL, extract from that point
        if sql_start_idx >= 0:
            sql = '\n'.join(lines[sql_start_idx:])
        # If no SQL keywords found but response has "SQL:" or similar markers
        elif 'SQL:' in sql or 'Query:' in sql or 'query:' in sql:
            # Try to extract after these markers
            for marker in ['SQL:', 'Query:', 'query:', 'sql:']:
                if marker in sql:
                    sql = sql.split(marker, 1)[1]
                    break

        # Final cleanup
        sql = sql.strip()

        # Check if we still don't have valid SQL
        if sql and not any(sql.upper().startswith(kw) for kw in sql_keywords):
            self.logger.warning(f"LLM returned natural language instead of SQL: {original_sql[:200]}...")
            return ""

        return sql

    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """Execute SQL on database with policy checks and multi-statement support"""
        statements = self._split_sql_statements(sql)
        if not statements:
            return False, "No SQL statement found"

        if len(statements) > 1 and not self.execution_policy.allow_multi_statement:
            return False, "Multiple SQL statements are not allowed"

        prepared = []
        for statement in statements:
            statement_type = self._classify_statement_type(statement)
            category = self._statement_category(statement_type)
            allowed, reason = self._is_statement_allowed(category)
            if not allowed:
                return False, reason

            adjusted_sql = self._apply_default_limit(statement, category)
            prepared.append({
                "sql": adjusted_sql,
                "category": category,
                "statement_type": statement_type
            })

        return self._execute_prepared_statements(prepared)

    def _execute_prepared_statements(self, prepared: List[Dict[str, Any]]) -> Tuple[bool, Any]:
        """Execute prepared statements with a shared connection"""
        import psycopg2
        import pymysql
        import clickhouse_connect

        conn = None
        cursor = None
        client = None
        try:
            if self.db_type == 'postgresql':
                conn_params = self._get_connection_params()
                conn = psycopg2.connect(**conn_params)
                cursor = conn.cursor()
                results = []
                total_rows = 0
                affected_rows = 0
                has_write = False

                for statement in prepared:
                    is_read = statement["category"] == "read"
                    cursor.execute(statement["sql"])
                    if is_read:
                        data = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        row_count = len(data)
                        total_rows += row_count
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "data": data,
                            "columns": columns,
                            "row_count": row_count
                        })
                    else:
                        has_write = True
                        row_count = cursor.rowcount if cursor.rowcount is not None else 0
                        affected_rows += max(row_count, 0)
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "row_count": row_count
                        })

                if has_write:
                    conn.commit()
                    self.logger.info("Transaction committed successfully")

                if len(results) == 1:
                    if results[0]["category"] == "read":
                        return True, {
                            "data": results[0]["data"],
                            "columns": results[0]["columns"],
                            "row_count": results[0]["row_count"]
                        }
                    return True, {
                        "data": [],
                        "columns": [],
                        "row_count": 0,
                        "affected_rows": affected_rows
                    }

                return True, {
                    "results": results,
                    "row_count": total_rows,
                    "affected_rows": affected_rows
                }

            if self.db_type == 'mysql':
                conn_params = self._get_connection_params()
                conn = pymysql.connect(**conn_params)
                cursor = conn.cursor()
                results = []
                total_rows = 0
                affected_rows = 0
                has_write = False

                for statement in prepared:
                    is_read = statement["category"] == "read"
                    cursor.execute(statement["sql"])
                    if is_read:
                        data = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        row_count = len(data)
                        total_rows += row_count
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "data": data,
                            "columns": columns,
                            "row_count": row_count
                        })
                    else:
                        has_write = True
                        row_count = cursor.rowcount if cursor.rowcount is not None else 0
                        affected_rows += max(row_count, 0)
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "row_count": row_count
                        })

                if has_write:
                    conn.commit()
                    self.logger.info("Transaction committed successfully")

                if len(results) == 1:
                    if results[0]["category"] == "read":
                        return True, {
                            "data": results[0]["data"],
                            "columns": results[0]["columns"],
                            "row_count": results[0]["row_count"]
                        }
                    return True, {
                        "data": [],
                        "columns": [],
                        "row_count": 0,
                        "affected_rows": affected_rows
                    }

                return True, {
                    "results": results,
                    "row_count": total_rows,
                    "affected_rows": affected_rows
                }

            if self.db_type == 'clickhouse':
                conn_params = self._get_connection_params()
                conn_params = self._normalize_clickhouse_params(conn_params)
                client = clickhouse_connect.get_client(**conn_params)
                results = []
                total_rows = 0

                for statement in prepared:
                    is_read = statement["category"] == "read"
                    if is_read:
                        query_result = client.query(statement["sql"])
                        data = query_result.result_rows
                        columns = query_result.column_names
                        row_count = len(data)
                        total_rows += row_count
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "data": data,
                            "columns": columns,
                            "row_count": row_count
                        })
                    else:
                        client.command(statement["sql"])
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "row_count": 0
                        })

                client.close()

                if len(results) == 1:
                    if results[0]["category"] == "read":
                        return True, {
                            "data": results[0]["data"],
                            "columns": results[0]["columns"],
                            "row_count": results[0]["row_count"]
                        }
                    return True, {
                        "data": [],
                        "columns": [],
                        "row_count": 0
                    }

                return True, {
                    "results": results,
                    "row_count": total_rows
                }

            if self.db_type == 'sqlite':
                import sqlite3
                db_path = self.db_config.get("database") or self.db_config.get("path")
                if not db_path:
                    return False, "SQLite database path not provided"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                results = []
                total_rows = 0
                affected_rows = 0
                has_write = False

                for statement in prepared:
                    is_read = statement["category"] == "read"
                    cursor.execute(statement["sql"])
                    if is_read:
                        data = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        row_count = len(data)
                        total_rows += row_count
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "data": data,
                            "columns": columns,
                            "row_count": row_count
                        })
                    else:
                        has_write = True
                        row_count = cursor.rowcount if cursor.rowcount is not None else 0
                        affected_rows += max(row_count, 0)
                        results.append({
                            "sql": statement["sql"],
                            "category": statement["category"],
                            "row_count": row_count
                        })

                if has_write:
                    conn.commit()

                if len(results) == 1:
                    if results[0]["category"] == "read":
                        return True, {
                            "data": results[0]["data"],
                            "columns": results[0]["columns"],
                            "row_count": results[0]["row_count"]
                        }
                    return True, {
                        "data": [],
                        "columns": [],
                        "row_count": 0,
                        "affected_rows": affected_rows
                    }

                return True, {
                    "results": results,
                    "row_count": total_rows,
                    "affected_rows": affected_rows
                }

            return False, f"Unsupported database type: {self.db_type}"

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                    self.logger.warning("Transaction rolled back due to error")
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")

            self.logger.error(f"SQL execution failed: {str(e)}")
            error_msg = self._sanitize_error_message(str(e))
            return False, error_msg
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error messages to remove sensitive information"""
        import re

        # Remove potential passwords (anything that looks like a password parameter)
        error_msg = re.sub(r'password["\']?\s*[:=]\s*["\']?[^"\'\s,)]+', 'password=***', error_msg, flags=re.IGNORECASE)

        # Remove connection strings that might contain credentials
        error_msg = re.sub(r'postgresql://[^@]+@', 'postgresql://***@', error_msg)
        error_msg = re.sub(r'mysql://[^@]+@', 'mysql://***@', error_msg)

        # Remove IP addresses that might be internal
        error_msg = re.sub(r'\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<internal-ip>', error_msg)

        # Common database error patterns that are safe to show
        safe_patterns = [
            r'syntax error',
            r'column .* does not exist',
            r'table .* does not exist',
            r'relation .* does not exist',
            r'permission denied',
            r'duplicate key value',
            r'foreign key constraint',
            r'not null constraint',
            r'invalid input syntax',
            r'division by zero',
            r'no such table',
            r'no such column',
            r'no such function',
            r'ambiguous column name',
            r'misuse of aggregate',
            r'unable to open database file',
        ]

        # Check if error contains any safe pattern
        for pattern in safe_patterns:
            if re.search(pattern, error_msg, re.IGNORECASE):
                # Extract just the relevant part
                match = re.search(f'({pattern}[^.]*)', error_msg, re.IGNORECASE)
                if match:
                    return match.group(1)

        # For other errors, return a generic message
        if 'connection' in error_msg.lower():
            return "Database connection error"
        elif 'timeout' in error_msg.lower():
            return "Query timeout"
        elif 'permission' in error_msg.lower():
            return "Permission denied"
        else:
            # Generic error message that doesn't reveal internal details
            return "Query execution failed. Please check your query syntax."

    def _evaluate_result_quality(self, query: str, sql: str, result: Dict) -> float:
        """Evaluate quality of SQL result"""
        query_lower = (query or "").lower()
        sql_lower = (sql or "").lower()

        def has_any(tokens: List[str], text: str) -> bool:
            return any(token in text for token in tokens)

        row_count = result.get("row_count")
        if row_count is None:
            if result.get("data") is not None:
                try:
                    row_count = len(result.get("data"))
                except Exception:
                    row_count = None
            elif result.get("results"):
                try:
                    row_count = sum(int(item.get("row_count", 0)) for item in result.get("results", []))
                except Exception:
                    row_count = None

        score = 0.45
        if result.get("data") is not None or result.get("results") is not None:
            score += 0.15
            if row_count == 0:
                score -= 0.15
            elif row_count == 1:
                score += 0.15
            elif row_count and row_count <= 20:
                score += 0.12
            elif row_count and row_count > 200:
                score -= 0.05
        elif result.get("affected_rows") is not None:
            score += 0.2

        agg_intent = has_any(
            ["count", "how many", "number of", "total", "sum", "avg", "average", "max", "min"],
            query_lower
        )
        if agg_intent:
            if re.search(r"\b(count|sum|avg|average|max|min)\b", sql_lower):
                score += 0.2
            else:
                score -= 0.1

        distinct_intent = has_any(["distinct", "different", "unique"], query_lower)
        if distinct_intent:
            if "distinct" in sql_lower:
                score += 0.1
            else:
                score -= 0.05

        group_intent = has_any(["per ", "by ", "group by", "each "], query_lower)
        if group_intent:
            if "group by" in sql_lower:
                score += 0.1
            else:
                score -= 0.05

        order_intent = has_any(["top", "highest", "lowest", "most", "least", "largest", "smallest"], query_lower)
        if order_intent:
            if "order by" in sql_lower:
                score += 0.08
            else:
                score -= 0.04

        if has_any(["limit", "first", "latest", "recent"], query_lower) and "limit" in sql_lower:
            score += 0.03

        if has_any(["where", "after", "before", "between", "greater", "less", "above", "below"], query_lower) and "where" in sql_lower:
            score += 0.05

        score = max(0.0, min(0.99, score))
        return score

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "cache_size": len(self.query_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "ambiguity_detection_stats": self.ambiguity_detector.get_risk_assessment(),
            "schema_tables": len(self.schema_cache.tables) if self.schema_cache else 0
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Would track hits vs misses
        return 0.0
