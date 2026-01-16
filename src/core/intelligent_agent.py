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
    output_plan: Optional[Dict[str, Any]] = None
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
        self.plan_cache = {}
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

    def _detect_exploration_intent(self, query: str) -> bool:
        """Return True only for explicit schema/database exploration requests."""
        if not query:
            return False
        q = query.strip().lower()
        if re.search(r"\b(schema|schemas)\b", q):
            return True
        patterns = [
            r"\bshow\s+tables\b",
            r"\blist\s+tables\b",
            r"\bwhat\s+tables\b",
            r"\btable\s+list\b",
            r"\bdescribe\s+(database|schema|tables?)\b",
            r"\b(database|schema)\s+(overview|structure)\b",
            r"\boverview\s+of\s+the\s+(database|schema)\b",
            r"\bwhat's\s+in\s+the\s+database\b",
            r"\bdata\s+dictionary\b",
            r"\bentity\s+relationship\b",
            r"\berd\b",
            r"\bshow\s+columns\b",
            r"\blist\s+columns\b",
            r"\bcolumn\s+list\b",
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    def _looks_like_exploration_sql(self, sql: str) -> bool:
        """Detect exploration SQL to avoid schema queries on normal questions."""
        if not sql:
            return False
        lower = sql.lower()
        keywords = [
            "information_schema",
            "sqlite_master",
            "pg_catalog",
            "system.tables",
            "system.columns",
            "show tables",
            "show databases",
            "describe ",
            "pragma ",
        ]
        if any(keyword in lower for keyword in keywords):
            return True
        statements = self._split_sql_statements(sql)
        return len(statements) > 1

    def _build_logic_hints(self, query: str) -> str:
        """Add dynamic logic hints based on query phrasing."""
        hints = []
        q = (query or "").lower()
        if re.search(r"\b(both|each|every|all of)\b", q):
            hints.append(
                "Query implies ALL of multiple values; use INTERSECT or GROUP BY HAVING "
                "COUNT(DISTINCT <field>) = N (do not use IN alone)."
            )
        if re.search(r"\b(youngest|oldest|highest|lowest|largest|smallest|most|least|top|bottom)\b", q):
            hints.append("Superlatives require ORDER BY and LIMIT.")
        if re.search(r"\b(average|avg|mean)\b", q):
            hints.append(
                "If schema has a column named 'average'/'avg', prefer the column unless the "
                "request explicitly asks to compute an aggregate of another column."
            )
        if re.search(r"\b(distinct|unique|different)\b", q):
            hints.append("Use DISTINCT to return unique values.")
        else:
            hints.append(
                "Use DISTINCT when listing categorical values to improve readability "
                "(e.g., countries), but avoid DISTINCT in aggregates unless requested."
            )
        if not hints:
            return "- None"
        return "\n".join(f"- {hint}" for hint in hints)

    def _normalize_token(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _tokenize_text(self, text: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]

    def _collect_schema_columns(self, schema: SchemaInfo) -> List[Dict[str, Any]]:
        columns = []
        if not schema or not schema.tables:
            return columns
        for table_name, table_info in schema.tables.items():
            if not table_info or not getattr(table_info, "columns", None):
                continue
            for col in table_info.columns:
                col_name = col.name
                tokens = self._tokenize_text(col_name)
                columns.append({
                    "table": table_name,
                    "name": col_name,
                    "tokens": tokens,
                    "normalized": self._normalize_token(col_name)
                })
        return columns

    def _pluralize_token(self, token: str) -> str:
        if token.endswith("y") and len(token) > 1:
            return token[:-1] + "ies"
        if token.endswith("s"):
            return token + "es"
        return token + "s"

    def _find_best_column_for_tokens(self, columns: List[Dict[str, Any]], tokens: List[str]) -> Optional[str]:
        """Enhanced column matching with synonyms and fuzzy matching."""
        wanted = [self._normalize_token(t) for t in tokens if t]
        if not wanted:
            return None

        # Synonym mapping for common column name variations
        synonyms = {
            "name": ["title", "label", "description"],
            "year": ["release_year", "yr", "anno"],
            "count": ["number", "quantity", "total", "sum"],
            "age": ["years_old"],
            "id": ["identifier", "key", "code"],
            "date": ["time", "datetime", "timestamp"],
            "price": ["cost", "amount", "value"],
            "status": ["state", "condition"],
        }

        # Expand tokens with synonyms
        expanded_tokens = set(wanted)
        for token in wanted:
            if token in synonyms:
                expanded_tokens.update([self._normalize_token(syn) for syn in synonyms[token]])
            # Check reverse mapping
            for key, syns in synonyms.items():
                if token in [self._normalize_token(s) for s in syns]:
                    expanded_tokens.add(key)

        # Phase 1: Exact match
        candidates = []
        for col in columns:
            if all(token in col["normalized"] for token in wanted):
                candidates.append((col, 1.0))  # Perfect match

        # Phase 2: Synonym match
        if not candidates:
            for col in columns:
                match_count = sum(1 for token in expanded_tokens if token in col["normalized"])
                if match_count >= len(wanted) * 0.7:  # At least 70% match
                    candidates.append((col, 0.8))

        # Phase 3: Partial match
        if not candidates:
            for col in columns:
                for token in wanted:
                    if token in col["normalized"] or col["normalized"] in token:
                        candidates.append((col, 0.6))
                        break

        # Phase 4: Edit distance matching (fuzzy match)
        if not candidates:
            try:
                import Levenshtein
                for col in columns:
                    min_distance = float('inf')
                    for token in wanted:
                        distance = Levenshtein.distance(token, col["normalized"])
                        min_distance = min(min_distance, distance)

                    # Dynamic threshold based on token length
                    threshold = min(3, len(col["normalized"]) // 3)
                    if min_distance <= threshold:
                        score = 1 - (min_distance / max(len(col["normalized"]), 1))
                        candidates.append((col, score * 0.5))
            except ImportError:
                # Fallback to simple substring matching if Levenshtein not available
                pass

        if not candidates:
            return None

        # Sort by score and then by name length (prefer shorter names)
        candidates.sort(key=lambda x: (-x[1], len(x[0]["normalized"])))
        return candidates[0][0]["name"]

    def _apply_output_plan_heuristics(self, plan: Dict[str, Any], query: str, schema: SchemaInfo) -> Dict[str, Any]:
        """Refine LLM output plan using schema-aware heuristics."""
        plan = plan if isinstance(plan, dict) else {}
        required = [str(x) for x in plan.get("required_columns") or [] if str(x).strip()]
        optional = [str(x) for x in plan.get("optional_columns") or [] if str(x).strip()]
        metrics = [str(x) for x in plan.get("metrics") or [] if str(x).strip()]
        group_by = [str(x) for x in plan.get("group_by") or [] if str(x).strip()]
        order_by = [str(x) for x in plan.get("order_by") or [] if str(x).strip()]
        notes = str(plan.get("notes") or "")
        limit = plan.get("limit")

        q = (query or "").lower()
        columns = self._collect_schema_columns(schema)

        def add_unique(items: List[str], value: Optional[str]) -> None:
            if value and value not in items:
                items.append(value)

        def remove_if(items: List[str], predicate) -> List[str]:
            return [item for item in items if not predicate(item)]

        def norm(item: str) -> str:
            return self._normalize_token(item)

        # Target-specific columns (song name/release year)
        if "song" in q and ("name" in q or "title" in q):
            song_name_col = self._find_best_column_for_tokens(columns, ["song", "name"])
            if not song_name_col:
                song_name_col = self._find_best_column_for_tokens(columns, ["song", "title"])
            add_unique(required, song_name_col)
            if song_name_col and not re.search(r"\b(singer|artist)\s+name\b", q):
                required = remove_if(required, lambda c: norm(c).endswith("name") and "song" not in norm(c) and "title" not in norm(c))

        if "release year" in q or ("song" in q and "year" in q):
            song_year_col = self._find_best_column_for_tokens(columns, ["song", "release", "year"])
            if not song_year_col:
                song_year_col = self._find_best_column_for_tokens(columns, ["release", "year"])
            if not song_year_col:
                song_year_col = self._find_best_column_for_tokens(columns, ["song", "year"])
            add_unique(required, song_year_col)

        # Aggregate heuristics
        agg_groups = {
            "AVG": ["average", "avg", "mean"],
            "MAX": ["maximum", "max", "highest", "largest"],
            "MIN": ["minimum", "min", "lowest", "smallest"],
            "SUM": ["sum", "total"],
            "COUNT": ["count", "number of", "how many"],
        }
        agg_tokens = set()
        for words in agg_groups.values():
            agg_tokens.update(words)

        token_to_cols: Dict[str, List[Dict[str, Any]]] = {}
        for col in columns:
            for token in col["tokens"]:
                if len(token) < 3:
                    continue
                token_to_cols.setdefault(token, []).append(col)

        explicit_for_agg = {key: False for key in agg_groups.keys()}

        def has_explicit_agg(agg_words: List[str], token: str) -> bool:
            plural = self._pluralize_token(token)
            for word in agg_words:
                if re.search(rf"\\b{re.escape(word)}\\b", q):
                    if re.search(rf"\\b{re.escape(word)}\\b\\s+(of\\s+)?{re.escape(token)}\\b", q):
                        return True
                    if re.search(rf"\\b{re.escape(word)}\\b\\s+(of\\s+)?{re.escape(plural)}\\b", q):
                        return True
                    if re.search(rf"\\b{re.escape(token)}\\b\\s+{re.escape(word)}\\b", q):
                        return True
            return False

        for token, cols in token_to_cols.items():
            if token in agg_tokens:
                continue
            if token not in q:
                continue
            col_name = self._find_best_column_for_tokens(columns, [token])
            if not col_name:
                continue
            for agg_name, agg_words in agg_groups.items():
                if has_explicit_agg(agg_words, token):
                    add_unique(metrics, f"{agg_name}({col_name})")
                    explicit_for_agg[agg_name] = True

        # If aggregate word used without explicit target, prefer matching column names
        def add_column_if_named(word: str) -> Optional[str]:
            return self._find_best_column_for_tokens(columns, [word])

        if any(w in q for w in agg_groups["AVG"]) and not explicit_for_agg["AVG"]:
            add_unique(required, add_column_if_named("average"))
        if any(w in q for w in agg_groups["MAX"]) and not explicit_for_agg["MAX"]:
            add_unique(required, add_column_if_named("highest"))
        if any(w in q for w in agg_groups["MIN"]) and not explicit_for_agg["MIN"]:
            add_unique(required, add_column_if_named("lowest"))

        # Remove aggregate-named columns when explicit aggregates are present
        if explicit_for_agg["AVG"]:
            required = remove_if(required, lambda c: norm(c) in ("average", "avg", "mean"))
        if explicit_for_agg["MAX"]:
            required = remove_if(required, lambda c: norm(c) in ("highest", "max", "maximum"))
        if explicit_for_agg["MIN"]:
            required = remove_if(required, lambda c: norm(c) in ("lowest", "min", "minimum"))

        # "Which year has most number of concerts" -> return year only, order by count
        if re.search(r"\b(year)\b", q) and re.search(r"\bmost\b", q) and "concert" in q:
            if re.search(r"\bwhich year\b|\bwhat year\b|\byear that\b", q):
                year_col = self._find_best_column_for_tokens(columns, ["year"])
                if year_col:
                    required = [year_col]
                    add_unique(group_by, year_col)
                add_unique(metrics, "COUNT(*)")
                add_unique(order_by, "COUNT(*) DESC")
                if limit in (None, "", 0):
                    limit = 1

        if re.search(r"\b(distinct|different|unique)\b", q):
            notes = (notes + " " if notes else "") + "Use DISTINCT for readability."

        plan["required_columns"] = required
        plan["optional_columns"] = optional
        plan["metrics"] = metrics
        plan["group_by"] = group_by
        plan["order_by"] = order_by
        plan["notes"] = notes
        plan["limit"] = limit
        return plan

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

            # If execution failed, try auto-repair
            if not success and isinstance(result, str):
                repaired_sql = self._auto_repair_sql(sql, result, context.schema_info)
                if repaired_sql and repaired_sql != sql:
                    self.logger.info(f"Attempting auto-repair for SQL error: {result[:100]}")
                    sql = repaired_sql
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
        """Generate SQL from natural language using LLM - with enhanced semantic understanding"""
        import requests
        import json

        # Format schema information for the LLM
        schema_description = self._format_schema_for_llm(schema)

        dialect = self._get_sql_dialect_name()
        q = query.lower()

        # Get enhanced semantic understanding
        semantic_analysis = self._enhanced_semantic_understanding(query, schema)
        join_decision = self._smart_join_decision(query, schema)

        # Build focused prompt with semantic insights
        special_instructions = ""

        # Add column warnings
        for warning in semantic_analysis['column_warnings']:
            special_instructions += f"\n- {warning}"

        # Add context hints
        for hint in semantic_analysis['context_hints']:
            special_instructions += f"\n- {hint}"

        # Add JOIN guidance
        if not join_decision['need_join']:
            special_instructions += f"\n- DO NOT USE JOIN: {join_decision.get('reason', 'All data in single table')}"
        elif join_decision.get('join_type'):
            special_instructions += f"\n- Use {join_decision['join_type']} JOIN: {join_decision['reason']}"

        # Original instructions for specific patterns
        # 1. Handle "which X" questions - return only X
        if "which" in q:
            if "year" in q:
                special_instructions += "\n- IMPORTANT: Return ONLY the year column (no COUNT)"
            elif "singer" in q or "artist" in q:
                special_instructions += "\n- IMPORTANT: Return ONLY the singer name (no other columns)"
            elif "song" in q:
                special_instructions += "\n- IMPORTANT: Return ONLY the song name (no other columns)"

        # 2. Handle "most/least" questions
        if ("most" in q or "least" in q) and ("which" in q or "what" in q):
            special_instructions += "\n- IMPORTANT: Use ORDER BY and LIMIT 1, return ONLY the requested entity"

        prompt = f"""Generate a {dialect} SQL query for this request.

DATABASE SCHEMA:
{schema_description}

USER REQUEST: {query}

CRITICAL RULES:
1. Return ONLY the SQL query - no explanations, no comments
2. Use EXACT column names from the schema (case-sensitive if needed)
3. Do NOT return extra columns unless specifically requested{special_instructions}
4. For aggregations, include proper GROUP BY
5. Do NOT use markdown or backticks

SQL:"""

        try:
            # Call LLM to generate SQL
            response = self._call_llm(prompt)

            # Clean up the SQL
            sql = self._clean_sql_response(response)

            if not sql:
                raise Exception("LLM did not return valid SQL")

            # Post-process to fix common issues
            sql = self._post_process_sql(sql, query, schema)

            return sql

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            # Use improved fallback
            return self._generate_fallback_sql(query, schema, [])

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
        output_plan = self._get_output_plan(query, schema)
        output_plan_text = ""
        if output_plan:
            output_plan_text = json.dumps(output_plan, indent=2, ensure_ascii=True)
        logic_hints = self._build_logic_hints(query)

        # Format validation results
        validation_info = "\nValidation Results:\n"
        for val in validations:
            status = "Success" if val.get("success") else f"Failed: {val.get('result')}"
            validation_info += f"- {val['purpose']}: {status}\n"

        dialect = self._get_sql_dialect_name()
        prompt = f"""You are a SQL expert. Generate a {dialect} query using the validation results.

Database Schema:
{schema_description}

Output Plan (required columns must be included):
{output_plan_text}

Logic Hints:
{logic_hints}

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
5. For ALL-of requests (both/each/every), use INTERSECT or GROUP BY HAVING COUNT(DISTINCT ...) = N (not IN alone)
6. If a column name equals an aggregate word (average/total/count), use the column directly unless the request explicitly asks for an aggregate
7. Use DISTINCT when listing unique values to improve readability (avoid DISTINCT in aggregates unless requested)
8. Do NOT alias SELECT expressions unless explicitly requested
9. Qualify columns with table aliases when joining multiple tables
10. DO NOT include any text before or after the SQL

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
        """Generate SQL learning from previous attempts with enhanced semantic understanding"""
        schema_description = self._format_schema_for_llm(schema)

        # Get enhanced semantic understanding
        semantic_analysis = self._enhanced_semantic_understanding(query, schema)
        join_decision = self._smart_join_decision(query, schema)

        # Build semantic guidance
        semantic_guidance = "\n==== SEMANTIC UNDERSTANDING ====\n"
        for warning in semantic_analysis['column_warnings']:
            semantic_guidance += f"- {warning}\n"
        for hint in semantic_analysis['context_hints']:
            semantic_guidance += f"- {hint}\n"

        # Add JOIN decision
        if not join_decision['need_join']:
            semantic_guidance += f"- NO JOIN NEEDED: {join_decision.get('reason', '')}\n"
        elif join_decision.get('join_type'):
            semantic_guidance += f"- USE {join_decision['join_type']} JOIN: {join_decision['reason']}\n"

        # Simplified output plan - only for complex queries
        output_plan_text = ""
        if self._is_complex_query(query):
            output_plan = self._get_simplified_output_plan(query, schema)
            if output_plan:
                output_plan_text = f"\nKey Requirements:\n"
                if output_plan.get("target"):
                    output_plan_text += f"- Return: {output_plan['target']}\n"
                if output_plan.get("aggregations"):
                    output_plan_text += f"- Aggregations: {', '.join(output_plan['aggregations'])}\n"
                if output_plan.get("limit"):
                    output_plan_text += f"- Limit: {output_plan['limit']}\n"

        logic_hints = self._build_logic_hints(query)

        # Enhanced error analysis
        attempts_info = "\n==== PREVIOUS ATTEMPTS AND ERROR ANALYSIS ====\n"
        error_patterns = {}

        for i, attempt in enumerate(attempts, 1):
            sql = attempt.get('sql', 'Unknown')
            error = attempt.get('error')
            error_text = str(error) if error is not None else "None"

            # Analyze error type
            error_type = self._classify_error(error_text)
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

            attempts_info += f"\nAttempt {i}:\n"
            attempts_info += f"SQL: {sql}\n"
            attempts_info += f"Error Type: {error_type}\n"
            attempts_info += f"Full Error: {error_text}\n"

            # Provide specific guidance based on error type
            guidance = self._get_error_guidance(error_type, error_text, schema)
            if guidance:
                attempts_info += f"Guidance: {guidance}\n"

        # Add pattern summary
        if error_patterns:
            attempts_info += "\n==== ERROR PATTERNS ====\n"
            for error_type, count in error_patterns.items():
                attempts_info += f"- {error_type}: {count} times\n"

        dialect = self._get_sql_dialect_name()
        prompt = f"""You are a SQL expert. Learn from the previous failed attempts and generate the CORRECT SQL.

DATABASE SCHEMA:
{schema_description}
{semantic_guidance}
{output_plan_text}
{attempts_info}

USER REQUEST: {query}

CRITICAL RULES TO AVOID PREVIOUS ERRORS:
1. Column Names: Use EXACT column names from schema (case-sensitive if needed)
2. Table Names: Use EXACT table names from schema
3. Aggregates vs Columns: If schema has column "average", use it directly unless user asks "average of X"
4. JOIN Usage: Only add JOINs when actually needed to link data
5. Result Columns: For "Which X" or "What X" questions, return ONLY the asked column(s)
6. GROUP BY: Include all non-aggregate columns from SELECT

COMMON FIXES:
- "no such column" → Check exact column name in schema
- "ambiguous column" → Add table prefix (table.column)
- "not in GROUP BY" → Add the column to GROUP BY clause
- Wrong results → Check if you're using the right aggregate function

Generate the CORRECT SQL (no explanations, no comments):"""

        try:
            sql = self._call_llm(prompt)
            sql = self._clean_sql_response(sql)

            # Post-process to fix common issues
            sql = self._post_process_sql(sql, query, schema)

            return sql
        except Exception as e:
            self.logger.error(f"LLM learning query failed: {str(e)}")
            # Improved fallback
            return self._generate_fallback_sql(query, schema, attempts)

    def _classify_error(self, error_text: str) -> str:
        """Classify error type for better learning"""
        if not error_text or error_text == "None":
            return "Unknown"

        error_lower = str(error_text).lower()

        if "no such column" in error_lower or "unknown column" in error_lower:
            return "Column not found"
        elif "ambiguous" in error_lower:
            return "Ambiguous column"
        elif "no such table" in error_lower or "table" in error_lower and "doesn't exist" in error_lower:
            return "Table not found"
        elif "group by" in error_lower or "not in group by" in error_lower:
            return "GROUP BY missing"
        elif "syntax error" in error_lower:
            return "Syntax error"
        elif "near" in error_lower and "syntax" in error_lower:
            return "SQL syntax invalid"
        elif "no such function" in error_lower:
            return "Function not found"
        elif "data type" in error_lower or "type mismatch" in error_lower:
            return "Type mismatch"
        else:
            return "Execution error"

    def _get_error_guidance(self, error_type: str, error_text: str, schema: SchemaInfo) -> str:
        """Provide specific guidance based on error type"""
        guidance = ""

        if error_type == "Column not found":
            # Extract the problematic column name
            import re
            match = re.search(r"column[: ]+(['\"`]?)(\w+)(['\"`]?)", error_text, re.IGNORECASE)
            if match:
                bad_col = match.group(2)
                columns = self._collect_schema_columns(schema)
                best_match = self._find_best_column_for_tokens(columns, [bad_col])
                if best_match:
                    guidance = f"Replace '{bad_col}' with '{best_match}'"
                else:
                    available_cols = [c["name"] for c in columns[:5]]
                    guidance = f"Column '{bad_col}' doesn't exist. Available columns include: {', '.join(available_cols)}"

        elif error_type == "Ambiguous column":
            guidance = "Add table prefix to disambiguate (e.g., table.column)"

        elif error_type == "GROUP BY missing":
            guidance = "Add all non-aggregate columns from SELECT to GROUP BY clause"

        elif error_type == "Table not found":
            available_tables = list(schema.tables.keys()) if schema and schema.tables else []
            guidance = f"Available tables: {', '.join(available_tables)}"

        return guidance

    def _is_complex_query(self, query: str) -> bool:
        """Determine if query is complex enough to need detailed planning"""
        q = query.lower()
        complex_indicators = [
            "join", "group by", "having", "union", "intersect",
            "with", "subquery", "nested", "average of", "sum of",
            "for each", "by each", "per"
        ]
        return any(indicator in q for indicator in complex_indicators)

    def _get_simplified_output_plan(self, query: str, schema: SchemaInfo) -> Dict[str, Any]:
        """Generate simplified output plan focusing on key requirements only"""
        # Use cache if available
        cache_key = (query or "").strip().lower()
        if cache_key in self.plan_cache:
            return self.plan_cache[cache_key]

        # Simple heuristic-based plan
        plan = {}
        q = query.lower()

        # Determine target entity
        if "which" in q or "what" in q:
            # Extract the entity being asked about
            if "which year" in q:
                plan["target"] = "year only"
            elif "which singer" in q or "which artist" in q:
                plan["target"] = "singer name only"
            elif "what song" in q or "which song" in q:
                plan["target"] = "song name only"

        # Detect aggregations
        aggregations = []
        if "count" in q or "how many" in q:
            aggregations.append("COUNT")
        if "average" in q and "average of" in q:
            aggregations.append("AVG")
        if "sum" in q or "total" in q:
            aggregations.append("SUM")
        if "maximum" in q or "max" in q or "highest" in q:
            aggregations.append("MAX")
        if "minimum" in q or "min" in q or "lowest" in q:
            aggregations.append("MIN")

        if aggregations:
            plan["aggregations"] = aggregations

        # Detect limit
        if "most" in q or "least" in q or "top" in q or "best" in q or "worst" in q:
            plan["limit"] = 1

        self.plan_cache[cache_key] = plan
        return plan

    def _post_process_sql(self, sql: str, query: str, schema: SchemaInfo) -> str:
        """Post-process SQL to fix common issues"""
        if not sql:
            return sql

        q = query.lower()

        # Fix: Remove extra columns for "Which X" questions
        if ("which" in q or "what" in q) and "," in sql:
            # Check if returning multiple columns when only one is asked
            if "which year" in q and "count" in sql.lower():
                # Remove COUNT column if only year is asked
                sql = self._remove_count_from_select(sql)

        return sql

    def _remove_count_from_select(self, sql: str) -> str:
        """Remove COUNT from SELECT when only the grouped column is needed"""
        import re
        # Pattern to match SELECT with COUNT
        pattern = r'SELECT\s+(\w+)\s*,\s*COUNT\([^)]*\)[^,]*\s+FROM'
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            # Keep only the first column
            sql = re.sub(pattern, f'SELECT {match.group(1)} FROM', sql, flags=re.IGNORECASE)
        return sql

    def _enhanced_semantic_understanding(self, query: str, schema: SchemaInfo) -> dict:
        """增强语义理解，避免列名混淆"""

        # 1. 检查是否存在容易与聚合函数混淆的列名
        ambiguous_columns = {
            'average': 'AVG',
            'count': 'COUNT',
            'sum': 'SUM',
            'maximum': 'MAX',
            'minimum': 'MIN'
        }

        column_warnings = []
        for table_name, table in schema.tables.items():
            for col in table.columns:
                if col.name.lower() in ambiguous_columns:
                    column_warnings.append(
                        f"IMPORTANT: '{col.name}' in table '{table.name}' is a COLUMN NAME, "
                        f"not the {ambiguous_columns[col.name.lower()]}() function!"
                    )

        # 2. 上下文理解增强 - 区分song names vs singer names
        context_hints = []
        query_lower = query.lower()

        if "song" in query_lower and "name" in query_lower:
            context_hints.append("When asking for song names, use 'song_name' column, not 'Name' column")
        elif "singer" in query_lower and "name" in query_lower and "song" not in query_lower:
            context_hints.append("When asking for singer names, use 'Name' column")

        # 3. 检测"the average"是否指列名
        if "the average" in query_lower and "of all" in query_lower:
            # Check if 'average' column exists
            for table_name, table in schema.tables.items():
                if any(col.name.lower() == 'average' for col in table.columns):
                    context_hints.append(
                        "CRITICAL: 'the average' likely refers to the 'average' COLUMN, not AVG() function"
                    )
                    break

        return {
            'column_warnings': column_warnings,
            'context_hints': context_hints
        }

    def _smart_join_decision(self, query: str, schema: SchemaInfo) -> dict:
        """智能判断是否需要JOIN以及JOIN类型"""

        query_lower = query.lower()

        # 1. 检查是否所有信息都在同一个表中
        # 例如：singer表同时包含singer信息和song信息
        if 'singer' in query_lower and 'song' in query_lower and 'concert' not in query_lower:
            # Check if singer table has song columns
            singer_table = next((t for t in schema.tables.values() if t.name.lower() == 'singer'), None)
            if singer_table:
                has_song_cols = any('song' in col.name.lower() for col in singer_table.columns)
                if has_song_cols:
                    return {
                        'need_join': False,
                        'reason': 'Song information is in singer table, no JOIN needed'
                    }

        # 2. 分析介词和关键词
        # "by singers above average age" - 这里的"by"不需要JOIN
        if "by singers" in query_lower and "average age" in query_lower:
            return {
                'need_join': False,
                'reason': 'Filter condition on same table, no JOIN needed'
            }

        # 3. "for each" 通常需要JOIN，但要选择正确的类型
        if 'for each' in query_lower:
            # "all stadiums" 或 "each stadium" - 使用LEFT JOIN包含所有
            if 'all' in query_lower or 'every' in query_lower:
                return {
                    'need_join': True,
                    'join_type': 'LEFT',
                    'reason': 'Show all records including those with 0 count'
                }
            else:
                # 默认INNER JOIN只显示有关联的
                return {
                    'need_join': True,
                    'join_type': 'INNER',
                    'reason': 'Only show records with matches'
                }

        # 4. 明确需要多表信息时
        if 'concert' in query_lower and ('singer' in query_lower or 'stadium' in query_lower):
            return {
                'need_join': True,
                'join_type': 'INNER',
                'reason': 'Query spans multiple tables'
            }

        return {'need_join': False}

    def _generate_fallback_sql(self, query: str, schema: SchemaInfo, attempts: List[Dict]) -> str:
        """Generate a better fallback SQL based on query intent"""
        q = query.lower()
        main_table = self._guess_main_table(schema)

        # Simple query patterns
        if "how many" in q or "count" in q:
            return f"SELECT COUNT(*) FROM {main_table}"
        elif "all" in q and any(word in q for word in ["show", "list", "display", "get"]):
            return f"SELECT * FROM {main_table} LIMIT 100"
        elif "average" in q:
            # Try to find a numeric column
            for table_info in schema.tables.values():
                for col in table_info.columns:
                    if "age" in col.name.lower() or "price" in col.name.lower() or "amount" in col.name.lower():
                        return f"SELECT AVG({col.name}) FROM {main_table}"

        return f"SELECT * FROM {main_table} LIMIT 10"

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

    def _get_output_plan(self, query: str, schema: SchemaInfo) -> Dict[str, Any]:
        """Simplified output plan generation - only for complex queries"""
        # For simple queries, skip plan generation entirely
        if not self._is_complex_query(query):
            return {}

        # Use the simplified version
        return self._get_simplified_output_plan(query, schema)

    def _apply_enhanced_plan_heuristics(self, plan: Dict[str, Any], query: str, schema: SchemaInfo) -> Dict[str, Any]:
        """Enhanced heuristics with better column matching and join detection."""
        # Call the existing heuristics first
        plan = self._apply_output_plan_heuristics(plan, query, schema)

        # Additional enhancements
        q = (query or "").lower()

        # Enhance join detection
        if "joins" not in plan or not plan["joins"]:
            plan["joins"] = self._detect_required_joins(plan.get("tables", []), schema)

        # Enhance semantic hints
        semantic_hints = plan.get("semantic_hints", [])

        # Detect common patterns that need special handling
        if re.search(r"\b(which|what)\s+\w+\s+(has|have)\s+(the\s+)?(most|highest|largest)", q):
            semantic_hints.append("Return only the entity with the maximum value, not all entities")
            if not plan.get("limit"):
                plan["limit"] = 1

        if re.search(r"\bfor each\b|\bevery\b", q):
            semantic_hints.append("Results should be grouped by the mentioned entity")

        if re.search(r"\bboth\b|\ball of\b|\beach of\b", q):
            semantic_hints.append("Use INTERSECT or GROUP BY HAVING COUNT for ALL-of conditions")

        plan["semantic_hints"] = semantic_hints

        return plan

    def _detect_required_joins(self, tables: List[str], schema: SchemaInfo) -> List[Dict[str, Any]]:
        """Detect required joins between tables based on schema relationships."""
        if len(tables) <= 1:
            return []

        joins = []
        # Simple heuristic: look for foreign key patterns
        for i, table1 in enumerate(tables[:-1]):
            for table2 in tables[i+1:]:
                join_condition = self._find_join_condition(table1, table2, schema)
                if join_condition:
                    joins.append({
                        "table1": table1,
                        "table2": table2,
                        "on": join_condition,
                        "type": "INNER"
                    })

        return joins

    def _find_join_condition(self, table1: str, table2: str, schema: SchemaInfo) -> Optional[str]:
        """Find join condition between two tables."""
        if not schema or not schema.tables:
            return None

        t1_info = schema.tables.get(table1)
        t2_info = schema.tables.get(table2)

        if not t1_info or not t2_info:
            return None

        # Look for foreign key patterns
        for col in t1_info.columns:
            col_lower = col.name.lower()
            # Check if column references table2
            if f"{table2}_id" in col_lower or f"{table2}id" in col_lower:
                return f"{table1}.{col.name} = {table2}.id"

        for col in t2_info.columns:
            col_lower = col.name.lower()
            # Check if column references table1
            if f"{table1}_id" in col_lower or f"{table1}id" in col_lower:
                return f"{table2}.{col.name} = {table1}.id"

        return None

    def _auto_repair_sql(self, sql: str, error: str, schema: SchemaInfo) -> Optional[str]:
        """Automatically repair common SQL errors."""
        if not sql or not error:
            return None

        error_lower = error.lower()
        repaired_sql = sql

        # 1. Fix column name errors
        if "no such column" in error_lower or "unknown column" in error_lower:
            # Extract the bad column name
            import re
            match = re.search(r"column[: ]+(['\"`]?)(\w+)(['\"`]?)", error, re.IGNORECASE)
            if match:
                bad_column = match.group(2)
                # Find best matching column
                columns = self._collect_schema_columns(schema)
                best_match = self._find_best_column_for_tokens(columns, [bad_column])
                if best_match:
                    pattern = r'\b' + re.escape(bad_column) + r'\b'
                    repaired_sql = re.sub(pattern, best_match, sql, flags=re.IGNORECASE)
                    self.logger.info(f"Auto-repair: Replaced '{bad_column}' with '{best_match}'")

        # 2. Fix ambiguous column errors
        elif "ambiguous column" in error_lower:
            match = re.search(r"column[: ]+(['\"`]?)(\w+)(['\"`]?)", error, re.IGNORECASE)
            if match:
                ambiguous_col = match.group(2)
                # Find tables that contain this column
                tables_with_col = []
                for table_name, table_info in schema.tables.items():
                    for col in table_info.columns:
                        if col.name.lower() == ambiguous_col.lower():
                            tables_with_col.append(table_name)
                            break

                if tables_with_col:
                    # Add table prefix to the ambiguous column (use first matching table)
                    pattern = r'\b' + re.escape(ambiguous_col) + r'\b'
                    replacement = f"{tables_with_col[0]}.{ambiguous_col}"
                    # Only replace in SELECT and WHERE clauses to avoid breaking aliases
                    repaired_sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                    self.logger.info(f"Auto-repair: Disambiguated '{ambiguous_col}' as '{replacement}'")

        # 3. Fix missing GROUP BY columns
        elif "not in group by" in error_lower or "must appear in the group by" in error_lower:
            # Extract non-aggregated columns from SELECT
            select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # Find non-aggregated columns
                non_agg_cols = []
                for part in select_clause.split(','):
                    part = part.strip()
                    # Skip aggregated columns
                    if not re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", part, re.IGNORECASE):
                        # Extract column name
                        col_match = re.match(r"([\w\.]+)(?:\s+AS\s+\w+)?", part, re.IGNORECASE)
                        if col_match:
                            non_agg_cols.append(col_match.group(1))

                if non_agg_cols:
                    # Check if GROUP BY exists
                    if re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
                        # Add missing columns to existing GROUP BY
                        group_match = re.search(r"(GROUP\s+BY\s+)(.*?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|\s*$)",
                                              sql, re.IGNORECASE)
                        if group_match:
                            existing = group_match.group(2)
                            new_cols = [c for c in non_agg_cols if c not in existing]
                            if new_cols:
                                new_group = existing + ", " + ", ".join(new_cols)
                                repaired_sql = sql[:group_match.start(2)] + new_group + sql[group_match.end(2):]
                                self.logger.info(f"Auto-repair: Added columns to GROUP BY: {new_cols}")
                    else:
                        # Add GROUP BY clause
                        insert_point = len(sql)
                        for keyword in ["ORDER BY", "LIMIT", ";"]:
                            match = re.search(r"\b" + keyword + r"\b", sql, re.IGNORECASE)
                            if match:
                                insert_point = min(insert_point, match.start())
                        group_by = " GROUP BY " + ", ".join(non_agg_cols)
                        repaired_sql = sql[:insert_point] + group_by + " " + sql[insert_point:]
                        self.logger.info(f"Auto-repair: Added GROUP BY clause with columns: {non_agg_cols}")

        # 4. Fix table name errors
        elif "no such table" in error_lower or ("table" in error_lower and "doesn't exist" in error_lower):
            match = re.search(r"table[: ]+(['\"`]?)(\w+)(['\"`]?)", error, re.IGNORECASE)
            if match:
                bad_table = match.group(2)
                # Find best matching table using edit distance
                best_table = None
                min_distance = float('inf')
                for table_name in schema.tables.keys():
                    # Simple edit distance calculation
                    distance = sum(1 for a, b in zip(bad_table.lower(), table_name.lower()) if a != b)
                    distance += abs(len(bad_table) - len(table_name))
                    if distance < min_distance:
                        min_distance = distance
                        best_table = table_name

                if best_table and min_distance <= 3:
                    pattern = r'\b' + re.escape(bad_table) + r'\b'
                    repaired_sql = re.sub(pattern, best_table, sql, flags=re.IGNORECASE)
                    self.logger.info(f"Auto-repair: Replaced table '{bad_table}' with '{best_table}'")

        return repaired_sql if repaired_sql != sql else None

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
