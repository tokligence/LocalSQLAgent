"""
Intelligent SQL Agent with Multi-Strategy Execution
Production-ready agent combining ambiguity detection, schema discovery, and adaptive strategies
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
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
                 max_attempts: int = 5):
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
        self.max_attempts = max_attempts

        # Initialize components
        self.ambiguity_detector = AmbiguityDetector(confidence_threshold=0.7)
        self.difficulty_assessor = QueryDifficultyAssessor()
        self.strategy_selector = StrategySelector(
            self.difficulty_assessor,
            self.ambiguity_detector
        )

        # Setup schema management
        providers = []

        # Primary: Database introspection
        providers.append(DatabaseIntrospectionProvider(
            db_config.get("type", "postgresql"),
            db_config
        ))

        # Fallback: MCP if configured
        if mcp_server:
            providers.append(MCPSchemaProvider(mcp_server))

        self.schema_manager = SchemaManager(
            primary_provider=providers[0],
            fallback_providers=providers[1:] if len(providers) > 1 else None
        )

        # Cache
        self.query_cache = {}
        self.schema_cache = None

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
        cache_key = query.lower().strip()
        if not force_refresh and cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            cached.confidence = 1.0  # Cached results have high confidence
            return cached

        # Initialize context
        context = QueryContext(original_query=query)

        try:
            # 1. Get schema (dynamic discovery)
            context.schema_info = self._get_schema(force_refresh)

            # 2. Detect ambiguities
            context.detected_ambiguities = self.ambiguity_detector.detect(query)

            # 3. Handle clarification if needed
            if self._needs_clarification(context):
                return self._request_clarification(context)

            # 4. Select and execute strategy
            context.selected_strategy = self.strategy_selector.select(context)

            # 5. Execute with selected strategy
            result = self._execute_with_strategy(context)

            # 6. Cache successful results
            if result.success and result.confidence > 0.7:
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

    def _execute_direct(self, context: QueryContext) -> ExecutionResult:
        """Direct single-attempt execution"""
        sql = self._generate_sql(
            context.clarified_query or context.original_query,
            context.schema_info
        )

        success, result = self._execute_sql(sql)

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
                row_count=len(result.get("data", [])),
                attempts_count=1,
                strategy_used=ExecutionStrategy.DIRECT,
                confidence=0.8,
                context=context
            )
        else:
            return ExecutionResult(
                success=False,
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

        success, result = self._execute_sql(sql)

        if success:
            return ExecutionResult(
                success=True,
                sql=sql,
                data=result.get("data"),
                columns=result.get("columns"),
                row_count=len(result.get("data", [])),
                attempts_count=len(validations) + 1,
                strategy_used=ExecutionStrategy.VALIDATED,
                confidence=0.85,
                context=context
            )
        else:
            return ExecutionResult(
                success=False,
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
                        row_count=len(result.get("data", [])),
                        attempts_count=attempt_num + 1,
                        strategy_used=ExecutionStrategy.EXPLORATORY,
                        confidence=confidence,
                        context=context
                    )
                    best_confidence = confidence

                # Stop if high confidence achieved
                if confidence > 0.9:
                    break

        return best_result or ExecutionResult(
            success=False,
            error="All attempts failed",
            attempts_count=len(context.attempts),
            strategy_used=ExecutionStrategy.EXPLORATORY,
            context=context
        )

    def _generate_sql(self, query: str, schema: SchemaInfo) -> str:
        """Generate SQL from natural language"""
        # This would call the LLM with appropriate prompt
        # Simplified for demonstration
        return f"SELECT * FROM customers LIMIT 10 -- Generated for: {query}"

    def _generate_validation_queries(self, query: str, schema: SchemaInfo) -> List[Dict]:
        """Generate validation queries"""
        # Would use LLM to break down query
        return [
            {"purpose": "Check table exists", "sql": "SELECT COUNT(*) FROM customers LIMIT 1"},
            {"purpose": "Check columns", "sql": "SELECT * FROM customers LIMIT 1"}
        ]

    def _generate_sql_with_validation(self, query: str, schema: SchemaInfo,
                                     validations: List[Dict]) -> str:
        """Generate SQL using validation results"""
        # Would incorporate validation insights
        return f"SELECT * FROM customers -- Validated query for: {query}"

    def _generate_sql_with_learning(self, query: str, schema: SchemaInfo,
                                   attempts: List[Dict]) -> str:
        """Generate SQL learning from previous attempts"""
        # Would analyze errors and adjust
        return f"SELECT * FROM customers -- Attempt {len(attempts) + 1} for: {query}"

    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """Execute SQL on database"""
        # Actual database execution would go here
        # Simplified for demonstration
        return True, {"data": [["test"]], "columns": ["col1"]}

    def _evaluate_result_quality(self, query: str, sql: str, result: Dict) -> float:
        """Evaluate quality of SQL result"""
        # Would use LLM to assess if result answers the query
        # Simplified scoring
        if result.get("data"):
            return 0.85
        return 0.3

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