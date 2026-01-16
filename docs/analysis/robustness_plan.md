# Robustness Plan (Schema Relations, Semantics, Evaluation)

## Goals

- Improve success rate on multi-table schemas and ambiguous field names.
- Reduce wrong-but-executable SQL by enforcing intent alignment.
- Align evaluation with real-world usage (LIMIT, multi-statement, final-result).
- Keep MCP optional; work well with local introspection first.

## Observed Failure Modes

1. Join semantics drift (INNER vs LEFT), leading to extra/missing rows.
2. Missing or wrong GROUP BY / aggregation intent.
3. Column selection mismatch (extra columns, wrong column for question).
4. Set logic confusion (INTERSECT vs OR).
5. Field semantics confusion (e.g., column named `average`).
6. Multi-statement output causing SQLite execution errors.
7. Default LIMIT penalized by strict benchmark matching.

## Proposal A: Relationship Graph + Join Guidance

### What

- Build a join graph per database using:
  - Foreign keys when available.
  - Heuristics when FKs missing: `_id` to PK match, name similarity, type match.
  - Optional overlap checks via small sample value overlaps.

### Why

- Reduces join mistakes and improves join path selection.

### Implementation Sketch

- Add `SchemaRelationGraph` to schema discovery.
- Emit relationship hints in schema prompt:
  - Suggested join paths for common table pairs.
  - Preferred join direction and keys.

## Proposal B: Field Semantics Enrichment

### What

- Profile columns with sample stats:
  - Distinct count, min/max, null rate, value examples.
- Maintain a configurable schema dictionary:
  - Field aliases and business definitions (YAML/JSON).
- Add semantic linking:
  - Synonyms map (name/title, year/release_year, user/customer).

### Why

- Reduces column mis-selection and wrong aggregation.

## Proposal C: Output Plan (Required Columns First)

### What

- Two-stage generation:
  1) Plan JSON: `required_columns`, `metrics`, `filters`, `group_by`, `order_by`.
  2) SQL generation constrained by plan.

### Why

- Prevents extra columns, wrong ordering, and missing groupings.

## Proposal D: Error-Driven Auto Repair

### What

- On execution error, apply rule-based fixes:
  - `no such column` -> fuzzy-match replacement.
  - `no such table` -> fuzzy-match replacement.
  - Multi-statement -> keep first statement or parse.

### Why

- Improves pass rate without external tools.

## Proposal E: MCP (Optional)

### When to Use

- Multiple external sources / catalog systems.
- Centralized data dictionary, dbt docs, BI metadata.

### Benefit

- Add canonical descriptions, ownership, column meaning.
- Improves disambiguation without hardcoding.

### Non-MCP Path

- Introspection + dictionary + profiling is sufficient for most local DBs.

## Proposal F: Evaluation Redefinition (Business-Realistic)

### EA (Execution Accuracy) Variants

1) Strict EA (current): result equals gold result.
2) Limit-Tolerant EA:
   - If predicted SQL has LIMIT N, compare against gold result head N.
3) Final-Result EA (Multi-Statement):
   - Allow multiple SQL statements; compare final SELECT result only.

### Why

- Reflects production usage (LIMIT is desirable).
- Multi-statement workflows are common in exploratory tasks.

## Rollout Plan

1. Join graph + guidance in prompt.
2. Output plan stage + required column enforcement.
3. Error-driven auto repair.
4. Benchmark EA variants + reporting.
5. Optional MCP integration (if needed).

## Metrics to Track

- Execution Accuracy (strict / limit-tolerant / final-result).
- Error breakdown (column/table/join/aggregation).
- Average attempts to success.
- Time per query and retry cost.
