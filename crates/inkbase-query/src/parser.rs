use pest::Parser;
use pest_derive::Parser;
use uuid::Uuid;

use crate::ast::{
    CompareOp, Condition, Entity, Query, SearchQuery, SearchType, SelectQuery, Value,
};
use crate::error::MqlError;

#[derive(Parser)]
#[grammar = "mql.pest"]
struct MqlParser;

/// Parse an MQL query string into an AST.
pub fn parse_query(input: &str) -> Result<Query, MqlError> {
    let pairs = MqlParser::parse(Rule::query, input)
        .map_err(|e| MqlError::ParseError(e.to_string()))?;

    let query_pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| MqlError::ParseError("empty input".into()))?;

    // The `query` rule contains either a select_query or search_query as inner pair.
    for inner in query_pair.into_inner() {
        match inner.as_rule() {
            Rule::select_query => return parse_select(inner),
            Rule::search_query => return parse_search(inner),
            Rule::EOI => {}
            _ => {}
        }
    }

    Err(MqlError::ParseError("no query found".into()))
}

fn parse_select(pair: pest::iterators::Pair<Rule>) -> Result<Query, MqlError> {
    let mut inner = pair.into_inner();

    let entity_pair = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected entity".into()))?;
    let entity = parse_entity(entity_pair)?;

    let mut conditions = Vec::new();
    let mut limit = None;

    for pair in inner {
        match pair.as_rule() {
            Rule::where_clause => {
                conditions = parse_where_clause(pair)?;
            }
            Rule::limit_clause => {
                limit = Some(parse_limit(pair)?);
            }
            _ => {}
        }
    }

    Ok(Query::Select(SelectQuery {
        entity,
        conditions,
        limit,
    }))
}

fn parse_entity(pair: pest::iterators::Pair<Rule>) -> Result<Entity, MqlError> {
    let text = pair.as_str().to_lowercase();
    match text.as_str() {
        "documents" => Ok(Entity::Documents),
        "blocks" => Ok(Entity::Blocks),
        "links" => Ok(Entity::Links),
        "tags" => Ok(Entity::Tags),
        _ => Err(MqlError::ParseError(format!("unknown entity: {}", text))),
    }
}

fn parse_where_clause(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Condition>, MqlError> {
    let mut conditions = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::condition {
            conditions.push(parse_condition(inner)?);
        }
    }
    Ok(conditions)
}

fn parse_condition(pair: pest::iterators::Pair<Rule>) -> Result<Condition, MqlError> {
    let mut inner = pair.into_inner();

    let field = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected field name".into()))?
        .as_str()
        .to_string();

    let op_pair = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected operator".into()))?;
    let op = parse_op(op_pair)?;

    let value_pair = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected value".into()))?;
    let value = parse_value(value_pair)?;

    Ok(Condition { field, op, value })
}

fn parse_op(pair: pest::iterators::Pair<Rule>) -> Result<CompareOp, MqlError> {
    match pair.as_str() {
        "=" => Ok(CompareOp::Eq),
        "!=" => Ok(CompareOp::Neq),
        ">" => Ok(CompareOp::Gt),
        "<" => Ok(CompareOp::Lt),
        ">=" => Ok(CompareOp::Gte),
        "<=" => Ok(CompareOp::Lte),
        other => Err(MqlError::ParseError(format!("unknown operator: {}", other))),
    }
}

fn parse_value(pair: pest::iterators::Pair<Rule>) -> Result<Value, MqlError> {
    // The `value` rule wraps one of: uuid_value, string_value, number_value
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| MqlError::ParseError("expected value".into()))?;

    match inner.as_rule() {
        Rule::uuid_value => {
            let raw = inner.as_str();
            let uuid = Uuid::parse_str(raw)
                .map_err(|e| MqlError::InvalidUuid(format!("{}: {}", raw, e)))?;
            Ok(Value::Uuid(uuid))
        }
        Rule::string_value => {
            let raw = inner.as_str();
            // Strip surrounding single quotes
            let content = &raw[1..raw.len() - 1];
            Ok(Value::String(content.to_string()))
        }
        Rule::number_value => {
            let raw = inner.as_str();
            let n: i64 = raw
                .parse()
                .map_err(|e| MqlError::InvalidNumber(format!("{}: {}", raw, e)))?;
            Ok(Value::Number(n))
        }
        _ => Err(MqlError::ParseError("unexpected value type".into())),
    }
}

fn parse_search(pair: pest::iterators::Pair<Rule>) -> Result<Query, MqlError> {
    let mut inner = pair.into_inner();

    let type_pair = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected search type".into()))?;
    let search_type = match type_pair.as_str().to_lowercase().as_str() {
        "fulltext" => SearchType::Fulltext,
        "semantic" => SearchType::Semantic,
        other => {
            return Err(MqlError::ParseError(format!(
                "unknown search type: {}",
                other
            )))
        }
    };

    let query_pair = inner
        .next()
        .ok_or_else(|| MqlError::ParseError("expected search query string".into()))?;
    let raw = query_pair.as_str();
    let query_text = raw[1..raw.len() - 1].to_string();

    let mut limit = None;
    for pair in inner {
        if pair.as_rule() == Rule::limit_clause {
            limit = Some(parse_limit(pair)?);
        }
    }

    Ok(Query::Search(SearchQuery {
        search_type,
        query_text,
        limit,
    }))
}

fn parse_limit(pair: pest::iterators::Pair<Rule>) -> Result<usize, MqlError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| MqlError::ParseError("expected limit value".into()))?;
    let raw = inner.as_str();
    raw.parse::<usize>()
        .map_err(|e| MqlError::InvalidNumber(format!("{}: {}", raw, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_documents_no_where() {
        let q = parse_query("SELECT documents").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Documents);
                assert!(sel.conditions.is_empty());
                assert_eq!(sel.limit, None);
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_select_blocks_with_string_condition() {
        let q = parse_query("SELECT blocks WHERE doc_id = 'some-value'").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Blocks);
                assert_eq!(sel.conditions.len(), 1);
                let cond = &sel.conditions[0];
                assert_eq!(cond.field, "doc_id");
                assert_eq!(cond.op, CompareOp::Eq);
                assert_eq!(cond.value, Value::String("some-value".into()));
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_select_blocks_with_multiple_conditions() {
        let q =
            parse_query("SELECT blocks WHERE block_type = 'heading' AND level = 2").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Blocks);
                assert_eq!(sel.conditions.len(), 2);

                assert_eq!(sel.conditions[0].field, "block_type");
                assert_eq!(sel.conditions[0].op, CompareOp::Eq);
                assert_eq!(
                    sel.conditions[0].value,
                    Value::String("heading".into())
                );

                assert_eq!(sel.conditions[1].field, "level");
                assert_eq!(sel.conditions[1].op, CompareOp::Eq);
                assert_eq!(sel.conditions[1].value, Value::Number(2));
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_select_documents_with_path() {
        let q = parse_query("SELECT documents WHERE path = 'notes/rust.md'").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Documents);
                assert_eq!(sel.conditions.len(), 1);
                assert_eq!(sel.conditions[0].field, "path");
                assert_eq!(
                    sel.conditions[0].value,
                    Value::String("notes/rust.md".into())
                );
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_select_with_limit() {
        let q = parse_query("SELECT blocks WHERE language = 'rust' LIMIT 10").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Blocks);
                assert_eq!(sel.conditions.len(), 1);
                assert_eq!(sel.conditions[0].field, "language");
                assert_eq!(
                    sel.conditions[0].value,
                    Value::String("rust".into())
                );
                assert_eq!(sel.limit, Some(10));
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_search_fulltext() {
        let q = parse_query("SEARCH fulltext 'rust programming'").unwrap();
        match q {
            Query::Search(s) => {
                assert_eq!(s.search_type, SearchType::Fulltext);
                assert_eq!(s.query_text, "rust programming");
                assert_eq!(s.limit, None);
            }
            _ => panic!("expected Search query"),
        }
    }

    #[test]
    fn test_search_semantic_with_limit() {
        let q = parse_query("SEARCH semantic 'how to parse markdown' LIMIT 5").unwrap();
        match q {
            Query::Search(s) => {
                assert_eq!(s.search_type, SearchType::Semantic);
                assert_eq!(s.query_text, "how to parse markdown");
                assert_eq!(s.limit, Some(5));
            }
            _ => panic!("expected Search query"),
        }
    }

    #[test]
    fn test_case_insensitivity() {
        let q = parse_query("select DOCUMENTS where PATH = 'test'").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Documents);
                assert_eq!(sel.conditions.len(), 1);
                // Field names are case-sensitive (identifiers), but keywords/entity are not
                assert_eq!(sel.conditions[0].field, "PATH");
                assert_eq!(
                    sel.conditions[0].value,
                    Value::String("test".into())
                );
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_invalid_query_returns_error() {
        let result = parse_query("INVALID query text");
        assert!(result.is_err());
    }

    #[test]
    fn test_select_links() {
        let q = parse_query("SELECT links").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Links);
                assert!(sel.conditions.is_empty());
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_select_tags() {
        let q = parse_query("SELECT tags").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Tags);
                assert!(sel.conditions.is_empty());
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_uuid_value_in_condition() {
        let q = parse_query(
            "SELECT blocks WHERE doc_id = 550e8400-e29b-41d4-a716-446655440000",
        )
        .unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions.len(), 1);
                let expected =
                    Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
                assert_eq!(sel.conditions[0].value, Value::Uuid(expected));
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let q = parse_query("SELECT blocks WHERE level >= 2").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions[0].op, CompareOp::Gte);
                assert_eq!(sel.conditions[0].value, Value::Number(2));
            }
            _ => panic!("expected Select query"),
        }

        let q = parse_query("SELECT blocks WHERE level != 0").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions[0].op, CompareOp::Neq);
            }
            _ => panic!("expected Select query"),
        }

        let q = parse_query("SELECT blocks WHERE level <= 5").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions[0].op, CompareOp::Lte);
            }
            _ => panic!("expected Select query"),
        }

        let q = parse_query("SELECT blocks WHERE level > 1").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions[0].op, CompareOp::Gt);
            }
            _ => panic!("expected Select query"),
        }

        let q = parse_query("SELECT blocks WHERE level < 3").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.conditions[0].op, CompareOp::Lt);
            }
            _ => panic!("expected Select query"),
        }
    }

    #[test]
    fn test_empty_input_returns_error() {
        let result = parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_select_with_limit_no_where() {
        let q = parse_query("SELECT documents LIMIT 20").unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(sel.entity, Entity::Documents);
                assert!(sel.conditions.is_empty());
                assert_eq!(sel.limit, Some(20));
            }
            _ => panic!("expected Select query"),
        }
    }
}
