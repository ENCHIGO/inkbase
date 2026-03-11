use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    Select(SelectQuery),
    Search(SearchQuery),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectQuery {
    pub entity: Entity,
    pub conditions: Vec<Condition>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Entity {
    Documents,
    Blocks,
    Links,
    Tags,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub field: String,
    pub op: CompareOp,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(i64),
    Uuid(Uuid),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchQuery {
    pub search_type: SearchType,
    pub query_text: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchType {
    Fulltext,
    Semantic,
}
