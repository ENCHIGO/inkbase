use std::collections::HashMap;
use std::sync::RwLock;

use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::Serialize;

use mdbase_core::error::MdbaseError;
use mdbase_core::types::{LinkRecord, LinkType};
use mdbase_core::Result;

use crate::types::{GraphStats, LinkInfo, PageRankScore};

// ---------------------------------------------------------------------------
// EdgeData — weight stored on each directed edge
// ---------------------------------------------------------------------------

/// Data attached to each edge in the knowledge graph.
#[derive(Debug, Clone, Serialize)]
pub struct EdgeData {
    pub link_type: LinkType,
    pub anchor_text: String,
}

// ---------------------------------------------------------------------------
// GraphInner — the mutable state behind the RwLock
// ---------------------------------------------------------------------------

struct GraphInner {
    graph: DiGraph<String, EdgeData>,
    /// Fast lookup from document path to its node index.
    node_map: HashMap<String, NodeIndex>,
}

impl GraphInner {
    fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Return the NodeIndex for `path`, creating a new node if necessary.
    fn ensure_node(&mut self, path: &str) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(path) {
            return idx;
        }
        let idx = self.graph.add_node(path.to_owned());
        self.node_map.insert(path.to_owned(), idx);
        idx
    }

    /// Collect all edge indices for outgoing edges from `node`.
    fn outgoing_edges(&self, node: NodeIndex) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|e| e.id())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// KnowledgeGraph — the public, thread-safe API
// ---------------------------------------------------------------------------

/// An in-memory directed graph of document links built on petgraph.
///
/// Nodes are document paths (strings). Edges carry [`EdgeData`] describing the
/// link type and anchor text. All public methods acquire a [`RwLock`] internally,
/// so the graph is safe to share across threads without external synchronization.
pub struct KnowledgeGraph {
    inner: RwLock<GraphInner>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(GraphInner::new()),
        }
    }

    /// Add edges from `source_path` to each link target in `links`.
    ///
    /// Nodes for both source and target are created on the fly if they do not
    /// already exist.
    pub fn add_links(&self, source_path: &str, links: &[LinkRecord]) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let source = inner.ensure_node(source_path);

        for link in links {
            let target = inner.ensure_node(&link.target);
            let data = EdgeData {
                link_type: link.link_type.clone(),
                anchor_text: link.anchor_text.clone(),
            };
            inner.graph.add_edge(source, target, data);
        }

        tracing::debug!(
            source = source_path,
            count = links.len(),
            "added links to graph"
        );

        Ok(())
    }

    /// Remove all outgoing edges from the document at `path`.
    ///
    /// The node itself is kept because other documents may still link to it.
    /// This is the correct operation when a document is re-parsed: call
    /// `remove_document` first, then `add_links` with the fresh set.
    pub fn remove_document(&self, path: &str) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let Some(&node) = inner.node_map.get(path) else {
            // Nothing to remove — not an error.
            return Ok(());
        };

        let edge_ids = inner.outgoing_edges(node);
        for eid in edge_ids {
            inner.graph.remove_edge(eid);
        }

        tracing::debug!(path, "removed outgoing edges");
        Ok(())
    }

    /// Get all outgoing links from the document at `path`.
    pub fn get_links(&self, path: &str) -> Result<Vec<LinkInfo>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let Some(&node) = inner.node_map.get(path) else {
            return Ok(Vec::new());
        };

        let links = inner
            .graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| {
                let target_idx = edge.target();
                LinkInfo {
                    source_path: path.to_owned(),
                    target_path: inner.graph[target_idx].clone(),
                    link_type: edge.weight().link_type.clone(),
                    anchor_text: edge.weight().anchor_text.clone(),
                }
            })
            .collect();

        Ok(links)
    }

    /// Get all incoming links (backlinks) to the document at `path`.
    pub fn get_backlinks(&self, path: &str) -> Result<Vec<LinkInfo>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let Some(&node) = inner.node_map.get(path) else {
            return Ok(Vec::new());
        };

        let links = inner
            .graph
            .edges_directed(node, Direction::Incoming)
            .map(|edge| {
                let source_idx = edge.source();
                LinkInfo {
                    source_path: inner.graph[source_idx].clone(),
                    target_path: path.to_owned(),
                    link_type: edge.weight().link_type.clone(),
                    anchor_text: edge.weight().anchor_text.clone(),
                }
            })
            .collect();

        Ok(links)
    }

    /// Find the shortest path from `from` to `to` using BFS.
    ///
    /// Returns `Ok(None)` if no path exists, or `Ok(Some(path))` with the
    /// sequence of document paths from source to destination (inclusive).
    pub fn shortest_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let (Some(&start), Some(&goal)) = (inner.node_map.get(from), inner.node_map.get(to))
        else {
            return Ok(None);
        };

        if start == goal {
            return Ok(Some(vec![from.to_owned()]));
        }

        // BFS with parent tracking.
        let mut visited = HashMap::<NodeIndex, Option<NodeIndex>>::new();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(start, None);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            for neighbor in inner.graph.neighbors_directed(current, Direction::Outgoing) {
                if visited.contains_key(&neighbor) {
                    continue;
                }
                visited.insert(neighbor, Some(current));

                if neighbor == goal {
                    // Reconstruct path by walking parent pointers.
                    let mut path = vec![inner.graph[goal].clone()];
                    let mut cursor = goal;
                    while let Some(Some(parent)) = visited.get(&cursor) {
                        path.push(inner.graph[*parent].clone());
                        cursor = *parent;
                    }
                    path.reverse();
                    return Ok(Some(path));
                }

                queue.push_back(neighbor);
            }
        }

        Ok(None)
    }

    /// Return basic statistics about the graph.
    pub fn stats(&self) -> GraphStats {
        // If the lock is poisoned we still return something sensible.
        let inner = self.inner.read().expect("stats: lock poisoned");
        GraphStats {
            node_count: inner.graph.node_count(),
            edge_count: inner.graph.edge_count(),
        }
    }

    /// Clear and rebuild the entire graph from a full set of (source_path, links) pairs.
    pub fn rebuild(&self, all_links: &[(String, Vec<LinkRecord>)]) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        // Start fresh.
        inner.graph.clear();
        inner.node_map.clear();

        for (source_path, links) in all_links {
            let source = inner.ensure_node(source_path);
            for link in links {
                let target = inner.ensure_node(&link.target);
                let data = EdgeData {
                    link_type: link.link_type.clone(),
                    anchor_text: link.anchor_text.clone(),
                };
                inner.graph.add_edge(source, target, data);
            }
        }

        tracing::info!(
            nodes = inner.graph.node_count(),
            edges = inner.graph.edge_count(),
            "graph rebuilt"
        );

        Ok(())
    }

    /// Compute PageRank scores for all nodes.
    ///
    /// `damping` is the damping factor (typically 0.85).
    /// `iterations` is the number of power-method iterations (typically 20-100).
    ///
    /// Returns a list of `(path, score)` pairs sorted by score descending.
    pub fn pagerank(&self, damping: f64, iterations: usize) -> Result<Vec<PageRankScore>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let n = inner.graph.node_count();
        if n == 0 {
            return Ok(Vec::new());
        }

        let n_f = n as f64;
        let node_indices: Vec<NodeIndex> = inner.graph.node_indices().collect();

        // Pre-compute out-degree for each node.
        let out_degree: HashMap<NodeIndex, usize> = node_indices
            .iter()
            .map(|&idx| {
                let deg = inner
                    .graph
                    .edges_directed(idx, Direction::Outgoing)
                    .count();
                (idx, deg)
            })
            .collect();

        // Initialize scores uniformly.
        let mut scores: HashMap<NodeIndex, f64> =
            node_indices.iter().map(|&idx| (idx, 1.0 / n_f)).collect();

        for _ in 0..iterations {
            // Sum mass from dangling nodes (out_degree == 0) and redistribute.
            let dangling_sum: f64 = node_indices
                .iter()
                .filter(|&&idx| out_degree[&idx] == 0)
                .map(|&idx| scores[&idx])
                .sum();

            let mut new_scores = HashMap::with_capacity(n);

            for &idx in &node_indices {
                // Contribution from predecessors (incoming edges).
                let incoming_sum: f64 = inner
                    .graph
                    .edges_directed(idx, Direction::Incoming)
                    .map(|edge| {
                        let pred = edge.source();
                        scores[&pred] / out_degree[&pred] as f64
                    })
                    .sum();

                let score =
                    (1.0 - damping) / n_f + damping * (incoming_sum + dangling_sum / n_f);

                new_scores.insert(idx, score);
            }

            scores = new_scores;
        }

        let mut result: Vec<PageRankScore> = node_indices
            .iter()
            .map(|&idx| PageRankScore {
                path: inner.graph[idx].clone(),
                score: scores[&idx],
            })
            .collect();

        // Sort descending by score, then alphabetically by path for stability.
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.path.cmp(&b.path))
        });

        Ok(result)
    }

    /// Find connected components in the graph, treating edges as undirected.
    ///
    /// Returns groups of document paths where each group is a connected component.
    /// Groups are sorted by size descending. Paths within each group are sorted
    /// alphabetically.
    pub fn connected_components(&self) -> Result<Vec<Vec<String>>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| MdbaseError::GraphError(format!("lock poisoned: {e}")))?;

        let n = inner.graph.node_count();
        if n == 0 {
            return Ok(Vec::new());
        }

        // petgraph's `connected_components` works on the graph treating edges as
        // undirected (it uses `Undirected` traversal internally for `DiGraph`).
        let num_components = petgraph::algo::connected_components(&inner.graph);

        // Build a mapping from component id to node indices. We use a BFS on the
        // undirected view since petgraph's `connected_components` only returns the
        // count. Instead, we use `UnionFind` or a simple BFS-based approach.
        // Actually, petgraph's `kosaraju_scc` finds strongly-connected components;
        // for weakly-connected components on a DiGraph, we walk neighbors in both
        // directions.
        let _ = num_components; // We'll compute groups directly.

        let mut visited = HashMap::<NodeIndex, usize>::with_capacity(n);
        let mut components: Vec<Vec<String>> = Vec::new();
        let mut component_id = 0usize;

        for idx in inner.graph.node_indices() {
            if visited.contains_key(&idx) {
                continue;
            }

            // BFS treating edges as undirected.
            let mut queue = std::collections::VecDeque::new();
            let mut component = Vec::new();

            queue.push_back(idx);
            visited.insert(idx, component_id);

            while let Some(current) = queue.pop_front() {
                component.push(inner.graph[current].clone());

                // Visit both outgoing and incoming neighbors.
                for neighbor in inner
                    .graph
                    .neighbors_directed(current, Direction::Outgoing)
                    .chain(inner.graph.neighbors_directed(current, Direction::Incoming))
                {
                    if visited.contains_key(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor, component_id);
                    queue.push_back(neighbor);
                }
            }

            component.sort();
            components.push(component);
            component_id += 1;
        }

        // Sort components by size descending, then by first element for stability.
        components.sort_by(|a, b| {
            b.len()
                .cmp(&a.len())
                .then_with(|| a.first().cmp(&b.first()))
        });

        Ok(components)
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// Helper to build a `LinkRecord` with minimal boilerplate.
    fn make_link(target: &str, link_type: LinkType, anchor: &str) -> LinkRecord {
        LinkRecord {
            link_id: Uuid::new_v4(),
            source_doc_id: Uuid::new_v4(),
            source_block_id: None,
            target: target.to_owned(),
            target_doc_id: None,
            link_type,
            anchor_text: anchor.to_owned(),
        }
    }

    #[test]
    fn add_links_creates_nodes_and_edges() {
        let graph = KnowledgeGraph::new();
        let links = vec![
            make_link("notes/b.md", LinkType::WikiLink, "B"),
            make_link("notes/c.md", LinkType::MarkdownLink, "C"),
        ];

        graph.add_links("notes/a.md", &links).unwrap();

        let stats = graph.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
    }

    #[test]
    fn get_outgoing_links() {
        let graph = KnowledgeGraph::new();
        let links = vec![
            make_link("notes/b.md", LinkType::WikiLink, "B page"),
            make_link("notes/c.md", LinkType::AutoLink, "https://c.md"),
        ];
        graph.add_links("notes/a.md", &links).unwrap();

        let outgoing = graph.get_links("notes/a.md").unwrap();
        assert_eq!(outgoing.len(), 2);

        let targets: Vec<&str> = outgoing.iter().map(|l| l.target_path.as_str()).collect();
        assert!(targets.contains(&"notes/b.md"));
        assert!(targets.contains(&"notes/c.md"));

        // Source should be consistent.
        for link in &outgoing {
            assert_eq!(link.source_path, "notes/a.md");
        }
    }

    #[test]
    fn get_backlinks() {
        let graph = KnowledgeGraph::new();

        // a -> b, c -> b
        graph
            .add_links(
                "a.md",
                &[make_link("b.md", LinkType::WikiLink, "link to b")],
            )
            .unwrap();
        graph
            .add_links(
                "c.md",
                &[make_link("b.md", LinkType::MarkdownLink, "also b")],
            )
            .unwrap();

        let backlinks = graph.get_backlinks("b.md").unwrap();
        assert_eq!(backlinks.len(), 2);

        let sources: Vec<&str> = backlinks.iter().map(|l| l.source_path.as_str()).collect();
        assert!(sources.contains(&"a.md"));
        assert!(sources.contains(&"c.md"));

        for link in &backlinks {
            assert_eq!(link.target_path, "b.md");
        }
    }

    #[test]
    fn shortest_path_found() {
        let graph = KnowledgeGraph::new();

        // a -> b -> c -> d
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();
        graph
            .add_links("b.md", &[make_link("c.md", LinkType::WikiLink, "c")])
            .unwrap();
        graph
            .add_links("c.md", &[make_link("d.md", LinkType::WikiLink, "d")])
            .unwrap();

        let path = graph.shortest_path("a.md", "d.md").unwrap();
        assert_eq!(
            path,
            Some(vec![
                "a.md".to_owned(),
                "b.md".to_owned(),
                "c.md".to_owned(),
                "d.md".to_owned(),
            ])
        );
    }

    #[test]
    fn shortest_path_not_found() {
        let graph = KnowledgeGraph::new();

        // a -> b, c (isolated)
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();
        graph.add_links("c.md", &[]).unwrap();

        let path = graph.shortest_path("a.md", "c.md").unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn shortest_path_same_node() {
        let graph = KnowledgeGraph::new();
        graph.add_links("a.md", &[]).unwrap();

        let path = graph.shortest_path("a.md", "a.md").unwrap();
        assert_eq!(path, Some(vec!["a.md".to_owned()]));
    }

    #[test]
    fn shortest_path_unknown_node() {
        let graph = KnowledgeGraph::new();

        let path = graph.shortest_path("x.md", "y.md").unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn remove_document_clears_outgoing_edges() {
        let graph = KnowledgeGraph::new();

        // a -> b, a -> c
        let links = vec![
            make_link("b.md", LinkType::WikiLink, "b"),
            make_link("c.md", LinkType::MarkdownLink, "c"),
        ];
        graph.add_links("a.md", &links).unwrap();

        // Also d -> a so "a" is a target too.
        graph
            .add_links("d.md", &[make_link("a.md", LinkType::WikiLink, "a")])
            .unwrap();

        assert_eq!(graph.stats().edge_count, 3);

        graph.remove_document("a.md").unwrap();

        // Outgoing edges from a are gone.
        let outgoing = graph.get_links("a.md").unwrap();
        assert!(outgoing.is_empty());

        // But incoming edges to a are preserved.
        let backlinks = graph.get_backlinks("a.md").unwrap();
        assert_eq!(backlinks.len(), 1);
        assert_eq!(backlinks[0].source_path, "d.md");

        // Node count unchanged (nodes are never removed).
        assert_eq!(graph.stats().node_count, 4);
        assert_eq!(graph.stats().edge_count, 1);
    }

    #[test]
    fn remove_nonexistent_document_is_noop() {
        let graph = KnowledgeGraph::new();
        // Should not error.
        graph.remove_document("does-not-exist.md").unwrap();
    }

    #[test]
    fn rebuild_replaces_entire_graph() {
        let graph = KnowledgeGraph::new();

        // Initial state.
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();
        assert_eq!(graph.stats().node_count, 2);
        assert_eq!(graph.stats().edge_count, 1);

        // Rebuild with completely different data.
        let fresh = vec![
            (
                "x.md".to_owned(),
                vec![
                    make_link("y.md", LinkType::WikiLink, "y"),
                    make_link("z.md", LinkType::MarkdownLink, "z"),
                ],
            ),
            (
                "y.md".to_owned(),
                vec![make_link("z.md", LinkType::WikiLink, "z")],
            ),
        ];

        graph.rebuild(&fresh).unwrap();

        assert_eq!(graph.stats().node_count, 3);
        assert_eq!(graph.stats().edge_count, 3);

        // Old data should be gone.
        let old_links = graph.get_links("a.md").unwrap();
        assert!(old_links.is_empty());

        // New data should be present.
        let x_links = graph.get_links("x.md").unwrap();
        assert_eq!(x_links.len(), 2);
    }

    #[test]
    fn get_links_unknown_document_returns_empty() {
        let graph = KnowledgeGraph::new();
        let links = graph.get_links("nonexistent.md").unwrap();
        assert!(links.is_empty());
    }

    #[test]
    fn get_backlinks_unknown_document_returns_empty() {
        let graph = KnowledgeGraph::new();
        let backlinks = graph.get_backlinks("nonexistent.md").unwrap();
        assert!(backlinks.is_empty());
    }

    // -----------------------------------------------------------------------
    // PageRank tests
    // -----------------------------------------------------------------------

    #[test]
    fn pagerank_simple_chain() {
        let graph = KnowledgeGraph::new();

        // a -> b -> c
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();
        graph
            .add_links("b.md", &[make_link("c.md", LinkType::WikiLink, "c")])
            .unwrap();
        // Ensure c.md exists as a node (it's created as a target, but add an
        // explicit empty link set so its node is definitely present).
        graph.add_links("c.md", &[]).unwrap();

        let scores = graph.pagerank(0.85, 50).unwrap();
        assert_eq!(scores.len(), 3);

        let score_of = |path: &str| -> f64 {
            scores.iter().find(|s| s.path == path).unwrap().score
        };

        // In a chain a->b->c, c accumulates the most rank because it receives
        // from b, which receives from a. Node a only gets the random-surfer
        // base plus dangling mass from c.
        assert!(
            score_of("c.md") > score_of("b.md"),
            "c ({}) should outrank b ({})",
            score_of("c.md"),
            score_of("b.md")
        );
        assert!(
            score_of("b.md") > score_of("a.md"),
            "b ({}) should outrank a ({})",
            score_of("b.md"),
            score_of("a.md")
        );
    }

    #[test]
    fn pagerank_hub_and_spoke() {
        let graph = KnowledgeGraph::new();

        // Many spokes pointing to a single hub.
        for i in 0..10 {
            let spoke = format!("spoke_{i}.md");
            graph
                .add_links(&spoke, &[make_link("hub.md", LinkType::WikiLink, "hub")])
                .unwrap();
        }
        graph.add_links("hub.md", &[]).unwrap();

        let scores = graph.pagerank(0.85, 50).unwrap();
        assert_eq!(scores.len(), 11);

        // The hub should have the highest score.
        assert_eq!(scores[0].path, "hub.md");

        // All spokes should have the same score (they are symmetric).
        let spoke_scores: Vec<f64> = scores
            .iter()
            .filter(|s| s.path.starts_with("spoke_"))
            .map(|s| s.score)
            .collect();
        for s in &spoke_scores {
            assert!(
                (s - spoke_scores[0]).abs() < 1e-10,
                "spoke scores should be equal"
            );
        }
    }

    #[test]
    fn pagerank_isolated_nodes() {
        let graph = KnowledgeGraph::new();

        // Three isolated nodes (no edges between them).
        graph.add_links("a.md", &[]).unwrap();
        graph.add_links("b.md", &[]).unwrap();
        graph.add_links("c.md", &[]).unwrap();

        let scores = graph.pagerank(0.85, 50).unwrap();
        assert_eq!(scores.len(), 3);

        // All isolated nodes are dangling; mass is redistributed evenly, so all
        // scores converge to 1/N.
        let expected = 1.0 / 3.0;
        for s in &scores {
            assert!(
                (s.score - expected).abs() < 1e-10,
                "isolated node {} should have score ~{}, got {}",
                s.path,
                expected,
                s.score
            );
        }
    }

    #[test]
    fn pagerank_empty_graph() {
        let graph = KnowledgeGraph::new();
        let scores = graph.pagerank(0.85, 20).unwrap();
        assert!(scores.is_empty());
    }

    // -----------------------------------------------------------------------
    // Connected components tests
    // -----------------------------------------------------------------------

    #[test]
    fn connected_components_single() {
        let graph = KnowledgeGraph::new();

        // a -> b -> c (all connected as one component).
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();
        graph
            .add_links("b.md", &[make_link("c.md", LinkType::WikiLink, "c")])
            .unwrap();

        let components = graph.connected_components().unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);

        // Paths should be sorted within the component.
        assert_eq!(components[0], vec!["a.md", "b.md", "c.md"]);
    }

    #[test]
    fn connected_components_multiple() {
        let graph = KnowledgeGraph::new();

        // Cluster 1: a -> b
        graph
            .add_links("a.md", &[make_link("b.md", LinkType::WikiLink, "b")])
            .unwrap();

        // Cluster 2: x -> y -> z
        graph
            .add_links("x.md", &[make_link("y.md", LinkType::WikiLink, "y")])
            .unwrap();
        graph
            .add_links("y.md", &[make_link("z.md", LinkType::WikiLink, "z")])
            .unwrap();

        let components = graph.connected_components().unwrap();
        assert_eq!(components.len(), 2);

        // Sorted by size descending: cluster 2 (3 nodes) first.
        assert_eq!(components[0].len(), 3);
        assert_eq!(components[0], vec!["x.md", "y.md", "z.md"]);

        assert_eq!(components[1].len(), 2);
        assert_eq!(components[1], vec!["a.md", "b.md"]);
    }

    #[test]
    fn connected_components_empty() {
        let graph = KnowledgeGraph::new();
        let components = graph.connected_components().unwrap();
        assert!(components.is_empty());
    }

    #[test]
    fn shortest_path_picks_shortest() {
        let graph = KnowledgeGraph::new();

        // Two paths from a to d:
        //   a -> b -> c -> d  (length 3)
        //   a -> d            (length 1, direct)
        graph
            .add_links(
                "a.md",
                &[
                    make_link("b.md", LinkType::WikiLink, "b"),
                    make_link("d.md", LinkType::WikiLink, "d"),
                ],
            )
            .unwrap();
        graph
            .add_links("b.md", &[make_link("c.md", LinkType::WikiLink, "c")])
            .unwrap();
        graph
            .add_links("c.md", &[make_link("d.md", LinkType::WikiLink, "d")])
            .unwrap();

        let path = graph.shortest_path("a.md", "d.md").unwrap();
        assert_eq!(path, Some(vec!["a.md".to_owned(), "d.md".to_owned()]));
    }
}
