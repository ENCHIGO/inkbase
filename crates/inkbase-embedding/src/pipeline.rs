use inkbase_core::types::BlockRecord;
use inkbase_core::Result;
use uuid::Uuid;

use crate::provider::EmbeddingProvider;
use crate::tfidf::TfIdfEmbedder;

/// Minimum text length (in characters) for a block to be worth embedding.
const MIN_BLOCK_TEXT_LEN: usize = 5;

/// Orchestration layer that wraps an [`EmbeddingProvider`] and exposes
/// convenient methods for embedding free text and document blocks.
pub struct EmbeddingPipeline {
    provider: Box<dyn EmbeddingProvider>,
}

impl EmbeddingPipeline {
    /// Create a pipeline backed by an arbitrary embedding provider.
    pub fn new(provider: Box<dyn EmbeddingProvider>) -> Self {
        Self { provider }
    }

    /// Create a pipeline backed by the default TF-IDF feature-hashing embedder
    /// with 256 dimensions.
    pub fn default_tfidf() -> Self {
        Self::new(Box::new(TfIdfEmbedder::default()))
    }

    /// The dimensionality of vectors produced by the underlying provider.
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    /// The model identifier reported by the underlying provider.
    pub fn model_id(&self) -> &str {
        self.provider.model_id()
    }

    /// Embed a single text string.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.provider.embed(text)
    }

    /// Embed multiple texts in one call.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.provider.embed_batch(texts)
    }

    /// Embed the text content of document blocks.
    ///
    /// Blocks whose `text_content` is shorter than [`MIN_BLOCK_TEXT_LEN`]
    /// characters (after trimming) are silently skipped -- they carry too
    /// little signal to produce a useful vector.
    ///
    /// Returns `(block_id, embedding)` pairs for every block that was embedded.
    pub fn embed_blocks(&self, blocks: &[BlockRecord]) -> Result<Vec<(Uuid, Vec<f32>)>> {
        let eligible: Vec<(&BlockRecord, &str)> = blocks
            .iter()
            .filter_map(|b| {
                let trimmed = b.text_content.trim();
                if trimmed.len() >= MIN_BLOCK_TEXT_LEN {
                    Some((b, trimmed))
                } else {
                    None
                }
            })
            .collect();

        if eligible.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<&str> = eligible.iter().map(|(_, t)| *t).collect();
        let vectors = self.provider.embed_batch(&texts)?;

        Ok(eligible
            .iter()
            .zip(vectors)
            .map(|((block, _), vec)| (block.block_id, vec))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use inkbase_core::types::{BlockRecord, BlockType};
    use uuid::Uuid;

    use super::*;

    fn make_block(text: &str) -> BlockRecord {
        BlockRecord {
            block_id: Uuid::new_v4(),
            doc_id: Uuid::new_v4(),
            block_type: BlockType::Paragraph,
            depth: 0,
            ordinal: 0,
            parent_block_id: None,
            text_content: text.to_string(),
            raw_markdown: text.to_string(),
        }
    }

    #[test]
    fn pipeline_default_tfidf_metadata() {
        let pipe = EmbeddingPipeline::default_tfidf();
        assert_eq!(pipe.dimension(), 256);
        assert_eq!(pipe.model_id(), "tfidf-hash-v1");
    }

    #[test]
    fn pipeline_embed_text_dimension() {
        let pipe = EmbeddingPipeline::default_tfidf();
        let vec = pipe.embed_text("hello world").unwrap();
        assert_eq!(vec.len(), 256);
    }

    #[test]
    fn pipeline_embed_texts_batch() {
        let pipe = EmbeddingPipeline::default_tfidf();
        let vecs = pipe
            .embed_texts(&["alpha beta", "gamma delta"])
            .unwrap();
        assert_eq!(vecs.len(), 2);
        assert!(vecs.iter().all(|v| v.len() == 256));
    }

    #[test]
    fn pipeline_embed_blocks_filters_short_text() {
        let pipe = EmbeddingPipeline::default_tfidf();
        let blocks = vec![
            make_block("hello world, this is a real block"),
            make_block("hi"),   // too short -- 2 chars
            make_block(""),     // empty
            make_block("    "), // whitespace only
            make_block("another meaningful block of text"),
        ];

        let results = pipe.embed_blocks(&blocks).unwrap();
        assert_eq!(results.len(), 2, "only 2 blocks should pass the length filter");
        assert_eq!(results[0].0, blocks[0].block_id);
        assert_eq!(results[1].0, blocks[4].block_id);
        assert!(results.iter().all(|(_, v)| v.len() == 256));
    }

    #[test]
    fn pipeline_embed_blocks_empty_input() {
        let pipe = EmbeddingPipeline::default_tfidf();
        let results = pipe.embed_blocks(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn pipeline_embed_blocks_all_filtered() {
        let pipe = EmbeddingPipeline::default_tfidf();
        let blocks = vec![make_block("ab"), make_block("cd")];
        let results = pipe.embed_blocks(&blocks).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn pipeline_custom_provider() {
        /// A trivial provider that returns all-ones vectors for testing.
        struct OnesProvider;

        impl EmbeddingProvider for OnesProvider {
            fn dimension(&self) -> usize {
                4
            }
            fn model_id(&self) -> &str {
                "ones-test"
            }
            fn embed(&self, _text: &str) -> inkbase_core::Result<Vec<f32>> {
                Ok(vec![1.0; 4])
            }
        }

        let pipe = EmbeddingPipeline::new(Box::new(OnesProvider));
        assert_eq!(pipe.dimension(), 4);
        assert_eq!(pipe.model_id(), "ones-test");

        let vec = pipe.embed_text("anything").unwrap();
        assert_eq!(vec, vec![1.0; 4]);
    }
}
