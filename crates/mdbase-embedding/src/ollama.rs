use mdbase_core::{MdbaseError, Result};
use serde::{Deserialize, Serialize};

use crate::provider::EmbeddingProvider;

const DEFAULT_MODEL: &str = "nomic-embed-text";
const DEFAULT_DIMENSION: usize = 768;
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Request body for the Ollama embed endpoint.
#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

/// Response from the Ollama embed endpoint.
#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Embedding provider backed by a local [Ollama](https://ollama.com) instance.
///
/// Uses the `/api/embed` endpoint, which supports batch input. Defaults to the
/// `nomic-embed-text` model (768 dimensions) on `localhost:11434`.
pub struct OllamaEmbedder {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
    dimension: usize,
}

impl OllamaEmbedder {
    /// Create with default settings: `nomic-embed-text` on `localhost:11434`
    /// (768 dimensions).
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            dimension: DEFAULT_DIMENSION,
        }
    }

    /// Create with a custom model and dimension, using the default base URL.
    pub fn with_model(model: String, dimension: usize) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model,
            dimension,
        }
    }

    /// Create with a custom base URL, model, and dimension.
    pub fn with_base_url(base_url: String, model: String, dimension: usize) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url,
            model,
            dimension,
        }
    }

    /// Call the Ollama embed API with the given texts.
    fn call_api(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| MdbaseError::EmbeddingError(format!("Ollama request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response
                .text()
                .unwrap_or_else(|_| "<could not read body>".to_string());
            return Err(MdbaseError::EmbeddingError(format!(
                "Ollama API returned {status}: {body_text}"
            )));
        }

        let resp: EmbedResponse = response
            .json()
            .map_err(|e| MdbaseError::EmbeddingError(format!("failed to parse response: {e}")))?;

        if resp.embeddings.len() != texts.len() {
            return Err(MdbaseError::EmbeddingError(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                resp.embeddings.len()
            )));
        }

        Ok(resp.embeddings)
    }
}

impl Default for OllamaEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for OllamaEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.call_api(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| MdbaseError::EmbeddingError("empty response from Ollama".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_construction() {
        let embedder = OllamaEmbedder::new();
        assert_eq!(embedder.model_id(), "nomic-embed-text");
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn default_trait() {
        let embedder = OllamaEmbedder::default();
        assert_eq!(embedder.model_id(), "nomic-embed-text");
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn custom_model() {
        let embedder = OllamaEmbedder::with_model("mxbai-embed-large".into(), 1024);
        assert_eq!(embedder.model_id(), "mxbai-embed-large");
        assert_eq!(embedder.dimension(), 1024);
    }

    #[test]
    fn custom_base_url() {
        let embedder =
            OllamaEmbedder::with_base_url("http://gpu-host:11434".into(), "test".into(), 512);
        assert_eq!(embedder.model_id(), "test");
        assert_eq!(embedder.dimension(), 512);
    }

    #[test]
    fn unreachable_api_returns_error() {
        let embedder =
            OllamaEmbedder::with_base_url("http://localhost:1".into(), "test".into(), 768);
        let result = embedder.embed("test");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("embedding error"),
            "unexpected error message: {err_msg}"
        );
    }

    #[test]
    fn unreachable_api_batch_returns_error() {
        let embedder =
            OllamaEmbedder::with_base_url("http://localhost:1".into(), "test".into(), 768);
        let result = embedder.embed_batch(&["hello", "world"]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_batch_returns_empty() {
        let embedder = OllamaEmbedder::new();
        let result = embedder.embed_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    #[ignore] // requires a running Ollama instance with nomic-embed-text
    fn live_embed() {
        let embedder = OllamaEmbedder::new();
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 768);
    }
}
