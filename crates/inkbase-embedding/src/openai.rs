use inkbase_core::{InkbaseError, Result};
use serde::{Deserialize, Serialize};

use crate::provider::EmbeddingProvider;

const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIMENSION: usize = 1536;
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Request body for the OpenAI embeddings endpoint.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

/// A single embedding entry in the API response.
#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Top-level response from the OpenAI embeddings endpoint.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// OpenAI embedding provider using the text-embedding-3-small model by default.
///
/// Calls the `/v1/embeddings` endpoint. Compatible with any API that speaks the
/// same protocol (Azure OpenAI, LiteLLM proxy, etc.) via [`with_base_url`](Self::with_base_url).
pub struct OpenAiEmbedder {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    dimension: usize,
    base_url: String,
}

impl OpenAiEmbedder {
    /// Create a new OpenAI embedder with the default model (`text-embedding-3-small`,
    /// 1536 dimensions).
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
            dimension: DEFAULT_DIMENSION,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create with a custom model name and dimension.
    pub fn with_model(api_key: String, model: String, dimension: usize) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            api_key,
            model,
            dimension,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create with a custom base URL (for OpenAI-compatible APIs).
    ///
    /// The embedder will POST to `{base_url}/v1/embeddings`. Uses the default
    /// model and dimension.
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
            dimension: DEFAULT_DIMENSION,
            base_url,
        }
    }

    /// Call the OpenAI embeddings API with the given texts.
    fn call_api(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| InkbaseError::EmbeddingError(format!("OpenAI request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response
                .text()
                .unwrap_or_else(|_| "<could not read body>".to_string());
            return Err(InkbaseError::EmbeddingError(format!(
                "OpenAI API returned {status}: {body_text}"
            )));
        }

        let resp: EmbeddingResponse = response
            .json()
            .map_err(|e| InkbaseError::EmbeddingError(format!("failed to parse response: {e}")))?;

        if resp.data.len() != texts.len() {
            return Err(InkbaseError::EmbeddingError(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                resp.data.len()
            )));
        }

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

impl EmbeddingProvider for OpenAiEmbedder {
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
            .ok_or_else(|| InkbaseError::EmbeddingError("empty response from OpenAI".to_string()))
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
    fn construction_defaults() {
        let embedder = OpenAiEmbedder::new("test-key".into());
        assert_eq!(embedder.model_id(), "text-embedding-3-small");
        assert_eq!(embedder.dimension(), 1536);
    }

    #[test]
    fn custom_model() {
        let embedder =
            OpenAiEmbedder::with_model("key".into(), "text-embedding-ada-002".into(), 1536);
        assert_eq!(embedder.model_id(), "text-embedding-ada-002");
        assert_eq!(embedder.dimension(), 1536);
    }

    #[test]
    fn custom_base_url() {
        let embedder = OpenAiEmbedder::with_base_url("key".into(), "http://my-proxy:8080".into());
        assert_eq!(embedder.model_id(), "text-embedding-3-small");
        assert_eq!(embedder.dimension(), 1536);
    }

    #[test]
    fn unreachable_api_returns_error() {
        let embedder = OpenAiEmbedder::with_base_url("key".into(), "http://localhost:1".into());
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
        let embedder = OpenAiEmbedder::with_base_url("key".into(), "http://localhost:1".into());
        let result = embedder.embed_batch(&["hello", "world"]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_batch_returns_empty() {
        let embedder = OpenAiEmbedder::new("key".into());
        let result = embedder.embed_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    #[ignore] // requires a valid OPENAI_API_KEY
    fn live_embed() {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for live tests");
        let embedder = OpenAiEmbedder::new(api_key);
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 1536);
    }
}
