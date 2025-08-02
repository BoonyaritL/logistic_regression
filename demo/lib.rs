pub mod model;
pub mod data;
pub mod metrics;

pub use model::LogisticRegression;
pub use data::DataProcessor;
pub use metrics::Metrics;

#[derive(Debug)]
pub struct DataRecord {
    pub feature1: f64,
    pub feature2: f64,
    pub label: f64,
}