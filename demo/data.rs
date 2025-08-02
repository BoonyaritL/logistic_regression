use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::thread_rng;
use std::error::Error;
use csv::Reader;
use serde::Deserialize;
#[derive(Debug, Deserialize)]
pub struct DataRecord {
    #[serde(rename = "feature1")]
    pub feature1: f64,
    #[serde(rename = "feature2")]  
    pub feature2: f64,
    #[serde(rename = "label")]
    pub label: f64,
}

pub struct DataProcessor;

impl DataProcessor {
    // สร้างข้อมูลจำลอง
    pub fn generate_sample_data(n_samples: usize, n_features: usize, noise: f64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = thread_rng();
        
        // สร้าง features แบบสุ่ม
        let X = Array::random_using((n_samples, n_features), Uniform::new(-2.0, 2.0), &mut rng);
        
        // สร้าง labels โดยใช้ linear combination + noise
        let true_weights = Array1::random_using(n_features, Uniform::new(-1.0, 1.0), &mut rng);
        let true_bias = 0.5;
        
        let linear_combination = X.dot(&true_weights) + true_bias;
        let noise_array = Array1::random_using(n_samples, Uniform::new(-noise, noise), &mut rng);
        let probabilities = (linear_combination + noise_array).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        
        let y = probabilities.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
        
        (X, y)
    }

    // Normalize features (Z-score normalization)
    pub fn normalize(X: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let means = X.mean_axis(Axis(0)).unwrap();
        let stds = X.std_axis(Axis(0), 1.0);
        
        let X_normalized = (X - &means) / &stds;
        
        (X_normalized, means, stds)
    }

    // แบ่งข้อมูล train/test
    pub fn train_test_split(
        X: &Array2<f64>,
        y: &Array1<f64>,
        test_size: f64,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_samples = X.nrows();
        let n_test = (n_samples as f64 * test_size) as usize;
        let n_train = n_samples - n_test;

        // สร้าง indices แบบสุ่ม
        let mut indices: Vec<usize> = (0..n_samples).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut thread_rng());

        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        let X_train = X.select(Axis(0), train_indices);
        let X_test = X.select(Axis(0), test_indices);
        let y_train = y.select(Axis(0), train_indices);
        let y_test = y.select(Axis(0), test_indices);

        (X_train, X_test, y_train, y_test)
    }

    // โหลดข้อมูลจาก CSV
    pub fn load_from_csv_simple(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
        let mut reader = Reader::from_path(file_path)?;
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for result in reader.deserialize() {
            let record: DataRecord = result?;
            features.push(vec![record.feature1, record.feature2]);
            labels.push(record.label);
        }

        let n_samples = features.len();
        let n_features = if n_samples > 0 { features[0].len() } else { 0 };

        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for (i, (feature_row, label)) in features.iter().zip(labels.iter()).enumerate() {
            for (j, &feature_val) in feature_row.iter().enumerate() {
                X[(i, j)] = feature_val;
            }
            y[i] = *label;
        }

        Ok((X, y))
    }
}