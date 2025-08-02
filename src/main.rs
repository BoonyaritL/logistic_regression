use csv::Reader;
use ndarray::{s, Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::thread_rng;
use serde::Deserialize;
use std::error::Error;
use std::f64;

// ข้อมูลสำหรับโหลดจาก CSV
#[derive(Debug, Deserialize)]
struct DataRecord {
    #[serde(rename = "feature1")]
    feature1: f64,
    #[serde(rename = "feature2")]
    feature2: f64,
    #[serde(rename = "label")]
    label: f64,
}

// โครงสร้างหลักของ Logistic Regression
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub cost_history: Vec<f64>,
}

impl LogisticRegression {
    // สร้าง instance ใหม่
    pub fn new(n_features: usize, learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
            learning_rate,
            max_iterations,
            tolerance: 1e-6,
            cost_history: Vec::new(),
        }
    }

    // Sigmoid function
    fn sigmoid(&self, z: &Array1<f64>) -> Array1<f64> {
        z.mapv(|x| {
            if x > 500.0 {
                1.0
            } else if x < -500.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        })
    }

    // คำนวณ cost function (Binary Cross Entropy)
    fn compute_cost(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let m = y_true.len() as f64;
        let epsilon = 1e-15; // ป้องกันการหาร log(0)

        let y_pred_clipped = y_pred.mapv(|x| x.max(epsilon).min(1.0 - epsilon));

        let cost = -1.0 / m
            * (y_true * y_pred_clipped.mapv(|x| x.ln())
                + (1.0 - y_true) * (1.0 - &y_pred_clipped).mapv(|x| x.ln()))
            .sum();

        cost
    }

    // Training function
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<(), Box<dyn Error>> {
        let m = X.nrows() as f64;
        let mut prev_cost = f64::INFINITY;

        println!("Starting training...");
        println!("Features: {}, Samples: {}", X.ncols(), X.nrows());

        for iteration in 0..self.max_iterations {
            // Forward pass
            let z = X.dot(&self.weights) + self.bias;
            let predictions = self.sigmoid(&z);

            // คำนวณ cost
            let cost = self.compute_cost(y, &predictions);
            self.cost_history.push(cost);

            // คำนวณ gradients
            let error = &predictions - y;
            let dw = X.t().dot(&error) / m;
            let db = error.sum() / m;

            // อัพเดท parameters
            self.weights = &self.weights - self.learning_rate * &dw;
            self.bias = self.bias - self.learning_rate * db;

            // แสดงผลความคืบหน้า
            if iteration % 100 == 0 {
                println!("Iteration {}: Cost = {:.6}", iteration, cost);
            }

            // ตรวจสอบ convergence
            if (prev_cost - cost).abs() < self.tolerance {
                println!("Converged at iteration {}", iteration);
                break;
            }
            prev_cost = cost;
        }

        println!("Training completed!");
        Ok(())
    }

    // ทำนายความน่าจะเป็น
    pub fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let z = X.dot(&self.weights) + self.bias;
        self.sigmoid(&z)
    }

    // ทำนายคลาส
    pub fn predict(&self, X: &Array2<f64>, threshold: f64) -> Array1<usize> {
        let probabilities = self.predict_proba(X);
        probabilities.mapv(|p| if p >= threshold { 1 } else { 0 })
    }

    // คำนวณ accuracy - แก้ไข syntax error
    pub fn score(&self, X: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let predictions = self.predict(X, 0.5);
        let y_classes = y.mapv(|x| x as usize);

        let correct = predictions
            .iter()
            .zip(y_classes.iter())
            .filter(|(pred, true_val)| **pred == **true_val) // แก้ไขตรงนี้
            .count();

        correct as f64 / y.len() as f64
    }
}

// ฟังก์ชันสำหรับการประเมินผล
pub struct Metrics;

impl Metrics {
    // แก้ไข accuracy function
    pub fn accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(true_val, pred_val)| **true_val == **pred_val) // แก้ไขตรงนี้
            .count();
        correct as f64 / y_true.len() as f64
    }

    pub fn precision_recall_f1(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> (f64, f64, f64) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (*true_val, *pred_val) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_count += 1,
                _ => {}
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }

    pub fn confusion_matrix(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Array2<usize> {
        let mut cm = Array2::zeros((2, 2));

        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            cm[(*true_val, *pred_val)] += 1;
        }

        cm
    }
}

// ฟังก์ชันสำหรับการเตรียมข้อมูล
pub struct DataProcessor;

impl DataProcessor {
    // สร้างข้อมูลจำลอง
    pub fn generate_sample_data(
        n_samples: usize,
        n_features: usize,
        noise: f64,
    ) -> (Array2<f64>, Array1<f64>) {
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

    // สร้างไฟล์ CSV
    pub fn create_sample_csv(
        filename: &str,
        n_samples: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rand::{thread_rng, Rng};
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;
        writeln!(file, "feature1,feature2,label")?;

        let mut rng = thread_rng();

        for _ in 0..n_samples {
            let feature1: f64 = rng.gen_range(-3.0..3.0);
            let feature2: f64 = rng.gen_range(-3.0..3.0);

            // สร้าง label ด้วยสมการเชิงเส้น + noise
            let noise: f64 = rng.gen_range(-0.3..0.3);
            let linear_combo = feature1 * 0.7 + feature2 * 1.1 + 0.2 + noise;
            let probability = 1.0 / (1.0 + (-linear_combo).exp());
            let label = if probability > 0.5 { 1.0 } else { 0.0 };

            writeln!(file, "{:.4},{:.4},{}", feature1, feature2, label)?;
        }

        println!("สร้างไฟล์ {} พร้อม {} ตัวอย่างแล้ว", filename, n_samples);
        Ok(())
    }

    // โหลดข้อมูลจาก CSV แบบง่าย
    pub fn load_from_csv_simple(
        file_path: &str,
    ) -> Result<(Array2<f64>, Array1<f64>), Box<dyn std::error::Error>> {
        use csv::Reader;

        // ถ้าไฟล์ไม่มี ให้สร้างใหม่
        if !std::path::Path::new(file_path).exists() {
            println!("ไม่พบไฟล์ {} กำลังสร้างใหม่...", file_path);
            Self::create_sample_csv(file_path, 1000)?;
        }

        let mut reader = Reader::from_path(file_path)?;
        let mut features = Vec::new();
        let mut labels = Vec::new();

        // อ่าน CSV แบบไม่ใช้ struct
        for result in reader.records() {
            let record = result?;

            // อ่านค่าจากแต่ละ column
            let feature1: f64 = record[0].parse()?;
            let feature2: f64 = record[1].parse()?;
            let label: f64 = record[2].parse()?;

            features.push(vec![feature1, feature2]);
            labels.push(label);
        }

        let n_samples = features.len();
        let n_features = 2; // feature1, feature2

        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for (i, (feature_row, label)) in features.iter().zip(labels.iter()).enumerate() {
            X[(i, 0)] = feature_row[0];
            X[(i, 1)] = feature_row[1];
            y[i] = *label;
        }

        println!(
            "โหลดข้อมูลจาก {} สำเร็จ: {} ตัวอย่าง, {} features",
            file_path, n_samples, n_features
        );

        Ok((X, y))
    }

    // โหลดข้อมูลจาก CSV (แบบเดิม - ใช้ struct)
    pub fn load_from_csv(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
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

// ฟังก์ชัน main สำหรับทดสอบ
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Logistic Regression Demo ===\n");

    // สร้างโฟลเดอร์ data ถ้าไม่มี
    std::fs::create_dir_all("data")?;

    // 1. ทดสอบข้อมูลจำลอง
    println!("1. Testing with generated data...");
    let (X_gen, y_gen) = DataProcessor::generate_sample_data(1000, 2, 0.1);
    println!(
        "Generated data shape: {} samples, {} features",
        X_gen.nrows(),
        X_gen.ncols()
    );

    // 2. ทดสอบ CSV
    println!("\n2. Testing with CSV data...");
    let csv_file = "data/sample_data.csv";
    let (X_csv, y_csv) = DataProcessor::load_from_csv_simple(csv_file)?;
    println!(
        "CSV data shape: {} samples, {} features",
        X_csv.nrows(),
        X_csv.ncols()
    );

    // 3. เลือกใช้ข้อมูลจำลอง (เร็วกว่า)
    println!("\n3. Training model with generated data...");
    let (X, y) = (X_gen, y_gen);

    // Normalize ข้อมูล
    let (X_normalized, means, stds) = DataProcessor::normalize(&X);
    println!("Data normalized with means: {:?}", means);
    println!("Standard deviations: {:?}", stds);

    // แบ่งข้อมูล train/test
    let (X_train, X_test, y_train, y_test) =
        DataProcessor::train_test_split(&X_normalized, &y, 0.2);
    println!("Train set: {} samples", X_train.nrows());
    println!("Test set: {} samples", X_test.nrows());

    // สร้างและ train model
    let mut model = LogisticRegression::new(X_train.ncols(), 0.01, 1000);
    model.fit(&X_train, &y_train)?;

    // ทำนายและประเมินผล
    let train_accuracy = model.score(&X_train, &y_train);
    let test_accuracy = model.score(&X_test, &y_test);

    println!("\n4. Results:");
    println!("Train Accuracy: {:.4}", train_accuracy);
    println!("Test Accuracy: {:.4}", test_accuracy);

    // แสดงผล detailed metrics
    println!("\n5. Detailed Metrics:");
    let y_test_classes = y_test.mapv(|x| x as usize);
    let test_predictions = model.predict(&X_test, 0.5);

    let (precision, recall, f1) = Metrics::precision_recall_f1(&y_test_classes, &test_predictions);
    println!("Precision: {:.4}", precision);
    println!("Recall: {:.4}", recall);
    println!("F1-Score: {:.4}", f1);

    // Confusion Matrix
    println!("\n6. Confusion Matrix:");
    let cm = Metrics::confusion_matrix(&y_test_classes, &test_predictions);
    println!("     Pred");
    println!("     0  1");
    println!("True 0 {} {}", cm[(0, 0)], cm[(0, 1)]);
    println!("     1 {} {}", cm[(1, 0)], cm[(1, 1)]);

    // แสดง learned parameters
    println!("\n7. Learned Parameters:");
    println!("Weights: {:?}", model.weights);
    println!("Bias: {:.6}", model.bias);

    // แสดง cost history (ตัวอย่าง 10 ค่าสุดท้าย)
    println!("\n8. Cost History (last 10 values):");
    let history_len = model.cost_history.len();
    let start_idx = if history_len > 10 {
        history_len - 10
    } else {
        0
    };
    for (i, &cost) in model.cost_history[start_idx..].iter().enumerate() {
        println!("  Step {}: {:.6}", start_idx + i, cost);
    }

    // ทดสอบการทำนายแต่ละตัวอย่าง
    println!("\n9. Sample Predictions:");
    let sample_predictions = model.predict_proba(&X_test.slice(s![0..5, ..]).to_owned());
    for i in 0..5.min(X_test.nrows()) {
        println!(
            "Sample {}: Probability = {:.4}, True = {}",
            i, sample_predictions[i], y_test[i]
        );
    }

    // สร้าง CSV ตัวอย่างเพิ่มเติม
    println!("\n10. Creating additional CSV files...");
    DataProcessor::create_sample_csv("data/small_dataset.csv", 100)?;
    DataProcessor::create_sample_csv("data/medium_dataset.csv", 500)?;
    DataProcessor::create_sample_csv("data/large_dataset.csv", 2000)?;

    println!("\n=== Demo completed successfully! ===");
    println!("Created CSV files:");
    println!("- data/sample_data.csv (1000 samples)");
    println!("- data/small_dataset.csv (100 samples)");
    println!("- data/medium_dataset.csv (500 samples)");
    println!("- data/large_dataset.csv (2000 samples)");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let model = LogisticRegression::new(1, 0.01, 100);
        let z = Array1::from_vec(vec![0.0, 1.0, -1.0, 10.0, -10.0]);
        let result = model.sigmoid(&z);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!(result[1] > 0.5);
        assert!(result[2] < 0.5);
        assert!(result[3] > 0.99);
        assert!(result[4] < 0.01);
    }

    #[test]
    fn test_data_generation() {
        let (X, y) = DataProcessor::generate_sample_data(100, 3, 0.1);
        assert_eq!(X.nrows(), 100);
        assert_eq!(X.ncols(), 3);
        assert_eq!(y.len(), 100);

        // Check that labels are binary
        for &label in y.iter() {
            assert!(label == 0.0 || label == 1.0);
        }
    }

    #[test]
    fn test_train_test_split() {
        let X = Array2::zeros((100, 2));
        let y = Array1::zeros(100);

        let (X_train, X_test, y_train, y_test) = DataProcessor::train_test_split(&X, &y, 0.2);

        assert_eq!(X_train.nrows(), 80);
        assert_eq!(X_test.nrows(), 20);
        assert_eq!(y_train.len(), 80);
        assert_eq!(y_test.len(), 20);
    }

    #[test]
    fn test_metrics() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0, 1, 0, 0]);
        let y_pred = Array1::from_vec(vec![1, 0, 1, 0, 0, 1, 0, 1]);

        let accuracy = Metrics::accuracy(&y_true, &y_pred);
        assert!((accuracy - 0.75).abs() < 1e-10);

        let (precision, recall, f1) = Metrics::precision_recall_f1(&y_true, &y_pred);
        assert!(precision > 0.0);
        assert!(recall > 0.0);
        assert!(f1 > 0.0);
    }
}
