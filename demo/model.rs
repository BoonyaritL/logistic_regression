use ndarray::{Array1, Array2};
use std::error::Error;

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

    // คำนวณ accuracy
    pub fn score(&self, X: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let predictions = self.predict(X, 0.5);
        let y_classes = y.mapv(|x| x as usize);

        let correct = predictions
            .iter()
            .zip(y_classes.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        correct as f64 / y.len() as f64
    }
}
