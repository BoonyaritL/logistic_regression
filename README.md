# Logistic Regression Flow Chart in Rust

## ภาพรวม (Overview)

Logistic Regression เป็นอัลกอริทึมการเรียนรู้ของเครื่องสำหรับการจำแนกประเภท (Classification) ที่ใช้ Sigmoid Function เพื่อทำนายความน่าจะเป็นของแต่ละคลาส
<img width="355" height="336" alt="ภาพ" src="https://github.com/user-attachments/assets/071ef786-bc2c-421a-b388-a829c04135a2" />

## Flow Chart การทำงาน

```
┌─────────────────────┐
│   เริ่มต้น (Start)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  โหลดข้อมูล (Load)   │
│     - Features (X)  │
│     - Labels (y)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ เตรียมข้อมูล (Prep) │
│  - Normalize        │
│  - Split Train/Test │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ กำหนดค่าเริ่มต้น     │
│ - Weights (w) = 0   │
│ - Bias (b) = 0      │
│ - Learning Rate     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   เริ่ม Training    │
│   Loop (epochs)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Forward Pass       │
│  z = X·w + b        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Sigmoid Function  │
│ σ(z) = 1/(1+e^(-z)) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   คำนวณ Cost        │
│ J = -Σ[y·log(σ) +   │
│    (1-y)·log(1-σ)]  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  คำนวณ Gradients    │
│  dw = X^T·(σ - y)   │
│  db = Σ(σ - y)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   อัพเดต Parameters │
│  w = w - α·dw       │
│  b = b - α·db       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ตรวจสอบ           │
│   Convergence?      │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
   No▼         Yes▼
┌─────────┐ ┌─────────┐
│ Continue│ │  ทำนาย  │
│ Training│ │(Predict)│
└────┬────┘ └─────────┘
     │           │
     └───────────┘
                 ▼
        ┌─────────────────┐
        │   ประเมินผล     │
        │   - Accuracy    │
        │   - Precision   │
        │   - Recall      │
        └─────────────────┘
                 ▼
        ┌─────────────────┐
        │     จบ (End)    │
        └─────────────────┘
```

## โครงสร้างโค้ด Rust

### 1. Dependencies (Cargo.toml)
```toml
[dependencies]
ndarray = "0.15"
ndarray-rand = "0.14"
rand = "0.8"
csv = "1.1"
serde = { version = "1.0", features = ["derive"] }
plotters = "0.3"
```

### 2. Main Structure
```rust
use ndarray::{Array1, Array2, Axis};
use std::f64;

pub struct LogisticRegression {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
}
```

### 3. Key Functions

#### 3.1 Sigmoid Function
```rust
fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}
```

#### 3.2 Cost Function
```rust
fn compute_cost(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let m = y_true.len() as f64;
    let cost = -1.0 / m * (
        y_true * y_pred.mapv(|x| x.ln()) + 
        (1.0 - y_true) * (1.0 - y_pred).mapv(|x| x.ln())
    ).sum();
    cost
}
```

#### 3.3 Training Loop
```rust
impl LogisticRegression {
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let m = X.nrows() as f64;
        
        for iteration in 0..self.max_iterations {
            // Forward pass
            let z = X.dot(&self.weights) + self.bias;
            let predictions = sigmoid(&z);
            
            // Compute cost
            let cost = compute_cost(y, &predictions);
            
            // Compute gradients
            let dw = X.t().dot(&(predictions - y)) / m;
            let db = (predictions - y).sum() / m;
            
            // Update parameters
            self.weights = &self.weights - self.learning_rate * &dw;
            self.bias = self.bias - self.learning_rate * db;
            
            // Print progress
            if iteration % 100 == 0 {
                println!("Cost after iteration {}: {}", iteration, cost);
            }
        }
    }
}
```

#### 3.4 Prediction
```rust
impl LogisticRegression {
    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let z = X.dot(&self.weights) + self.bias;
        sigmoid(&z)
    }
    
    pub fn predict_classes(&self, X: &Array2<f64>) -> Array1<usize> {
        let probabilities = self.predict(X);
        probabilities.mapv(|p| if p >= 0.5 { 1 } else { 0 })
    }
}
```

## การใช้งาน (Usage)

### 1. สร้าง Model
```rust
let mut model = LogisticRegression {
    weights: Array1::zeros(n_features),
    bias: 0.0,
    learning_rate: 0.01,
    max_iterations: 1000,
};
```

### 2. Training
```rust
model.fit(&X_train, &y_train);
```

### 3. Prediction
```rust
let predictions = model.predict(&X_test);
let classes = model.predict_classes(&X_test);
```

### 4. Evaluation
```rust
fn accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64 {
    let correct = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(&true_val, &pred_val)| true_val == pred_val)
        .count();
    correct as f64 / y_true.len() as f64
}
```

## จุดเด่นของ Rust

1. **Performance**: ความเร็วสูงในการคำนวณ
2. **Memory Safety**: จัดการหน่วยความจำอย่างปลอดภัย
3. **Concurrency**: 
4. **Zero-cost Abstractions**: 
## ตัวอย่างการใช้งานจริง

### การโหลดข้อมูล CSV
```rust
use csv::Reader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DataPoint {
    feature1: f64,
    feature2: f64,
    label: usize,
}

fn load_data(filename: &str) -> (Array2<f64>, Array1<usize>) {
    // Implementation for loading CSV data
    // ...
}
```

### การแบ่งข้อมูล Train/Test
```rust
fn train_test_split(
    X: &Array2<f64>, 
    y: &Array1<usize>, 
    test_size: f64
) -> (Array2<f64>, Array2<f64>, Array1<usize>, Array1<usize>) {
    // Implementation for splitting data
    // ...
}
```

## การติดตั้งและรัน

```bash
# สร้างโปรเจค Rust ใหม่
cargo new logistic_regression
cd logistic_regression

# เพิ่ม dependencies ใน Cargo.toml
# แล้วรัน
cargo run
```

## ข้อควรพิจารณา

1. **Feature Scaling**: ควร normalize ข้อมูลก่อน training
2. **Regularization**: เพิ่ม L1/L2 regularization เพื่อป้องกัน overfitting
3. **Convergence**: ตรวจสอบการ converge ของ cost function
4. **Cross Validation**: ใช้ k-fold cross validation เพื่อประเมินโมเดล

## สรุป

Logistic Regression ใน Rust ให้ประสิทธิภาพสูงและความปลอดภัยในการจัดการหน่วยความจำ เหมาะสำหรับงานที่ต้องการความเร็วและความแม่นยำในการประมวลผล
