use ndarray::{Array1, Array2};

pub struct Metrics;


impl Metrics {
    pub fn accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> f64 {
        let correct = y_true.iter()
            .zip(y_pred.iter())
            .filter(|(&true_val, &pred_val)| true_val == pred_val)
            .count();
        correct as f64 / y_true.len() as f64
    }

    pub fn precision_recall_f1(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> (f64, f64, f64) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
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
        
        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            cm[(true_val, pred_val)] += 1;
        }
        
        cm
    }
}

