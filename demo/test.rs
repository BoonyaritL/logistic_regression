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
        
        let (X_train, X_test, y_train, y_test) = 
            DataProcessor::train_test_split(&X, &y, 0.2);
        
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