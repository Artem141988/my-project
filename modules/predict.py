import os
import json
import dill
import pandas as pd
from datetime import datetime
from glob import glob

def load_latest_model():
    """Загружает последнюю обученную модель"""
    model_dir = os.path.join(os.environ.get('PROJECT_PATH', '.'), 'data/models')
    model_files = glob(os.path.join(model_dir, 'cars_pipe_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No models found in {model_dir}")
    
    latest_model = max(model_files, key=os.path.getctime)
    with open(latest_model, 'rb') as f:
        model = dill.load(f)
    
    print(f"Loaded model: {os.path.basename(latest_model)}")
    return model

def load_test_data():
    """Загружает все тестовые данные из папки test"""
    test_dir = os.path.join(os.environ.get('PROJECT_PATH', '.'), 'data/test')
    test_files = glob(os.path.join(test_dir, '*.json'))
    
    if not test_files:
        raise FileNotFoundError(f"No test files found in {test_dir}")
    
    test_data = []
    for file in test_files:
        with open(file) as f:
            data = json.load(f)
            data['car_id'] = os.path.splitext(os.path.basename(file))[0]
            test_data.append(data)
    
    return pd.DataFrame(test_data)

def save_predictions(predictions):
    """Сохраняет предсказания в CSV файл"""
    pred_dir = os.path.join(os.environ.get('PROJECT_PATH', '.'), 'data/predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    pred_file = os.path.join(pred_dir, f'preds_{timestamp}.csv')
    
    predictions.to_csv(pred_file, index=False)
    print(f"Predictions saved to {pred_file}")

def predict():
    """Основная функция для выполнения предсказаний"""
    try:
        # 1. Загружаем модель
        model = load_latest_model()
        
        # 2. Загружаем тестовые данные
        test_df = load_test_data()
        
        # 3. Делаем предсказания
        predictions = model.predict(test_df)
        
        # 4. Формируем DataFrame с результатами
        results = pd.DataFrame({
            'car_id': test_df['car_id'],
            'pred': predictions
        })
        
        # 5. Сохраняем результаты
        save_predictions(results)
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    predict()