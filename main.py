import pandas as pd
from src.data_processing import load_data, enhanced_preprocess_data, create_enhanced_features, prepare_train_test_data, scale_features
from src.models import train_ensemble_models
from src.evaluation import evaluate_model, plot_evaluation_results
from src.prediction import predict_future_demand

def main():
    print("1. 加载与预处理数据...")
    item_feature, _, meta_data = load_data('data/raw/')
    item_feature = enhanced_preprocess_data(item_feature, meta_data)
    
    print("2. 特征工程...")
    item_feature_enhanced = create_enhanced_features(item_feature)
    
    selected_features = [
        'brand_id', 'supplier_id', 'region_id', 'collect_uv', 'ss_effectiveness', 
        'cart_conversion_rate', 'collect_conversion_rate', 'price_indicator', 
        'item_avg_sales', 'cate_avg_sales', 'region_avg_sales', 
        'lag_1', 'lag_3', 'lag_7', 'rolling_mean_3', 'rolling_mean_14'
    ]
    target_column = 'qty_alipay'
    
    model_data = item_feature_enhanced.copy().fillna(item_feature_enhanced.median(numeric_only=True))
    model_data, scaler = scale_features(model_data, selected_features)
    train_data, test_data = prepare_train_test_data(model_data)
    
    X_train, y_train = train_data[selected_features], train_data[target_column]
    X_test, y_test = test_data[selected_features], test_data[target_column]
    
    print("3. 模型训练与评估...")
    rf_model, gb_model, ensemble_pred, _, _, rf_weight, gb_weight = train_ensemble_models(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, ensemble_pred, "集成模型")
    plot_evaluation_results(y_test, ensemble_pred, test_data, rf_model, gb_model, selected_features, rf_weight, gb_weight)
    
    print("4. 未来预测...")
    national_forecast_df, _ = predict_future_demand(model_data)
    national_forecast_df.to_csv('outputs/csv/未来两周全国总仓商品需求量变化趋势.csv', index=False)

if __name__ == "__main__":
    main()
