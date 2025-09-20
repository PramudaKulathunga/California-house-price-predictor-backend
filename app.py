from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Configure CORS for production
allowed_origins = os.environ.get('ALLOWED_ORIGINS', '').split(',')
if not any(allowed_origins):
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://californiahouseprize.vercel.app",
        "https://*.netlify.app"
    ]

print(f"Allowed origins: {allowed_origins}")

CORS(app, origins=allowed_origins, supports_credentials=True)


# Add OPTIONS handler for preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load model and scaler with error handling
try:
    model = joblib.load('models/california_housing_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None


def create_features(input_data):
    """Create engineered features from raw input"""
    features = input_data.copy()
    
    # Add engineered features (same as during training)
    features['rooms_per_household'] = features['total_rooms'] / features['households']
    features['bedrooms_per_room'] = features['total_bedrooms'] / features['total_rooms']
    features['population_per_household'] = features['population'] / features['households']
    
    return features


@app.route('/')
def home():
    return jsonify({
        'message': 'California Housing Price Prediction API',
        'status': 'healthy',
        'model_loaded': model is not None,
        'endpoints': {
            'GET /': 'API information',
            'POST /predict': 'Predict house price',
            'GET /health': 'Health check',
            'GET /features': 'List of features used'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        }), 500
        
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Prepare basic features
        basic_features = np.array([[
            float(data['longitude']),
            float(data['latitude']),
            float(data['housing_median_age']),
            float(data['total_rooms']),
            float(data['total_bedrooms']),
            float(data['population']),
            float(data['households']),
            float(data['median_income']),
            int(data['ocean_proximity'])
        ]])
        
        # Create DataFrame with correct column names
        feature_names = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity'
        ]
        
        df_basic = pd.DataFrame(basic_features, columns=feature_names)
        
        # Add engineered features
        df_with_engineered = create_features(df_basic.iloc[0])
        
        # Convert to array in correct order (same as training)
        expected_features = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity', 'rooms_per_household', 'bedrooms_per_room',
            'population_per_household'
        ]
        
        final_features = np.array([df_with_engineered[expected_features].values])
        
        # Scale features
        features_scaled = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        predicted_price = prediction * 100000  # Convert to actual dollars
        
        return jsonify({
            'success': True,
            'prediction': float(predicted_price),
            'formatted_prediction': f"${predicted_price:,.2f}",
            'features_used': expected_features,
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Ensure all 9 basic features are provided correctly'
        }), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'California Housing API',
        'timestamp': pd.Timestamp.now().isoformat()
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to see expected features"""
    expected_features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity', 'rooms_per_household', 'bedrooms_per_room',
        'population_per_household'
    ]
    return jsonify({
        'expected_features': expected_features,
        'input_features': 9,
        'engineered_features': 3,
        'total_features': 12,
        'note': 'Frontend only needs to provide the first 9 features'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'POST /predict',
            'GET /health',
            'GET /features'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
