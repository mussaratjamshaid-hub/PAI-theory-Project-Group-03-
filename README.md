# Email Spam Classifier

A modern web application for classifying emails as spam or legitimate using Machine Learning.

## Project Structure

```
├── app.py                 # Flask backend API
├── model_training.py      # Model training script
├── gui.py                 # Tkinter desktop GUI (optional)
├── dataset.csv            # Training dataset
├── templates/
│   └── index.html         # Interactive web interface
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

✨ **Interactive Web Interface**
- Clean, modern, responsive design
- Real-time email classification
- Visual confidence indicators
- Mobile-friendly layout

🔧 **RESTful API Backend**
- `/api/classify` - Classify email text
- `/api/status` - Check API and model status
- Error handling and validation

📊 **Machine Learning**
- TF-IDF vectorization
- Logistic Regression & SVM models
- Model persistence with pickle
- Confidence scoring

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
First time only - train the model using your dataset:
```bash
python model_training.py
```

This will create:
- `vectorizer.pkl` - Text vectorizer
- `model_lr.pkl` - Logistic Regression model
- `model_svm.pkl` - SVM model

### 3. Run the Application
```bash
python app.py
```

The application will start at: **http://127.0.0.1:5000**

## Usage

### Web Interface
1. Open browser to `http://127.0.0.1:5000`
2. Paste email text in the textarea
3. Click "Check Email" or press Ctrl+Enter
4. View results with confidence scores

### API Usage
```bash
# Classify an email
curl -X POST http://127.0.0.1:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your email text here"}'
```

Response:
```json
{
  "prediction": "Not Spam",
  "status": "✅",
  "confidence": 95.5,
  "spam_probability": 4.5
}
```

### Check API Status
```bash
curl http://127.0.0.1:5000/api/status
```

## Dataset Format

`dataset.csv` should have the following columns:
- `Category` - Label: "spam" or "ham"
- `Message` - Email text content

Example:
```csv
Category,Message
ham,This is a legitimate email
spam,Click here to win FREE money!!!
```

## Model Performance

The model is trained on TF-IDF vectorization with two algorithms:
- **Logistic Regression** - Default classifier (faster, good accuracy)
- **SVM** - Alternative classifier (higher accuracy, slower)

Check `model_training.py` output for accuracy metrics.

## Desktop GUI (Optional)

You can also use the Tkinter desktop interface:
```bash
python gui.py
```

## Troubleshooting

**Models not found error:**
- Run `python model_training.py` first
- Ensure `vectorizer.pkl` and `model_lr.pkl` exist

**Port already in use:**
- Modify port in `app.py`: `app.run(debug=True, port=5001)`

**CORS errors:**
- Flask-CORS is configured for all origins (development only)
- Restrict in production as needed

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: scikit-learn, pandas
- **Data Handling**: pickle, CSV

## Future Enhancements

- [ ] User authentication
- [ ] Email history storage
- [ ] Advanced model ensemble
- [ ] Real-time training updates
- [ ] Docker containerization
- [ ] Database integration (SQLite/PostgreSQL)

## License

Open source project - Free to use and modify

## Author

Created for email spam classification task
