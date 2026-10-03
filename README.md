# Early Diabetes Detection — Deep Learning

This project trains a small PyTorch neural network on common diabetes screening measurements. It accepts the widely used Pima CSV layout and predicts the probability of a positive diabetes outcome.

## Setup

```powershell
cd diabetes_early_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Run with a Pima-style CSV containing these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

```powershell
python train.py --data path\to\diabetes.csv --output-dir artifacts
```

To verify the project without a CSV:

```powershell
python train.py --output-dir artifacts
```

## Predict one record

Values must follow the displayed feature order.

```powershell
python predict.py --model artifacts\diabetes_risk_model.pt --values 2 130 70 25 120 31.5 0.45 35
```

`Outcome=1` means diabetes was recorded in the source data; it is not a diagnosis. This project is for research and education and needs external clinical validation before any health-care use.
