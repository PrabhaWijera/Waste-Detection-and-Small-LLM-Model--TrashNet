# AI-Powered Waste Classification & Recommendation Backend

This is a production-ready Python backend (FastAPI) for a smart waste management system for Sri Lanka.
It auto-downloads the TrashNet dataset, fine-tunes a CNN (ResNet18), serves predictions, and stores usage in MongoDB.  

## Features
- 🚀 FastAPI REST endpoints for classification & recommendations
- 🧠 CNN fine-tuning on TrashNet (auto-download)
- 🧾 Rule-based, region-aware guidance (LLM-pluggable)
- 🗄️ MongoDB integration (submissions, analytics)
- 🔒 CORS configurable; environment-driven settings

## Structure
```
waste_management_system/
├── app/
│   ├── routes/
│   │   ├── user_routes.py          # /api endpoints for users
│   │   └── admin_routes.py         # /api/admin analytics
│   ├── models/
│   │   ├── waste_classifier.py     # loads model artifacts for inference
│   │   └── recommendation_engine.py
│   ├── database/
│   │   └── mongo_client.py         # MongoDB client
│   └── utils/
│       ├── preprocessing.py
│       └── helpers.py
├── scripts/
│   └── train_model.py              # download + train + save artifacts
├── models/
│   └── artifacts/                  # best_model.pt, label_map.json
├── data/                           # trashnet data (auto)
├── run.py                          # FastAPI app
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Train / Fine-tune (auto-downloads TrashNet)
```bash
python scripts/train_model.py --epochs 8 --batch_size 32
```

> If the GitHub URL rate-limits you, download TrashNet manually and place class folders under `data/trashnet/images/{cardboard,glass,metal,paper,plastic,trash}` then re-run the script.

3. Run API
```bash
uvicorn run:app --reload --port 8000
```

4. Test with cURL
```bash
curl -X POST "http://localhost:8000/api/classify"   -F "image=@/path/to/your/image.jpg"   -F "region=LK-11" -F "city=Colombo"
```

## Admin Analytics
- `GET /api/admin/metrics?days=30` — counts by class and city
- `GET /api/admin/submissions?page=1&page_size=20` — paginated submissions

## Notes
- Artifacts are saved to `models/artifacts/`. Make sure `label_map.json` and `best_model.pt` exist before running inference.
- You can plug in an LLM inference (OpenAI, etc.) inside `recommendation_engine.py` if needed.
- Update region/city-specific guidance rules as your municipality requires.
