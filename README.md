# Computer Vision API

A FastAPI application for computer vision tasks.

## Project Structure

```
computer vision/
├── app/
│   ├── __init__.py
│   ├── routers/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:zakriya057/Computer-vision.git
cd "computer vision"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative API documentation (ReDoc): `http://localhost:8000/redoc`

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check endpoint

## Development

To add new routes, create router files in the `app/routers/` directory and include them in `main.py`.

## License

MIT
