from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
from model_utils import load_and_preprocess_image, get_predictions, compute_attributions, visualize_attribution, model 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="./templates")


@app.post("/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Analyze the uploaded image and return predictions and visualization."""
    try:
        # Read image file
        image_bytes = io.BytesIO(await file.read())

        # Process the image
        img = load_and_preprocess_image(image_bytes)

        # Get predictions
        predictions = get_predictions(img)
        predictions = {k: float(v) for k, v in predictions.items()}

        # Get the top prediction
        top_pathology = max(predictions, key=predictions.get)
        top_probability = predictions[top_pathology]

        # Compute attributions
        target_class = model.pathologies.index(top_pathology) 
        attributions = compute_attributions(img, target_class)

        # Visualize the attribution
        visualization = visualize_attribution(img, attributions, top_pathology)

        response_data = {
            "predictions": predictions,
            "top_pathology": top_pathology,
            "top_probability": float(top_probability),
            "visualization": visualization
        }
        return JSONResponse(content=response_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Update uvicorn.run command