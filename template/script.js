const fileInput = document.getElementById('fileInput');
const analyzeButton = document.getElementById('analyzeButton');
const imagePreview = document.getElementById('imagePreview');
const resultList = document.getElementById('resultList');
const attributionImage = document.getElementById('attributionImage');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const topPathologyDiv = document.getElementById('topPathology');

const API_URL = 'http://localhost:8000/analyze'; // Constant for API URL

fileInput.addEventListener('change', handleFileInputChange);
analyzeButton.addEventListener('click', analyzeImage);

function handleFileInputChange(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" class="max-w-full h-auto rounded-lg" alt="X-ray preview">`;
        };
        reader.readAsDataURL(file);
    }
}

async function analyzeImage() {
    const file = fileInput.files[0];
    if (!file) {
        displayErrorMessage('Please select a file');
        return;
    }

    showLoadingIndicator();
    clearErrorMessage();

    try {
        const uploadedFile = await uploadFile(file);
        const analysisData = await processResponse(uploadedFile);
        updateUI(analysisData);
    } catch (error) {
        handleError(error);
    } finally {
        hideLoadingIndicator();
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Network response was not ok (status: ${response.status})`);
        }

        return response;
    } catch (error) {
        throw new Error(`Error uploading file: ${error.message}`);
    }
}

async function processResponse(response) {
    try {
        const data = await response.json();

        if (!data.predictions || typeof data.predictions !== 'object') {
            throw new Error('Invalid response format: predictions are missing or invalid');
        }

        return data;
    } catch (error) {
        throw new Error(`Error processing server response: ${error.message}`);
    }
}

function updateUI(analysisData) {
    displayResults(analysisData.predictions);
    displayAttributionImage(analysisData.visualization);
    displayTopPathology(analysisData.top_pathology, analysisData.top_probability);
}

function handleError(error) {
    console.error('Error:', error);
    displayErrorMessage(`An error occurred: ${error.message}`);
}

function displayResults(results) {
    resultList.innerHTML = '';

    const sortedResults = Object.entries(results)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

    sortedResults.forEach(([pathology, probability]) => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';

        const resultLabel = document.createElement('div');
        resultLabel.className = 'result-label';
        resultLabel.textContent = pathology;
        resultItem.appendChild(resultLabel);

        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';
        const barFill = document.createElement('div');
        barFill.className = 'bar-fill';
        barFill.style.width = `${(probability * 100).toFixed(1)}%`;
        barContainer.appendChild(barFill);
        resultItem.appendChild(barContainer);

        const resultPercentage = document.createElement('div');
        resultPercentage.className = 'result-percentage';
        resultPercentage.textContent = `${(probability * 100).toFixed(1)}%`;
        resultItem.appendChild(resultPercentage);

        resultList.appendChild(resultItem);
    });
}

function displayAttributionImage(base64Image) {
    if (base64Image) {
        attributionImage.src = `data:image/png;base64,${base64Image}`;
        attributionImage.style.display = 'block';
    } else {
        attributionImage.style.display = 'none';
    }
}

function displayTopPathology(pathology, probability) {
    topPathologyDiv.textContent = `${pathology} (${(probability * 100).toFixed(1)}%)`;
}

function showLoadingIndicator() {
    loadingIndicator.style.display = 'block';
}

function hideLoadingIndicator() {
    loadingIndicator.style.display = 'none';
}

function displayErrorMessage(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function clearErrorMessage() {
    errorMessage.style.display = 'none';
}