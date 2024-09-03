**Chest X-Ray Analysis with Deep Learning**

This project utilizes deep learning models to analyze chest X-ray images and identify potential pathologies. It provides a web application where users can upload X-ray images and receive predictions about the presence of various conditions, along with visualization of the areas contributing to the predictions.

**Key Features:**

*   **Pneumonia and other pathology detection:** The application leverages a pre-trained DenseNet model to identify pathologies like pneumonia and other lung conditions.
*   **Attribution visualization:** Integrated Gradients are used to highlight the areas of the X-ray image that most strongly influence the model's predictions.
*   **User-friendly web interface:** The application offers an intuitive interface for uploading images and viewing results.

**Potential Applications:**

*   **Assisting radiologists in preliminary diagnosis:** The application can help radiologists quickly identify potential areas of concern in X-ray images, aiding in faster and more efficient diagnosis.
*   **Educational tool for medical professionals:** It can be used as a learning tool to understand how deep learning models analyze medical images and how attributions provide insights into the model's decision-making process.
*   **Research and development:** The codebase can serve as a foundation for further research and development in the field of medical image analysis.

**Disclaimer:**

This project is for educational and research purposes only and should not be used for clinical diagnosis. Always consult with a qualified medical professional for medical advice.

**Note:**

This project utilizes the `torchxrayvision` library for loading and preprocessing X-ray images and the `Captum` library for attribution visualization.
