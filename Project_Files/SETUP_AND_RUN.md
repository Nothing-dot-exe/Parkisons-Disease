# Setup and Execution Guide

*Developed by [Piyush Kadam (@piyushkadam96k)](https://github.com/piyushkadam96k)*

This guide will walk you through exactly how to set up your computer and run the Parkinson's Disease Detection Machine Learning pipeline.

---

## 1. Prerequisites (Python Installation)

Before running the code, you need to have **Python** installed on your system. 

*   **Recommended Version:** Python 3.8, 3.9, 3.10, or 3.11 (Avoid extremely new Beta versions like 3.13 if some libraries aren't fully tested yet).
*   **Download Link:** [Download Python from python.org](https://www.python.org/downloads/)
*   **Crucial Step During Installation:** When you run the Python installer, make sure you **CHECK THE BOX** at the very bottom that says `"Add python.exe to PATH"`. If you forget this, the command line won't recognize the `python` command!

---

## 2. Installing Requirements

The project relies on specific Data Science and Machine Learning libraries (like `scikit-learn`, `xgboost`, and `pandas`). 
I have created a `requirements.txt` file for you to make this process incredibly easy.

1. Open your terminal or Command Prompt (cmd).
2. Use `cd` to navigate into your project folder. For example:
   ```cmd
   cd C:\Users\kadam\OneDrive\Parkinson-Disease-Detection\Project_Files
   ```
3. Type the following command and hit Enter to install all dependencies automatically:
   ```cmd
   pip install -r requirements.txt
   ```

---

## 3. How to Run the Project

Once the installation from Step 2 is complete, you are ready to run the scripts.
Ensure your terminal is still navigated to the `Project_Files` folder.

### A. To Train the Machine Learning Models:
Execute the main pipeline. This will download the remote dataset, process the math, train the models on all your CPU cores, print out the metrics, and dump the models into the `ML_Models` folder.
```cmd
python main.py
```
*(Wait a couple of minutes for this to finish, as it is mathematically intensive!)*

### B. To Generate the PowerPoint Presentation:
After the models have trained and the metrics exist, you can instantly turn those findings into a highly-formatted `.pptx` slide deck.
```cmd
python create_ppt.py
```
*(You will see `Parkinsons_Project_Presentation.pptx` appear in the folder!)*
