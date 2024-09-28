# Object-Detection-for-AIC

Note 1: Extract image_detection.zip and put image_detection.db same folder with this project or specify your path in `DB_NAME` inside `constants.py`.

Note 2: Replace `ROOT_FOLDER` with your images root folder, also inside `constants.py`.

Note 3: Color Detection feature is not completed. Please don't use it.

Note 4: If you have different Qdrant Client, like not use localhost but use cloud, please change `QDRANT_CLIENT` to your client in `constants.py`.

Note 5: This implementation use video root folder to use video seeker, and DJV to play video. If you don't have both, please don't use video seeker. If not, change `VIDEO_ROOT_FOLDER` to your video root folder in `constants.py`.

Run: `streamlit run app.py`

Reproduce environment: `pip install -r requirements.txt` **with Python 3.11.x**

Or install these packages: `pip install numpy pandas scikit-learn torch tqdm opencv-python streamlit transformers accelerate ultralytics qdrant_client` (just run and check if there is any missing package hihi)
