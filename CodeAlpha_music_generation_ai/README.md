# 1. Go to project root folder

cd path\to\project

# 2. Create virtual environment (one time only)

python -m venv venv

# 3. Activate virtual environment

.\venv\Scripts\Activate.ps1

# 4. Install dependencies (one time per venv)

python -m pip install -r requirements.txt

# 5. Go to source code folder

cd music_generation_ai

# 6. Prepare data (run once unless data changes)

python prepare_data.py

# 7. Train model

python train_model.py

# 8. Generate output

python generate_music.py

# 9. Exit virtual environment

deactivate
