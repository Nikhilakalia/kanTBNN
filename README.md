Setting Up the Environment

1. Clone the Repository

Navigate to the desired directory and clone the repository:

cd /path/to/your/directory
git clone https://github.com/your-repo-url/kanTBNN.git
cd kanTBNN

cd /path/to/your/directory
git clone https://github.com/your-repo-url/kanTBNN.git
cd kanTBNN

2. Create and Activate a Virtual Environment

It is recommended to create a virtual environment to avoid conflicts with system-wide Python packages:
python3 -m venv kan_project
source kan_project/bin/activate

3. Install Dependencies

Install the required Python packages:
pip install -r requirements.txt
Alternatively, manually install packages if requirements.txt is not present:
pip install numpy pandas torch matplotlib scipy optuna

Running the Flat Plate Case

1. Navigate to the Configuration Directory

The configuration file for the flat plate case is located in the config directory. Navigate there:
cd /path/to/kanTBNN/config


2. Prepare Input Data

Ensure that the input data required for the flat plate case is available in the data directory. If not, create the necessary files or folders and ensure the following:

Input data file (csv file shared on One Drive) is formatted as required by the kanTBNN framework.

Data should be structured with relevant features and target values for training.

3. Run the Training Script

Use the following command to start the training process:
python3 kanTBNN_training_run_config.py FLAT_PLATE_config
Replace FLAT_PLATE_config with the actual configuration file name if different.

4. Output Files

The following output files will be generated in the models or output directory:

Logs: Training and evaluation logs (e.g., kanTBNN_flat_plate.log).

Model: Saved model files (e.g., flat_plate_model.pth).

Plots: Loss curves and evaluation plots (e.g., flat_plate_loss.png).

