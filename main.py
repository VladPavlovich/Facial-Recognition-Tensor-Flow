import os
import sys

# Add the scripts directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import data_augmentation, preprocess, train, verify

# Define paths
POS_PATH = "data/positive"

# Check if the positive directory is empty or not
def is_directory_empty(path):
    return len(os.listdir(path)) == 0

# Run data augmentation if the positive directory is empty
if is_directory_empty(POS_PATH):
    data_augmentation.run()
    print("Data augmentation complete")
else:
    print("Data augmentation skipped. Positive directory is not empty.")

# Preprocess data
anchor, positive, negative, train_data, test_data = preprocess.run()
print("Data preprocessing complete")

# Train the model
train.run(train_data, test_data)
print("Model training complete")

# Verify using real-time video feed
verification_choice = input("Do you want to verify using real-time video feed? (Type 'yes' to verify): ").strip().lower()

if verification_choice == 'yes':
    verify.run()
    print("Verification complete")
else:
    print("Verification skipped.")
