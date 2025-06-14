import pickle
import sys
import os

def check_pickle_file(file_path):
    print(f"\nChecking pickle file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
            print(f"Type of content: {type(content)}")
            if isinstance(content, tuple):
                print(f"Number of items in tuple: {len(content)}")
                for i, item in enumerate(content):
                    print(f"\nItem {i}:")
                    print(f"Type: {type(item)}")
                    if hasattr(item, '__dict__'):
                        print("Attributes:", item.__dict__.keys())
                    elif hasattr(item, 'shape'):
                        print(f"Shape: {item.shape}")
                    else:
                        print("Content:", item)
            else:
                print("Content:", content)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

def main():
    # Check bureau model
    bureau_path = 'model/Rigorious/bureau_loan_repayment_model.pkl'
    if os.path.exists(bureau_path):
        check_pickle_file(bureau_path)
    else:
        print(f"\nBureau model file not found at: {bureau_path}")

    # Check income model
    income_path = 'model/Rigorious/income_prediction_model.pkl'
    if os.path.exists(income_path):
        check_pickle_file(income_path)
    else:
        print(f"\nIncome model file not found at: {income_path}")

if __name__ == "__main__":
    main() 