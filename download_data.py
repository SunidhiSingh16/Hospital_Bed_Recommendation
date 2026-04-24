import os
from src.data_loader import generate_synthetic_data

def create_synthetic_data():
    """
    Generate and save synthetic hospital admissions data
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate synthetic data
    print("Generating synthetic hospital admissions data...")
    df = generate_synthetic_data()
    
    # Save processed dataset
    df.to_csv('data/hospital_admissions.csv', index=False)
    print(f"Dataset saved to data/hospital_admissions.csv")
    print(f"Total records: {len(df)}")
    print("\nSynthetic data includes:")
    print("- Daily admission patterns")
    print("- Weekly variations (weekday/weekend effects)")
    print("- Seasonal patterns")
    print("- Random special events")
    
if __name__ == "__main__":
    try:
        create_synthetic_data()
    except Exception as e:
        print(f"Error: {str(e)}") 