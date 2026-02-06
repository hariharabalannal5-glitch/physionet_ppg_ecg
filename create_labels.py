"""
Create REFERENCE.csv Label File for PhysioNet Challenge 2015
==============================================================
This script helps you create the label file (REFERENCE.csv) needed for training.

The labels can be obtained from:
1. The sample entry answers.txt file from PhysioNet
2. Manual annotation based on the alarm types

Author: Your Team
Date: 2024
"""

import os
import requests
import zipfile
import io


def download_sample_entry_labels():
    """
    Download the sample entry from PhysioNet which contains answers.txt
    with labels for all training records
    """
    
    print("\n" + "="*60)
    print("Downloading Sample Entry from PhysioNet...")
    print("="*60 + "\n")
    
    # URL for the sample entry
    sample_entry_url = "https://physionet.org/files/challenge-2015/1.0.0/entry.zip"
    
    try:
        print(f"📥 Downloading from: {sample_entry_url}")
        response = requests.get(sample_entry_url, timeout=30)
        response.raise_for_status()
        
        print("✓ Download complete!")
        print("📦 Extracting files...")
        
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Look for answers.txt
            if 'answers.txt' in zip_ref.namelist():
                answers_content = zip_ref.read('answers.txt').decode('utf-8')
                
                # Save as REFERENCE.csv
                output_file = 'training/REFERENCE.csv'
                os.makedirs('training', exist_ok=True)
                
                print(f"📝 Creating {output_file}...")
                
                with open(output_file, 'w') as f:
                    f.write("record_name,label\n")
                    
                    for line in answers_content.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Format: record_name,label
                            f.write(f"{line}\n")
                
                print(f"✅ Successfully created {output_file}!")
                print(f"   Labels for {len(answers_content.strip().split(chr(10)))} records")
                return True
            else:
                print("⚠️  answers.txt not found in the sample entry")
                return False
                
    except Exception as e:
        print(f"❌ Error downloading sample entry: {e}")
        print("\nYou can manually download the sample entry from:")
        print("https://physionet.org/files/challenge-2015/1.0.0/entry.zip")
        print("Extract it and find answers.txt, then create REFERENCE.csv manually")
        return False


def create_manual_reference_template():
    """
    Create a template REFERENCE.csv file that users can fill in manually
    """
    
    print("\n" + "="*60)
    print("Creating Manual REFERENCE.csv Template...")
    print("="*60 + "\n")
    
    # Get list of all .mat files in training directory
    training_dir = 'training'
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        print("   Please make sure you have downloaded the training data")
        return False
    
    mat_files = sorted([f[:-4] for f in os.listdir(training_dir) if f.endswith('.mat')])
    
    if not mat_files:
        print(f"❌ No .mat files found in {training_dir}")
        return False
    
    output_file = os.path.join(training_dir, 'REFERENCE_TEMPLATE.csv')
    
    print(f"📝 Creating template: {output_file}")
    print(f"   Found {len(mat_files)} records")
    
    with open(output_file, 'w') as f:
        f.write("record_name,label\n")
        f.write("# Fill in the label column with:\n")
        f.write("#   1 = True alarm (genuine arrhythmia)\n")
        f.write("#   0 = False alarm\n")
        f.write("#\n")
        
        for record in mat_files[:10]:  # Show first 10 as examples
            f.write(f"{record},  # <- Fill this in\n")
        
        if len(mat_files) > 10:
            f.write(f"# ... and {len(mat_files) - 10} more records\n")
    
    print(f"\n✅ Template created: {output_file}")
    print("\n⚠️  You need to manually fill in the labels!")
    print("   After filling in, rename to REFERENCE.csv")
    
    return True


def verify_reference_file():
    """
    Verify that REFERENCE.csv exists and is properly formatted
    """
    
    print("\n" + "="*60)
    print("Verifying REFERENCE.csv...")
    print("="*60 + "\n")
    
    ref_file = 'training/REFERENCE.csv'
    
    if not os.path.exists(ref_file):
        print(f"❌ {ref_file} not found")
        return False
    
    print(f"✓ Found {ref_file}")
    
    # Read and verify format
    labels = {}
    with open(ref_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Skip header
        if i == 0 and line.startswith('record'):
            continue
        
        # Parse record,label
        parts = line.split(',')
        if len(parts) >= 2:
            record_name = parts[0].strip()
            label = parts[1].strip()
            
            if label in ['0', '1']:
                labels[record_name] = int(label)
            else:
                print(f"⚠️  Line {i+1}: Invalid label '{label}' for record '{record_name}'")
    
    print(f"\n✅ Valid labels found: {len(labels)}")
    print(f"   True alarms (1): {sum(1 for v in labels.values() if v == 1)}")
    print(f"   False alarms (0): {sum(1 for v in labels.values() if v == 0)}")
    
    return len(labels) > 0


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print(" PhysioNet Challenge 2015 - Label File Creator")
    print("="*70)
    
    print("\nThis script helps you create the REFERENCE.csv file containing")
    print("labels (True/False alarm) for all training records.")
    print("\nChoose an option:")
    print("  1. Download labels from PhysioNet sample entry (RECOMMENDED)")
    print("  2. Create a manual template (you fill in labels yourself)")
    print("  3. Verify existing REFERENCE.csv file")
    print("  4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            success = download_sample_entry_labels()
            if success:
                verify_reference_file()
            break
        
        elif choice == '2':
            create_manual_reference_template()
            break
        
        elif choice == '3':
            verify_reference_file()
            break
        
        elif choice == '4':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print("\n" + "="*70)
    print("After creating REFERENCE.csv, run: python extract_signals.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()