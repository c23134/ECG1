import wfdb
import numpy as np
import pandas as pd
import os

def extract_beat(record_name, annotation_label='N', output_dir="test_data"):
    """Extract beats from MIT-BIH records with proper annotation handling"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load signal and annotations
        record = wfdb.rdrecord(f'mitbih_database/{record_name}')
        annotation = wfdb.rdann(f'mitbih_database/{record_name}', 'atr')

        #############################################
        # TEMPORARY DEBUGGING CODE (DELETE LATER)
        print("\nDEBUGGING ANNOTATIONS:")
        print("All symbols:", annotation.symbol)
        print("Unique symbols:", set(annotation.symbol))
        print("Sample positions:", annotation.sample[:10])  # First 10 beat locations
        #############################################
        
        # Get valid beat indices (MIT-BIH uses symbols not aux_note)
        beat_indices = [
            ann_sample for ann_sample, symbol in zip(annotation.sample, annotation.symbol)
            if symbol == annotation_label
        ]
        
        # Extract 361-sample segments around R-peaks
        beats = []
        for i in beat_indices:
            start = i - 180
            end = i + 181
            if start >= 0 and end <= len(record.p_signal):
                beat = record.p_signal[start:end, 0]  # MLII lead (channel 0)
                beats.append(beat)
        
        # REPLACE THIS SECTION:
        # Save each beat as individual CSV file
        for idx, beat in enumerate(beats):
            output_path = f"{output_dir}/{record_name}_{annotation_label}_{idx}.csv"
            pd.DataFrame([beat]).to_csv(output_path, index=False, header=False)
            print(f"Saved beat {idx} to {output_path}")
            
        print(f"✅ Saved {len(beats)} individual beat files")
            
    except Exception as e:
        print(f"🔥 Error processing {record_name}: {str(e)}")

if __name__ == "__main__":
    #extract_beat('100', 'N')      # Normal beats
    #extract_beat('200', 'V')
    extract_beat('119', 'V')      # PVCs
    #extract_beat('209', 'A')      # Aberrated atrial premature