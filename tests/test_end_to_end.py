import pytest
import os
import json
import pandas as pd
from validation.external_pmram import PMRAMValidator
from evaluation.consolidate_reports import ReportConsolidator

def test_pmram_validation_metadata_filtering(tmp_path):
    """Test that PMRAMValidator strictly filters augmented images."""
    mock_meta_path = tmp_path / "pmram_metadata.csv"
    mock_metrics_path = tmp_path / "metrics.json"
    
    # Create mixed mock data
    df = pd.DataFrame({
        'path': ['1.jpg', '2.jpg', '3.jpg'],
        'provenance': ['original', 'augmented', 'original']
    })
    df.to_csv(mock_meta_path, index=False)
    
    with open(mock_metrics_path, 'w') as f:
        json.dump([{'val_metrics': {'accuracy': 0.9}}], f)
        
    validator = PMRAMValidator(
        model_path='dummy.pth',
        pmram_meta_path=str(mock_meta_path),
        brisc_metrics_path=str(mock_metrics_path)
    )
    
    filtered_df = validator.load_pmram_metadata()
    
    assert len(filtered_df) == 2
    assert all(filtered_df['provenance'] == 'original')

def test_report_consolidation_no_crash(tmp_path):
    """Verify that consolidate_reports runs fully and handles missing files gracefully."""
    # Switch working directory temporarily
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        os.makedirs('results', exist_ok=True)
        os.makedirs('reports/figures', exist_ok=True)
        
        # Inject mock data
        with open('results/metrics_exp1_baseline.json', 'w') as f:
            json.dump([{'val_metrics': {'accuracy': 0.85}}], f)
            
        consolidator = ReportConsolidator()
        consolidator.run()
        
        assert os.path.exists('reports/PHASE_I_EVALUATION_REPORT.md')
        assert os.path.exists('reports/PHASE_II_FINAL_REPORT.md')
        
        with open('reports/PHASE_I_EVALUATION_REPORT.md', 'r') as f:
            content = f.read()
            assert "85.00%" in content  # _pct() converts 0.85 → "85.00%"
            
    finally:
        os.chdir(original_cwd)
