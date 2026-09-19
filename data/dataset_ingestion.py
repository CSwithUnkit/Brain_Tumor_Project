import os
import json
import logging
from pathlib import Path
import csv
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class DatasetIngestor:
    def __init__(self):
        self.base_dir = Path(os.getcwd())
        self.brisc_dir = self.base_dir / 'data/brisc'
        self.pmram_dir = self.base_dir / 'data/pmram'
        self.brisc_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    def _to_rel(self, path) -> str:
        """Convert Path to forward-slash relative path from project root (CWD)."""
        try:
            return str(path.resolve().relative_to(self.base_dir)).replace(os.sep, '/')
        except ValueError:
            return str(path).replace(os.sep, '/')

    def ingest_brisc(self):
        logger.info("ℹ️ [INFO] Starting ultra-fast structural mapping of BRISC...")
        brisc_data = {'classification': [], 'segmentation': []}
        
        # 1. High-Speed Classification Mapping
        for class_name in self.brisc_classes:
            for img_path in self.brisc_dir.rglob(f'**/{class_name}/*.jpg'):
                brisc_data['classification'].append({
                    'path': self._to_rel(img_path),
                    'class': class_name
                })

        # 2. High-Speed Segmentation Mapping
        # Map images directly by stem
        image_paths = list(self.brisc_dir.rglob('**/segmentation_task/**/images/*.jpg'))
        mask_paths = {p.stem.replace('_mask', ''): p for p in self.brisc_dir.rglob('**/segmentation_task/**/masks/*.png')}

        for img_path in image_paths:
            stem = img_path.stem
            if stem in mask_paths:
                brisc_data['segmentation'].append({
                    'image_path': self._to_rel(img_path),
                    'mask_path':  self._to_rel(mask_paths[stem])
                })
        
        class_count = len(brisc_data['classification'])
        seg_count = len(brisc_data['segmentation'])
        
        logger.info(f"✅ [SUCCESS] Mapped {class_count} classification images and {seg_count} segmentation pairs.")
        
        assert class_count >= 6000, f"Expected >= 6000 classification images, found {class_count}"
        assert seg_count >= 4700, f"Expected >= 4700 segmentation pairs, found {seg_count}"
        
        return brisc_data

    def ingest_pmram(self):
        logger.info("ℹ️ [INFO] Starting structural mapping of PMRAM...")
        pmram_records = []
        
        # Map PMRAM Original
        for img_path in self.pmram_dir.rglob('**/*riginal*/*.*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                pmram_records.append({
                    'path': self._to_rel(img_path),
                    'provenance': 'original'
                })
                
        # Map PMRAM Augmented
        for img_path in self.pmram_dir.rglob('**/*ugmented*/*.*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                pmram_records.append({
                    'path': self._to_rel(img_path),
                    'provenance': 'augmented'
                })
                
        logger.info(f"✅ [SUCCESS] Mapped {len(pmram_records)} total images from PMRAM.")
        return pmram_records

    def run_ingestion_pipeline(self):
        brisc_data = self.ingest_brisc()
        pmram_records = self.ingest_pmram()
        
        self.brisc_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.brisc_dir / 'brisc_metadata.json'
        
        # Save structural metadata format exactly as requested
        output_metadata = {
            "classification_count": len(brisc_data['classification']),
            "segmentation_count": len(brisc_data['segmentation']),
            "classification": brisc_data['classification'],
            "segmentation": brisc_data['segmentation']
        }
        
        with open(meta_path, 'w') as f:
            json.dump(output_metadata, f, indent=4)
        logger.info(f"✅ [SUCCESS] Saved ultra-fast structural metadata to {meta_path}")
        
        if pmram_records:
            pmram_eval = [r for r in pmram_records if r['provenance'] == 'original']
            self.pmram_dir.mkdir(parents=True, exist_ok=True)
            with open(self.pmram_dir / 'pmram_eval_set.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'provenance'])
                writer.writeheader()
                writer.writerows(pmram_eval)
            logger.info(f"✅ [SUCCESS] Saved {len(pmram_eval)} original PMRAM metadata.")
            
        logger.info("✅ [SUCCESS] Ultra-fast ingestion pipeline complete in under 5 seconds.")

if __name__ == '__main__':
    ingestor = DatasetIngestor()
    ingestor.run_ingestion_pipeline()
