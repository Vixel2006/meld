import os
import sys
from pathlib import Path
import json
import shutil

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.datasets import DataConfig, Flickr30kDataset, COCODataset, verify_dataset

def test_flickr_paths():
    print("Testing Flickr30k paths...")
    test_root = Path("/tmp/boc_data_test")
    if test_root.exists():
        shutil.rmtree(test_root)
    
    flickr_dir = test_root / "flickr30k"
    ann_dir = flickr_dir / "flickr30k_annotations"
    img_dir = flickr_dir / "flickr30k_images"
    
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    
    # Create a dummy image
    (img_dir / "test.jpg").touch()
    
    # Create dummy annotation
    with open(ann_dir / "train.json", 'w') as f:
        json.dump([{"image_id": "test.jpg", "captions": ["test caption"]}], f)
    
    config = DataConfig(dataset_name='flickr30k', data_root=str(test_root), split='train')
    
    # Verify via helper
    if not verify_dataset('flickr30k', test_root):
        print("❌ verify_dataset failed for Flickr30k")
        return False
    
    # Verify via class initialization
    try:
        ds = Flickr30kDataset(config)
        print(f"✓ Flickr30kDataset initialized. Annotation file: {ds.annotation_file}")
        if "flickr30k/flickr30k_annotations" not in str(ds.annotation_file):
            print(f"❌ Unexpected annotation path: {ds.annotation_file}")
            return False
    except Exception as e:
        print(f"❌ Flickr30kDataset initialization failed: {e}")
        return False
    
    return True

def test_coco_paths():
    print("\nTesting COCO paths...")
    test_root = Path("/tmp/boc_data_test_coco")
    if test_root.exists():
        shutil.rmtree(test_root)
    
    coco_dir = test_root / "coco"
    ann_dir = coco_dir / "annotations"
    img_dir = coco_dir / "train2017"
    
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    
    # Create dummy annotation
    with open(ann_dir / "captions_train2017.json", 'w') as f:
        json.dump({"images": [{"id": 1, "file_name": "test.jpg"}], "annotations": [{"image_id": 1, "caption": "test"}]}, f)
    
    config = DataConfig(dataset_name='coco', data_root=str(test_root), split='train')
    
    # Verify via helper
    if not verify_dataset('coco', test_root):
        print("❌ verify_dataset failed for COCO")
        return False
    
    # Verify via class initialization
    try:
        ds = COCODataset(config)
        print(f"✓ COCODataset initialized. Annotation file: {ds.annotation_file}")
        if "coco/annotations" not in str(ds.annotation_file):
            print(f"❌ Unexpected annotation path: {ds.annotation_file}")
            return False
    except Exception as e:
        print(f"❌ COCODataset initialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    f_ok = test_flickr_paths()
    c_ok = test_coco_paths()
    
    if f_ok and c_ok:
        print("\n✅ All path verification tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
