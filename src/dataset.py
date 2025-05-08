from wildlife_datasets.datasets import AnimalCLEF2025
import pandas as pd
from torchvision.transforms import functional as TF

'''
AnimalCLEF2025 로드하고, query/database/calibration 분리까지 담당
'''

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, metadata: pd.DataFrame, image_dir: str, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        split = row.get("split", "database")  # default fallback
        img_path = os.path.join(self.image_dir, split, f"{image_id}.png")  # split 포함

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": row.get("identity", "unknown"),
            "image_id": image_id,
            "dataset": row["dataset"]
        }

    def get_subset(self, condition):
        subset_metadata = self.metadata[condition].reset_index(drop=True)
        return PreprocessedDataset(subset_metadata, self.image_dir, self.transform)

def salamander_orientation_transform(image, metadata):
    # Only apply to SalamanderID2025 dataset
    if metadata.get("dataset") == "SalamanderID2025":
        orientation = metadata.get("orientation", "top")
        # Align 'right' orientation to 'top' by rotating -90 degrees
        if orientation == "right":
            return TF.rotate(image, -90)
        # Align 'left' orientation to 'top' by rotating +90 degrees
        elif orientation == "left":
            return TF.rotate(image, 90)
        # 'top' orientation needs no change
    return image

def load_datasets(root, calibration_size=100):
    # Apply rotation transform for SalamanderID2025 samples during dataset loading
    dataset = AnimalCLEF2025(root, load_label=True, transform=salamander_orientation_transform)

    dataset.metadata["path"] = dataset.metadata.apply(
        lambda row: f"processed/{row['split']}/{row['image_id']}.png", axis=1
    )
    
    dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
    dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

    calib_meta = dataset_database.metadata[:calibration_size].copy()
    calib_meta["path"] = calib_meta.apply(
        lambda row: f"processed/database/{row['image_id']}.png", axis=1
    )
    dataset_calibration = AnimalCLEF2025(
        root, df=calib_meta, load_label=True, transform=salamander_orientation_transform
    )
    
    return dataset, dataset_database, dataset_query, dataset_calibration


# Return database and query datasets split by species
def load_datasets_by_species(root, calibration_size=100):
    dataset = AnimalCLEF2025(root, load_label=True)

    species_groups = {}
    for dataset_name in dataset.metadata['dataset'].unique():
        is_dataset = dataset.metadata['dataset'] == dataset_name
        db_df = dataset.metadata[is_dataset & (dataset.metadata['split'] == 'database')]
        print(f"[INFO] Dataset: {dataset_name} | Total DB samples: {len(db_df)}")
        query_df = dataset.metadata[is_dataset & (dataset.metadata['split'] == 'query')]

        '''
        calib의 역할
        * db에서 일정 개수만 샘플링한 부분집합
        * 유사도 score가 어느 정도면 믿을 수 있는지 학습하기 위한 용도
        * 현 코드에선 matcher의 점수 보정용으로 사용
        * validation dataset과 유사한 역할
        '''
        calib_df = db_df.sample(n=min(calibration_size, len(db_df)), random_state=42)
        db_df = db_df.drop(calib_df.index)
        dataset_db = AnimalCLEF2025(root, df=db_df, load_label=True)
        dataset_query = AnimalCLEF2025(root, df=query_df, load_label=True)

        dataset_calib = AnimalCLEF2025(root, df=calib_df, load_label=True)

        species_groups[dataset_name] = {
            'db': dataset_db,
            'query': dataset_query,
            'calib': dataset_calib
        }

    return species_groups

from src.transforms import transform  # 함수 상단에 추가

def load_datasets_with_processed(root, metadata_path="metadata.csv", calibration_size=100):
    metadata = pd.read_csv(os.path.join(root, metadata_path))
    processed_dir = os.path.join(root, "processed")

    df_db = metadata[metadata["split"] == "database"]
    df_query = metadata[metadata["split"] == "query"]
    df_calib = df_db.sample(n=min(calibration_size, len(df_db)), random_state=42)

    # 🔧 transform=transform 추가!
    dataset_db = PreprocessedDataset(df_db, processed_dir, transform=transform)
    dataset_query = PreprocessedDataset(df_query, processed_dir, transform=transform)
    dataset_calib = PreprocessedDataset(df_calib, processed_dir, transform=transform)

    return metadata, dataset_db, dataset_query, dataset_calib


