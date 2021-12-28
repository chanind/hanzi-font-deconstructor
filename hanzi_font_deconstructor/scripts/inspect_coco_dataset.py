import fiftyone as fo


# [fo.load_dataset(d).delete() for d in fo.list_datasets()]
dataset = fo.Dataset.from_dir(
    data_path="/Users/davidchanin/dev/hanzi-font-deconstructor/data/coco",
    labels_path="/Users/davidchanin/dev/hanzi-font-deconstructor/data/coco/data.json",
    dataset_type=fo.types.COCODetectionDataset,
    label_types=["detections", "segmentations"],
    # name="data",
)
fo.launch_app(dataset)
