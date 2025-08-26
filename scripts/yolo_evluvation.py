import yaml
from pathlib import Path

yaml_file = Path(r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\data\archive\dataset_split\waste_data.yaml")
with open(yaml_file) as f:
    data = yaml.safe_load(f)

print("Train folder exists:", Path(data["train"]).exists())
print("Val folder exists:", Path(data["val"]).exists())
