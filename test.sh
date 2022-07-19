cd federated
# Classification
python fed_train.py --test --test_path ../models/camelyon17/harmofl --data camelyon17
# Segmentation
python fed_train.py --test --test_path ../models/prostate/harmofl --data prostate


