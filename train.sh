cd federated
# Classification
python fed_train.py --log --data camelyon17
# Segmentation
python fed_train.py --log --data prostate --batch 16

