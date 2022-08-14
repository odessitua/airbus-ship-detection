
# Put kaggle.json to C:\Users\<Windows-username>\.kaggle\kaggle.json
# or ~/.kaggle/kaggle.json for unix system

!pip install kaggle
# !mkdir -p ~/.kaggle
# !cp {path_kaggle_json} ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c airbus-ship-detection
!unzip airbus-ship-detection.zip

# Output:
# test_v2 (folder)
# train_v2 (folder)
# sample_submission_v2.csv
# train_ship_segmentations_v2.csv