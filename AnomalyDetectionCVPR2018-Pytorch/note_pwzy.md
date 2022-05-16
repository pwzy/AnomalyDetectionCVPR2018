# 从原始视频中提取特征
# python feature_extractor.py --dataset_path ../../dataset/UCF-Crime_unzip --model_type 'c3d' --pretrained_3d  ./c3d.pickle

# 下载别人已经提取好的特征
Can be downloaded from: https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing

# 基于提取好的特征进行训练 
# python TrainingAnomalyDetector_public.py --features_path ../../dataset/UCF-Crime_unzip/pretrained_feature --annotation_path ./Train_Annotation.txt

# Generate ROC Curve
# python generate_ROC.py --features_path ../../dataset/UCF-Crime_unzip/pretrained_feature --annotation_path Test_Annotation.txt --model_path exps/models/epoch_80000.pt

--features_path is the path of Precomputed Features;
--annotation_path is the annoation of video clip;
--model_path is the mode path








