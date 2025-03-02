import os
import numpy as np

ucf_cls_ict = {
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "RoadAccidents",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
    "Normal_Videos_": "Testing_Normal_Videos_Anomaly",
    "Normal_Videos": "Training_Normal_Videos_Anomaly"
}

def split_and_save_features(input_folder, output_folder):
    # 遍历输入文件夹中的所有.npy文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            input_path = os.path.join(input_folder, filename)
            features = np.load(input_path)             # 加载原始特征文件
            video_name = filename[:-4]  # 去除文件扩展名获取视频名称
            class_name = None

            # 根据文件名查询字典获取对应的类别名称
            for key in ucf_cls_ict.keys():
                if key in filename:
                    class_name = ucf_cls_ict[key]
                    break

            if class_name is not None:
                # 创建存储拆分特征的子文件夹
                class_output_folder = os.path.join(output_folder, class_name)
                if not os.path.exists(class_output_folder):
                    os.makedirs(class_output_folder)

                # 加载原始特征文件
                features = np.load(input_path)

                # 拆分特征并保存为多个文件
                for i, feature in enumerate(features):
                    output_filename = f"{video_name}__{i}.npy"
                    output_path = os.path.join(class_output_folder, output_filename)
                    np.save(output_path, feature)
                    print(f"写入文件: {output_filename}, ID: {i+1}/{len(features)}")
            else:
                for i, feature in enumerate(features): #  无分类数据
                    output_filename = f"{video_name}__{i}.npy"
                    output_path = os.path.join(output_folder, output_filename)
                    np.save(output_path, feature)
                    print(f"写入文件: {output_filename}, ID: {i+1}/{len(features)}")

if __name__ == "__main__":
    input_folder = "/home/cw/sh/dataset/Crime/feature/8-26_8-22_SVMAE_wo-loss_CLIP-B"  # 输入文件夹路径
    output_folder = "/home/cw/sh/dataset/Crime/feature/8-26_8-22_SVMAE_wo-loss_CLIP-B_URDMU"  # 输出文件夹路径
    split_and_save_features(input_folder, output_folder)