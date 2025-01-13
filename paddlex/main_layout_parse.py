# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path
import img2pdf
from PIL import Image
import tempfile
import glob
import cv2
import json
import sys
from .inference.pipelines_new  import create_pipeline

def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    os.makedirs(path)  # 重新创建空目录

def clear_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted file: {path}")

def prepare_directories(save_dir):
    paths = ['imgs', 'images', 'jsons','gt_jsons']
    for dir_name in paths:
        clear_directory(os.path.join(save_dir, dir_name))

def process_image_file(image_file, pipeline, save_dir, is_save_gt=True, is_eval=False, gt_json_path=None, is_only_xycut=False,start_id=0,annotations_path=False):
    page_files = pipeline.predict(image_file,start_id=start_id,annotations_path=annotations_path)
    base_name = os.path.basename(image_file)
    file_name = os.path.splitext(base_name)[0]
    gt_path = f"{save_dir}/gt_jsons/{file_name}.json"

    total_bleu_score, total_ard, total_tau = 0, 0, 0
    for res in page_files:
        if is_save_gt:
            res.save_gt_json(gt_path, is_eval=True, is_only_xycut=is_only_xycut)
        if is_eval and gt_json_path:
            bleu_score, ard, tau = res.eval_layout_ordering(gt_json_path=gt_json_path)
            total_bleu_score += bleu_score
            total_ard += ard
            total_tau += tau
        res.save_to_ordering(f"{save_dir}/ordering", is_eval=is_eval, is_only_xycut=is_only_xycut)
        res.save_to_img(f"{save_dir}/images")
        res.save_to_json(f"{save_dir}/jsons")

    return total_bleu_score, total_ard, total_tau

def png_to_pdf(png_files):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        pdf_path = temp_pdf.name
        temp_image_files = []

        for image_file in png_files:
            with Image.open(image_file) as img:
                img = img.rotate(0, expand=True)
                temp_image_path = image_file + "_temp.png"
                img.save(temp_image_path)
                temp_image_files.append(temp_image_path)

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(temp_image_files))

        for temp_file in temp_image_files:
            os.remove(temp_file)
    
    return pdf_path

def get_gt_json_and_eval(image_files=None, save_dir="./output", is_save_gt=True, is_eval=False, gt_json_path=None, is_only_xycut=False,annotations_path=False):
    prepare_directories(save_dir)
    pipeline = create_pipeline(pipeline="layout_parsing", device="gpu:0")

    if not isinstance(image_files, list):
        image_files = [image_files]

    total_bleu_score, total_ard, total_tau, total_files = 0, 0, 0, len(image_files)
    
    for idx,image_file in enumerate(image_files):
        bleu_score, ard, tau = process_image_file(image_file, pipeline, save_dir, is_save_gt, is_eval, gt_json_path, is_only_xycut,start_id=idx,annotations_path=annotations_path)
        total_bleu_score += bleu_score
        total_ard += ard
        total_tau += tau

    if is_eval and total_files > 0:
        print(total_bleu_score / total_files, total_ard / total_files, total_tau / total_files)

# def process_images(image_dir, save_dir="./output", is_save_gt=True,is_eval=True,gt_json_path=None, is_only_xycut=False):
#     output_pdf_path = os.path.join(save_dir, 'input.pdf') 
#     if not os.path.exists(output_pdf_path):
#         extends = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
#         image_files = []
#         for extend in extends:
#             image_files.extend(glob.glob(os.path.join(image_dir, f'*.{extend}')))
#         pdf_path = png_to_pdf(png_files=image_files)
#         shutil.copy(pdf_path, output_pdf_path)
#         print(f"PDF copy path: {output_pdf_path}")

#     get_gt_json_and_eval(image_files=output_pdf_path, save_dir=save_dir, is_save_gt=is_save_gt, is_eval=is_eval, gt_json_path=gt_json_path, is_only_xycut=is_only_xycut)
def find_images_recursively(image_dir, extensions):
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                image_files.append(os.path.join(root, file))
    return image_files

def rename_and_save_images(image_files, output_dir):
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个图像文件
    for idx, image_path in enumerate(image_files):
        # 读取图像
        image = cv2.imread(image_path)
        
        # 确认图像是否成功加载
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # 获取原始文件名和扩展名
        original_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(original_filename)
        
        # 创建新的文件名
        new_filename = f"{idx}_{filename}{ext}"
        
        # 构建保存路径
        save_path = os.path.join(output_dir, new_filename)
        
        # 保存图像
        cv2.imwrite(save_path, image)
        print(f"Saved {save_path}")


def process_images(image_dir, annotations_path, save_dir="./output", is_save_gt=True,is_eval=True,gt_json_path=None, is_only_xycut=False,is_gen_pdf=True,is_rename_imgs=False):
    output_pdf_path = os.path.join(save_dir, 'input.pdf')

    if annotations_path:
        image_files = []
        with open(annotations_path, 'r') as json_file:
            annotations_data = json.load(json_file)
            for i in range(len(annotations_data)):
                image_files.append(os.path.join(image_dir, annotations_data[str(i)]['input_path']))
    else:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
        image_files = find_images_recursively(image_dir, extensions)

    if is_rename_imgs:
        rename_and_save_images(image_files, output_dir=f'{save_dir}/inputs')

    if is_gen_pdf:
        if not os.path.exists(output_pdf_path) and len(image_files) != 0:  # 确保找到图像文件
            pdf_path = png_to_pdf(png_files=image_files)  # 假定该函数实现了将图像转换为 PDF 的功能
            shutil.copy(pdf_path, output_pdf_path)
            print(f"PDF copy path: {output_pdf_path}")
        elif len(image_files) == 0:
            print("No image files found to convert to PDF.")
            return
        # 调用后续处理函数
        get_gt_json_and_eval(
            image_files=output_pdf_path,
            save_dir=save_dir,
            is_save_gt=is_save_gt,
            is_eval=is_eval,
            gt_json_path=gt_json_path,
            is_only_xycut=is_only_xycut,
            annotations_path=annotations_path
        )
    else:
        get_gt_json_and_eval(
            image_files=image_files,
            save_dir=save_dir,
            is_save_gt=is_save_gt,
            is_eval=is_eval,
            gt_json_path=gt_json_path,
            is_only_xycut=is_only_xycut,
            annotations_path=annotations_path
        )


def main(image_files=None,save_dir = './output/new', is_eval=True, is_only_xycut=False):
    save_path = f'{save_dir}/layout_parsing_result.md'
    clear_file(save_path)
    prepare_directories(save_dir)

    pipeline = create_pipeline(pipeline="layout_parsing", device="gpu:0")

    if not isinstance(image_files, list):
        image_files = [image_files]

    for image_file in image_files:
        page_files = pipeline.predict(image_file)
        for res in page_files:
            res.save_to_markdown(f"{save_path}",is_eval=is_eval, is_only_xycut=is_only_xycut)
            res.save_to_ordering(f"{Path(save_path).parent}/ordering", is_eval=is_eval, is_only_xycut=is_only_xycut)
            # res.save_results(f"{Path(save_path).parent}")
            # res.save_to_json(f"{Path(save_path).parent}/jsons")
            # res.save_to_img(f"{Path(save_path).parent}/images")

# 使用封装好的函数
if __name__ == "__main__":
    files_path = "/workspace/shuailiu35/eval_datasets/image70"  # image or pdf
    # annotations_path = "/workspace/shuailiu35/eval_datasets/instance_val_70.json"  # image or pdf
    annotations_path = None
    save_dir = "/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v6"
    # files_path = "/workspace/shuailiu35/eval_datasets/eval_datasets_new"
    # annotations_path = None
    # save_dir = "/workspace/shuailiu35/compare_with_mineru/30/mask_xycut"
    # process_images(image_dir=files_path,
    #                annotations_path=annotations_path,
    #                save_dir=save_dir,
    #                is_save_gt=True,
    #                is_eval=False, 
    #                gt_json_path=None,
    #                is_only_xycut=False,
    #                is_gen_pdf=True,
    #                is_rename_imgs=False)
    
    # files_path = "/workspace/shuailiu35/eval_datasets/eval_datasets_new"  # image or pdf
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/18_010201_00115.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/35_00075JL1MLDG8J51MLDG0K0IL9R_42.png"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/43_PMC5047312_00000.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/44_PMC4529683_00000.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/51_010201_01221.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/66_0301_0204_0055.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/64_2205.13262_17.png"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/23_0202_0008_0048.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/55_H3_AP202303151584292935_1.pdf_1678911358000_24.png"]

    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v1/inputs/42_00075JL1MLDG8J106HCG0K0IL9R_4.png"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/55_H3_AP202303151584292935_1.pdf_1678911358000_24.png"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/7_train_4972.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/12_0401_02168.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/18_010201_00115.jpg"]
    # image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/inputs/25_00075JL1MLDG8J1XMLC01K0IL9R_44.png"]
    image_files = ["/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v1/inputs/37_train_0392.jpg"]
    main(image_files,is_eval=False)

    # main(output_pdf_path,is_eval=False)
    # main("/workspace/shuailiu35/eval_datasets/0201_2021041001_2.jpg")
    # "/workspace/shuailiu35/eval_datasets/new_input/H3_AP202306151590970885_1pdf_1686850436000_29.png"
    # files = ["/workspace/shuailiu35/eval_datasets/new_input/H3_AP202306151590970885_1pdf_1686850436000_29.png",]
    # main(files,is_eval=True)


  
    
    
    
    
    
    
    
