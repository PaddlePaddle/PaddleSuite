import sys
sys.path.append("../")
from paddlex import create_pipeline
import numpy as np
from pathlib import Path
import uuid
import cv2
import os
from layout_parse.XYCut import recursive_yx_cut,recursive_xy_cut
from layout_parse.Writer import DiskReaderWriter
import json
import glob
import time

def process_data(d, save_dir):
    if isinstance(d, dict):
        if 'input_path' in d:
            del d['input_path']  # 去除 input_path 
        for k, v in d.items():
            if k == 'img' and isinstance(v, np.ndarray) and v.ndim==3:  # 检查是否为图像数组
                # 使用 UUID 生成唯一文件名
                img_name = f"image_{uuid.uuid4().hex}.png"
                img_path = Path(save_dir) / img_name
                # 保存图像
                cv2.imwrite(str(img_path), v)
                # 用路径替换数组
                d[k] = f"imgs/{img_name}"
            else:
                process_data(v, save_dir)  # 递归处理嵌套结构
    elif isinstance(d, list):
        for item in d:
            process_data(item, save_dir)     
        

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_to_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving to JSON: {e}")


def convert_to_markdown(parsing_results):
    markdown_content = []

    # 遍历解析结果，根据内容类型进行处理
    for block in sorted(parsing_results,key=lambda x:x['index']):
        # 处理段落标题
        if ('paragraph_title',"without_layout") in block.keys():
            content_value = block['paragraph_title']
            if '.' in content_value:
                level = content_value.count('.') + 1
                markdown_content.append(f"{'#' * level} {content_value}")
            else:
                markdown_content.append(f"# {content_value}")

        # 处理文档标题
        elif 'doc_title' in block.keys():
            markdown_content.append(f"# {block['doc_title']}")

        # 处理图表和表格标题
        elif 'table_title' in block.keys():
            markdown_content.append(f"**{block['table_title']}**")
        elif 'figure_title' in block.keys():
            markdown_content.append(f"**{block['figure_title']}**")

        # 处理文本
        elif 'text' in block.keys():
            markdown_content.append(block['text'].replace('\n', ' ').strip())

        # 处理数字
        elif 'number' in block.keys():
            markdown_content.append(str(block['number']))

        # 处理摘要
        elif 'abstract' in block.keys():
            markdown_content.append(f"*Abstract*: {block['abstract']}")

        # 处理正文内容
        elif 'content' in block.keys():
            markdown_content.append(block['content'])

        # 处理图片
        elif 'image' in block.keys():
            if block['image'].get('img'):
                markdown_content.append(f"![Image]({block['image']['img']})")
            if block['image'].get('image_text'):
                markdown_content.append(f"![Image]({block['image']['image_text']})")

        # 处理公式
        elif 'formula' in block.keys():
            markdown_content.append(f"$$ {block['formula']} $$")

        # 处理表格
        elif 'table' in block.keys():
            markdown_content.append(block['table'])

        # 处理参考文献
        elif 'reference' in block.keys():
            markdown_content.append(f"[Reference]: {block['reference']}")

        # 处理脚注
        elif 'footnote' in block.keys():
            markdown_content.append(f"[^1]: {block['footnote']}")

        # 处理算法
        elif 'algorithm' in block.keys():
            markdown_content.append(f"**Algorithm**: {block['algorithm']}")

        # 处理印章
        elif 'seal' in block.keys():
            markdown_content.append(f"**Seal**: {block['seal']}")

        # 忽略页眉和页脚
        elif 'header' in block or 'footer' in block.keys():
            continue

        else:
            # 未知类型，做一些默认处理或忽略
            continue

    return "\n\n".join(markdown_content)


def sort_by_xycut(block_bboxes,direction=0):
    random_boxes = np.array(block_bboxes)
    res = []
    if direction == 1:
        recursive_yx_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
    else:
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
    return res

from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(image_path, bbox_index_pairs, output_path='output.png'):
    # 打开原始图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 遍历每个边界框及其索引，并在图像上绘制
    for bbox, index in bbox_index_pairs:
        # bbox是一个列表或元组，格式假设为 [x_min, y_min, x_max, y_max]
        draw.rectangle(bbox, outline="blue", width=3)  # 使用蓝色绘制边界框

        # 在边界框的左上角绘制索引
        text_position = (bbox[2], bbox[1] - 15)  # 文字位于框上方
        draw.text(text_position, str(index), fill="red")  # 使用红色标注索引

    # 保存结果图像
    image.save(output_path)


def main(image_dir=None):
    extends = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    image_files = []
    for extend in extends:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*.{extend}')))
    pipeline = create_pipeline(pipeline="layout_parsing",device="gpu:1")
    total_times = 0 
    for i in range(len(image_files)): 
        image_file = image_files[i]
        output = pipeline.predict(image_file)
        start_time = time.time()
        res = next(output) 
        total_times+=round(time.time() - start_time, 2)
        # 获取解析结果
        json_data = res.print() 
        if not os.path.exists(f'./output/{i}/imgs'): 
            os.makedirs(f'./output/{i}/imgs')
        # 预处理解析结果，后续需要预处理所有重叠框
        process_data(json_data,save_dir=f'output/{i}/imgs') 
        if json_data is not None: 
            layout_parsing_result = json_data['layout_parsing_result']
            parsing_result = layout_parsing_result['parsing_result']
            block_bboxes = []
            for block in parsing_result:
                block['layout_bbox'] = list(block['layout_bbox'])
                block['layout_bbox'] = [float(x) for x in block['layout_bbox']] 
                block_bboxes.append(block["layout_bbox"])
            block_bboxes = np.array(block_bboxes)
            res = sort_by_xycut(block_bboxes)
            sorted_boxes = block_bboxes[np.array(res)].tolist()  
            for block in parsing_result:
                try:
                    block['index'] = sorted_boxes.index(block['layout_bbox'])
                except Exception as e:
                    print(block['layout_bbox'])
            # 保存结果到 JSON 文件中
            save_to_json(json_data, f'output/{i}/{os.path.basename(image_files[i]).split(".")[0]}.json')
            # 绘制边界框
            draw_bounding_boxes(image_path=image_files[i], bbox_index_pairs=[(block['layout_bbox'], block['index']) for block in parsing_result],output_path=f'./output/{i}/{os.path.basename(image_files[i]).split(".")[0]}_layout.png')
            # 将解析结果转换为 Markdown 格式
            md_writer = DiskReaderWriter(f'output/{i}/')
            md_content = convert_to_markdown(parsing_results=parsing_result)
            md_writer.write(
                content=md_content,
                path=f'{os.path.basename(image_files[i]).split(".")[0]}.md'
            ) 
    print(total_times/len(image_files)) 

main("/workspace/shuailiu35/paddlex/input")
  
    
    
    
    
    
    
    
