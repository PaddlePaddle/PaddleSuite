
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc")

# img_path = "./test_demo_imgs/vehicle_certificate-1.png"
# key_list = ['驾驶室准乘人数']

# img_path = "./test_demo_imgs/test_layout_parsing.jpg"
# key_list = ['3.2的标题']

img_path = "./test_demo_imgs/seal_text_det.png"
key_list = ['印章上公司']

# visual_predict_res = pipeline.visual_predict(img_path, 
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True)

# ####[TODO] 增加类别信息
# visual_info_list = []
# for res in visual_predict_res:
#     visual_info_list.append(res["visual_info"])

# pipeline.save_visual_info_list(visual_info_list, "./res_visual_info/visual_info3.json")

visual_info_list = pipeline.load_visual_info_list("./res_visual_info/visual_info3.json")

vector_info = pipeline.build_vector(visual_info_list)

print(vector_info)

final_results = pipeline.chat(visual_info_list, key_list, vector_info)

print(final_results)
