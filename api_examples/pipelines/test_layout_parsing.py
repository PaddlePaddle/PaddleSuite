
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing")

output = pipeline.predict("./test_imgs/test_layout_parsing.jpg",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True)

# output = pipeline("./test_imgs/demo_paper.png")
# output = pipeline("./test_imgs/table_recognition.jpg")
# output = pipeline.predict("./test_imgs/seal_text_det.png")
# output = pipeline.predict("./test_imgs/img_rot180_demo.jpg")
for res in output:
    # print(res)
    res.save_results("./output")