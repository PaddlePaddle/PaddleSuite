
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="OCR")

# output = pipeline.predict("./test_imgs/general_ocr_002.png")

output = pipeline.predict("./test_imgs/seal_text_det.png")
for res in output:
    print(res)
    res.save_to_img("./output")


