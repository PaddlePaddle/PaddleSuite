
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="doc_preprocessor")

test_img_path = "./test_imgs/img_rot180_demo.jpg"
# test_img_path = "./test_imgs/doc_distort_test.jpg"

output = pipeline.predict(test_img_path, 
    use_doc_orientation_classify=True,
    use_doc_unwarping=True)

for res in output:
    print(res)
    res.save_to_img("./output")
