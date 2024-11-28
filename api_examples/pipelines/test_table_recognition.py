
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline("./test_imgs/table_recognition.jpg")
for res in output:
    print(res)
    res.save_to_img("./output/") ## 保存img格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
    res.save_to_html("./output/") ## 保存html结果


