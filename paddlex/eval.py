from .inference.pipelines_new.layout_parsing.utils import calculate_metrics_with_page, load_data_from_json, paddlex_generate_input_data, mineru_generate_input_data

if __name__ == '__main__':
    import json
    import glob
    # gt_jsons = glob.glob("/workspace/shuailiu35/PaddleX_new/PaddleX/paddlex/output/new/gt_jsons/*")
    # gt_jsons.sort()
    # gt_data = []
    # for gt_json in gt_jsons:
    #     gt_data.append(load_data_from_json(gt_json)[0])
    gt_data = load_data_from_json("/workspace/shuailiu35/compare_with_mineru/70/gt_70.json") # block bbox太大，bleu score可能不是一个很好的指标
    # gt_data = load_data_from_json("/workspace/shuailiu35/compare_with_mineru/30/gt_30.json") # block bbox太大，bleu score可能不是一个很好的指标
    # gt bbox划分细一点，最好是和模型输出block一致,这样可以更准确的计算bleu score
    # gt 不需要考虑图像和表格区数据，只需要考虑文本区数据

    # input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v1/jsons/input_pdf/*.json")
    # input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v2/jsons/input_pdf/*.json")
    input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/70/mask_xycut-v5/jsons/input_pdf/*.json")
    # input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/70/xycut/jsons/input_pdf/*.json")
    # input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/30/mask_xycut/jsons/input_pdf/*.json")
    # input_jsons = glob.glob("/workspace/shuailiu35/compare_with_mineru/70/xycut/jsons/input_pdf/*.json")
    input_jsons.sort()
    input_data = []
    for i,input_json in enumerate(input_jsons):
        data = load_data_from_json(input_json)
        input_data.append(paddlex_generate_input_data(data,[gt_data[i]])[0])

    # data = load_data_from_json("/workspace/shuailiu35/compare_with_mineru/70/mineru/input/input_middle.json")
    # input_data = mineru_generate_input_data(data,gt_data) # 对于页面进行了规格化处理，需要生成新的GT
    bleu_score, ard , tau = calculate_metrics_with_page(input_data,gt_data)
    print(f"BLEU score: {bleu_score}, ARD: {ard}, Tau :{tau}")