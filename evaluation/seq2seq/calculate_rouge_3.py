import time
from pyrouge import Rouge155


def calculate_rouge_3(path_to_reference, path_to_modelsummary):
    r = Rouge155()
    r.system_dir = path_to_reference
    r.model_dir = path_to_modelsummary
    r.system_filename_pattern = '(\d+)_reference.txt'
    r.model_filename_pattern = '#ID#_modelsummary.txt'
    print("Starting to convert and evaluate")
    output = r.convert_and_evaluate()  # tar Lang tid
    return output


if __name__ == '__main__':

    # path_to_reference = "../for_rouge/pretrained1/reference_test2/"
    # path_to_modelsummary = "../for_rouge/pretrained1/modelsummary_test/"
    # path_to_modelsummary = "../for_rouge/pretrained1/cnn_aftergan_test2/"

    path_to_reference = "../for_rouge/pretrained1/reference_new/"
    path_to_modelsummary = "../for_rouge/pretrained1/cnn_pretrain_new/"

    # path_to_reference = "../for_rouge/pretrained1/reference_new/"
    # path_to_modelsummary = "../for_rouge/pretrained1/cnn_pretrain_epoch13/"


    # LEAD 3
    # path_to_reference = "../for_rouge/lead3/reference/"
    # path_to_modelsummary = "../for_rouge/lead3/modelsummary/"

    # get to the point
    # path_to_reference = "../for_rouge/get-to-the-point/reference/"
    # path_to_modelsummary = "../for_rouge/get-to-the-point/pointer-gen-cov/"
    before = time.time()
    output = calculate_rouge_3(path_to_reference, path_to_modelsummary)
    total_time = time.time() - before
    print(total_time)
    print(output)
    print("Done")
