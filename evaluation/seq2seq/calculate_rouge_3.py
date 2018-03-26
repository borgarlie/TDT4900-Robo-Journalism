from pyrouge import Rouge155


if __name__ == '__main__':

    # path_to_reference = "../for_rouge/pretrained1/reference_test/"
    # path_to_modelsummary = "../for_rouge/pretrained1/modelsummary_test/"

    # LEAD 3
    path_to_reference = "../for_rouge/lead3/reference/"
    path_to_modelsummary = "../for_rouge/lead3/modelsummary/"

    # get to the point
    # path_to_reference = "../for_rouge/get-to-the-point/reference/"
    # path_to_modelsummary = "../for_rouge/get-to-the-point/pointer-gen-cov/"

    r = Rouge155()
    r.system_dir = path_to_reference
    r.model_dir = path_to_modelsummary
    r.system_filename_pattern = '(\d+)_reference.txt'
    r.model_filename_pattern = '#ID#_modelsummary.txt'
    print("Starting to convert and evaluate")
    output = r.convert_and_evaluate()  # tar Lang tid
    print(output)
    output_dict = r.output_to_dict(output)
    print("Done")
