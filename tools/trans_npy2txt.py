import numpy as np

def trans(npy_file_path, txt_file_path):
    npy_data = np.load(npy_file_path)
    data_list = npy_data.tolist()
    with open(txt_file_path, 'w') as f:
        for i in range(len(data_list)):
            f.write(" ".join([str(i) for i in data_list[i]]) + "\n")

def get_dur_txt(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data = f.read()

    data = data.split('\n')
    data = [x.split(' ') for x in data]
    totol_len = len(data) - 1
    arr = np.zeros((totol_len, 9))
    for i in range(totol_len):
        for j in range(9):
            try:
                arr[i][j] = float(data[i][j])
            except Exception as e:
                print(e)
                print(i,j)
    data = arr
    assert data.shape[1] == 9
    # T,C
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    # culsum along T for each C
    data = data.cumsum(axis=0)
    data = data.astype(int)

    # write to file
    list_data = data.tolist()
    with open(output_file_path,"w")as f:
        for line in list_data:
            f.write(" ".join([str(x) for x in line]) + "\n")
