def filename2id(x):
    # print(f"[filename2id] x: {x}")
    return int(x.split('_')[1].split('.')[0])

def dir_filename2id(x) -> tuple:
    return (int(x.split('/')[0]), int(x.split('_')[1].split('.')[0]))