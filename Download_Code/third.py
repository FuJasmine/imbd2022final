import os,glob
folder_path = "D://test//data"
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    print(filename)
    base = os.path.splitext(filename)[0]
    os.rename(filename, base + '.csv')