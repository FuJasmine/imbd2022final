import os,glob
folder_path = "D://test//data"

for filename in glob.glob(os.path.join(folder_path, '*.txt')):
  with open(filename, 'r') as f:
    # text = f.read()
    print (filename)
    # print (len(text))

    with open(filename, 'r+') as fd:
        lines = fd.readlines()
        fd.seek(0)
        fd.writelines(line for line in lines if line.strip())
        fd.truncate()