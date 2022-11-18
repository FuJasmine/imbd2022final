import os,glob
folder_path = "D://test//data//"
log_file = folder_path + "putty.txt"
new_file_name = folder_path + "new_file.txt"

with open(log_file) as bigfile:
    outF = open(new_file_name, "w")
    for line in bigfile.readlines():
        if "cat" in line and "111052" in line:
            new_file_name = folder_path + line.split("cat ")[1].split("\n")[0].split(".py")[0] + ".py"
            print(line)
            outF = open(new_file_name, "w")
        else:
            outF.write(line)
        # if "cat" in line:
        #     print(line)