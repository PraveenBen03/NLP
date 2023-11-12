import os
gen_directory = os.getcwd()



curd = "\save"


    
data_directory = gen_directory + curd

files_in_directory = os.listdir(data_directory)

print(files_in_directory)