import zipfile
zip_f_path = '/mnt/d/python_project/convert_app/convert_app/output_fbxs/results.zip'
output_paths =['/mnt/d/python_project/convert_app/convert_app/output_fbxs/final_all_4.fbx','/mnt/d/python_project/convert_app/convert_app/output_fbxs/final_all_3.fbx']
with zipfile.ZipFile(zip_f_path, "w") as zipf:
    for file in output_paths:
        zipf.write(file)
# import os

# os.replace(os.path.join(os.getcwd(),"results.zip"), "/mnt/d/python_project/convert_app/convert_app/output_fbxs/results.zip")
