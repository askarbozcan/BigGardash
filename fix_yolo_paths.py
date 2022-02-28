import os
import glob
from os.path import join as pjoin
import click

@click.command()
@click.option("-f","--models_path", help="Folder where YOLO model folders are located. Assume name of .data == name of .names", required=True)
def run(models_path):
    assert os.path.exists(models_path), "This folder does not exist."

    for model_name in os.listdir(models_path):
        model_folder = pjoin(models_path, model_name)
        if not os.path.isdir(model_folder):
            print(f"Skipping {model_name} (not a folder)")
            continue
        
        
        data_file = glob.glob(model_folder+"/*.data")
        assert len(data_file) == 1, ".data file either does not exist or there are a multiple of them!"
        data_file = data_file[0]

        data_file_short = os.path.split(data_file)[-1] # get just the file name (no dirs)
        data_file_name = os.path.splitext(data_file_short)[0] # get the name of .data file (no ext)
    
        names_file = pjoin(model_folder, f"{data_file_name}.names")

        
        with open(data_file, "r") as f:
            lines = f.readlines()

        for i,l in enumerate(lines):
            if "names" in l:
                lines[i] = "names = " + os.path.abspath(names_file) + "\n"
        
        with open(data_file, "w") as f:
            f.writelines(lines)

        print(f"Successfully changed \"{data_file}\"")

if __name__ == "__main__":
    run()