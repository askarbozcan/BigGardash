import os
import sys
import click

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


@click.command()
@click.option("-p","--darknet_path", default=os.getcwd(), \
              show_default=True, help="Where to install & build Darknet", required=False)

def main(darknet_path, pretrained_name=None):
    # folder validation
    if not os.path.isdir(darknet_path):
        raise ValueError(f"Provided folder path does not exist. ({darknet_path})")

    print("\n============================")
    print(f"Working in {os.getcwd()}")
    print(f"Virtual environment used: {sys.prefix}")
    print(f"Darknet will be installed in {darknet_path}")
    print("============================\n")
    # user sanity check
    _y = input("Continue? [y/n]").lower()
    if _y == 'n':
        print('Got [n]. Ending run without setup.')
        exit()
    if _y != 'y':
        raise ValueError(f'Unexpected value {_y}. Please re-run easy_setup.py and press y to continue setup.')

    print(f"Cloning into {os.path.join(darknet_path,'darknet')}")

    with cd(darknet_path):
        os.system('git clone https://github.com/AlexeyAB/darknet > install.log')
        print("Modifying 'darknet' to satisfy library requirements")
    
    with cd(os.path.join(darknet_path, "darknet")):
        os.system("sed -i 's/AVX=0/AVX=1/' Makefile")
        os.system("sed -i 's/LIBSO=0/LIBSO=1/' Makefile")

        print("Running make in 'darknet' directory")
        os.system('make >> install.log')
        print("Build complete")

    print("Downloading yolov4-tiny to pretrained_models...")
    os.system('python download_yolo_pretrained.py -f pretrained_models -n yolov4-tiny >> install.log')

    print("Fixing paths provided in YOLO confs...")
    os.system('python fix_yolo_paths.py -f pretrained_models >> install.log')

    print("Continuing normal setup...")
    os.system('python setup.py build_ext --inplace >> install.log')
    print(" ======= Setup complete. ======")

    print(f"""
Make sure to set environment variable for DARKNET_PATH as such (for Unix based systems):
$ export DARKNET_PATH={os.path.join(darknet_path, 'darknet')}
    """)


if __name__ == "__main__":
    main()