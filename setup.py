import setuptools
import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules=[
    setuptools.Extension("nms.seqnms_module.compute_overlap",    # location of the resulting .so
                        ["car_tracking/nms/seqnms_module/compute_overlap.pyx"],) ]
setuptools.setup(
    name="IBB Traffic Tracking",
    # version="0.0.1",
    # author="Example Author",
    # author_email="author@example.com",
    description="Car tracking package",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    package_dir={"": "car_tracking"},
    packages=setuptools.find_packages(where="car_tracking"),
    python_requires=">=3.6",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    # ext_modules = cythonize("./car_tracking/nms/seqnms_module/compute_overlap.pyx"),
    include_dirs     = [numpy.get_include()],
    setup_requires   = ["cython>=0.28", "numpy>=1.14.0"]
)