import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys, os, pdb

compile_flags = ["-O3"]
if sys.platform == "win32":
    compile_flags = ["/EHsc"] # exception handling: gives warnings otherwise
    # TODO: add optimization equivalents of O3, march=native
# stupid mac is stupid
elif sys.platform != "darwin" and sys.platform != "linux2":
    compile_flags.append("-march=native")
elif sys.platform == "darwin":
    os.environ['ARCHFLAGS'] = "-arch i386 -arch x86_64"
    compile_flags.append("-stdlib=libstdc++") # for tr1/* includes

ext_modules = [Extension(
    name="ystruct",
    language="c++", # grr, mac g++ -dynamic != gcc -dynamic
    sources=["ystruct.pyx","dai/exceptions.cpp","dai/util.cpp","dai/graph.cpp","dai/dag.cpp"],
    include_dirs = [numpy.get_include()],
    extra_compile_args = compile_flags,
    libraries = [],
    library_dirs = [],
    )]

setup(
    name = 'ystruct',
    cmdclass = {'build_ext': build_ext, 'inplace': True},
    ext_modules = ext_modules,
    )
