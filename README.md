# Dense and CSR Sparse matrix library in C++ leveraging Intel oneMKL (CBLAS)

## What this project does

In this project of ours we are developing a high-perfomance Dense and CSR Sparse matrix library in modern C++ (icpx), leveraging Intel oneMKL (CBLAS + MKL Sparse BLAS)
for numerical computing and HPC applications.

## How to build & compile the project

The following build and compile instructions are for Linux (we will assume that you are using Ubuntu/Debian like distros). If you use Windows you may encounter problems.

### Requirements
Make sure you have **all of the following installed** before reading the Setup Instructions section. 
* **Git:**

    https://git-scm.com/install/linux

    or

    ```
    sudo apt-get install git
    ```
* **CMake:**

    ```
    sudo apt-get install cmake
    ```
* **Intel oneAPI Base Toolkit:** 

    https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
* **VS Code** (or you IDE of preference but we will assume you use VS Code below)

    You usually can install it directly from the App Center or from [here](https://code.visualstudio.com/download)

### Setup Instructions
1. **First you will need to clone the repository to your desired destination.**

    Go to the directory you want to clone the project:
    ```
    cd <path_to_your_dir>
    ```

    Clone the repo:
    ```
    git clone https://github.com/VictorMeimariss/Cpp-Project.git
    ```

    Then get inside the project root folder:
    ```
    cd Cpp-Project
    ```

2. **Next, you need to source/setup the oneAPI variables:**
    ```
    source /opt/intel/oneapi/setvars.sh
    ```
    The path above is usually the default, it may vary if you changed anything in the installation process

3. **Open your IDE:**
   
   For VS Code do:
    ```
    code .
    ```

### Build & Linking Instructions
Now to run the CMake Configuration run the following command in the VScode terminal:
```
cmake -S . -B obj
```
this will create a directory called /obj where all the .o files will go.

Next, Build and Link everything with the following command:
```
cmake --build obj
```

### Run a test (optional)
Now to confirm that everything went well you can run one of the test functions .cpp files (all the executables are inside /bin)
```
./bin/test_CG_dsolve 
```
it doesn't need to be test_CG_dsolve, this was just used as an example.

## Documentation
To access the documentation page of the library simply follow the link:
[Documentation](https://johnzik.github.io/Cpp-Project/)