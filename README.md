# Energy Time Fairness (ETF): A GPU Scheduler For Balancing Energy and Time Fairness

ETF is a GPU scheduler that enables configurable trade-off between time and energy fairness. 

### Introduction

Traditional time-based fair schedulers is agnostic of energy consumption, which is a crucial and costly resource in both 
cloud and edge environments. Inequitable energy allocation could lead to problems such as premature battery depletion, 
compromising applications' functionality and the continuity of the services they provide. To address this problem, 
we introduce a novel concept called Energy Time Fairness and implement the ETF scheduler to enforce it for deep learning
inference workloads. ETF is capable of switching between time-, energy-, and hybird-fair mode, making it flexible to support
both energy-efficient and latency-sensitive tasks. 


### Paper

```
@inproceedings{sec2023-etf,
  title={Energy Time Fairness: Balancing Fair Allocation of Energy and Time for GPU Workloads},
  author={Qianlin Liang and Walid A. Hanafy and Noman Bashir and David Irwin and Shenoy, Prashant},
  booktitle={Proceedings of the 8th ACM/IEEE Symposium on Edge Computing (SEC)},
  year={2023},
  month = {12},
  doi = {10.1145/3583740.3628435},
  isbn = {979-8-4007-0123-8/23/12},
}
```

### Get Started

The current version is tested on Nvidia Jetson TX2. The program is built on top of a customized version of [Apache TVM](https://github.com/apache/tvm), 
which is included as a submodule. To build this program, clone the source code using the `--recursive` flag to include the 
dependency:

```shell
git clone --recursive https://github.com/umassos/energy-time-fairness.git
```

> **_NOTE:_** This repo also contains two sample models - `Resnet18` and `Resnet50` compiled on TX2 for testing purposes. 
> These files are uploaded using Git Large File System (LFS). Consider using filter when cloning if you don't want these files.  

### Build

To build this, using the following commands:

```shell
# In the project root
mkdir build
cd build
cmake ..
make
```

The building process may take a while. 

### Test

This project includes an executable, `efair_unittest`, for unit testing. After building, test it using: 

```shell
sudo ./efair_unittest
```

`sudo` is required to change the GPU frequencies. The test requires the two sample models files and their corresponding profile 
to be available under `models/` folder. You should see the following outputs if it works successfully: 

```shell
Running main() from gtest_main.cc
[==========] Running 7 tests from 2 test cases.
[----------] Global test environment set-up.
[----------] 1 test from SchedulerTest
[ RUN      ] SchedulerTest.schedulerOperations
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20231120 17:47:25.062916 18155 scheduler.cpp:184] Created schedule entity ID <0> with priority 0
I20231120 17:47:25.229256 18155 scheduler.cpp:114] Loaded model ID <0> resnet18 with max power 3736 mWatt
I20231120 17:47:25.229339 18155 scheduler.cpp:116] Model 0 execution frequency 1300500000 power 3736
I20231120 17:47:25.229382 18155 scheduler.cpp:122] Entity 0 current average power 3736
I20231120 17:47:25.229558 18155 scheduler.cpp:420] Scheduler started
I20231120 17:47:25.256263 18158 scheduler.cpp:354] Finished task <0> in 26651 µs
I20231120 17:47:25.256374 18155 scheduler.cpp:427] Stopping scheduler...
I20231120 17:47:25.256382 18158 scheduler.cpp:401] Entity <0> runtime: 26809 µs Time used this quantum: 26809 µs Frequency: 1300500000
I20231120 17:47:25.256641 18155 chfreq.cpp:135] Frequency controller is shutdown.
I20231120 17:47:25.257462 18155 scheduler.cpp:431] Scheduler has stopped.
[       OK ] SchedulerTest.schedulerOperations (206 ms)
[----------] 1 test from SchedulerTest (206 ms total)

[----------] 6 tests from ExecutorTest
[ RUN      ] ExecutorTest.getNumKernels
[       OK ] ExecutorTest.getNumKernels (343 ms)
[ RUN      ] ExecutorTest.getInputShape
[       OK ] ExecutorTest.getInputShape (257 ms)
[ RUN      ] ExecutorTest.getDType
[       OK ] ExecutorTest.getDType (256 ms)
[ RUN      ] ExecutorTest.executeResnet18
[       OK ] ExecutorTest.executeResnet18 (289 ms)
[ RUN      ] ExecutorTest.executeResnet50
[       OK ] ExecutorTest.executeResnet50 (314 ms)
[ RUN      ] ExecutorTest.getKernelName
[       OK ] ExecutorTest.getKernelName (256 ms)
[----------] 6 tests from ExecutorTest (1715 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 2 test cases ran. (1921 ms total)
[  PASSED  ] 7 tests.
```

### ETF Server

The ETF server runs the scheduler and provides model serving APIs such as `LoadModel()` and `Infer()`. The server 
requires three arguments:

1. `quantum_size`: the total quantum size (in ms) that is allocated to all applications. 
2. `phi`: The time fair factor. See the paper for more details. 
3. `device`: `gpu` for running using GPU, otherwise will run using CPU.

Following is a running example:

```shell
sudo ./run_server 40000 0.7 gpu
```

On exiting, the server should print the resource usage:

```shell
I20231120 18:02:28.089967 18534 scheduler.cpp:456] Time usage:
I20231120 18:02:28.090013 18534 scheduler.cpp:458] Model# 0: 2109185 µs	 Frequency 1300500000
I20231120 18:02:28.090072 18534 scheduler.cpp:458] Model# 1: 3914055 µs	 Frequency 726750000
I20231120 18:02:28.090108 18534 scheduler.cpp:461] Energy usage:
I20231120 18:02:28.090137 18534 scheduler.cpp:463] Model# 0: 8986264 µJ	 Frequency 1300500000
I20231120 18:02:28.090168 18534 scheduler.cpp:463] Model# 1: 6719130 µJ	 Frequency 726750000
```

### ETF Client

The client communicate with server using its APIs. `run_client` load a model on the server and send inference requests.
Running the client requires the following arguments:

1. `model_path`: the path to the compiled `.so` model file. 
2. `model_profile_path`: the path to the model profile `.json` file. 
3. `frequency_idx`: the index of frequency the model is running at, for example, index 12 means running at 1300 MHz on TX2.  
4`priorty`: the priority of this model, ranging from -20 to 19,
5`num_threads`: the number of threads used to send inference requests. 
6`delay_start_time (optional):` sleep `x` seconds before sending requests
7`duration (optional):` exit after `x` seconds. 

Following is a running example:

```shell
./run_client ../models/resnet18/resnet18.so ../models/resnet18/resnet18_profile.json 12 0 2
```

Typical output of running the client should be:

```shell
Created entity \#0
Created entity \#1
Loaded model \#0
Loaded model \#1
Infer finished in 148479 µs task id 0
Infer finished in 178076 µs task id 2
Infer finished in 286721 µs task id 1
Infer finished in 305795 µs task id 3
Infer finished in 258959 µs task id 5
...
```

