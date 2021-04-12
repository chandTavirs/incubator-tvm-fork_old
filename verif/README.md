<!--- This file was modified by contributors from Intel Labs -->
<!--- All json files within this directory or subdirectories were also modified by contributors from Intel Labs -->

# Dynamic Verification (DV) Package

## Introduction

The dynamic verification package provides a flexible way of running arbitrary tests using a combination of parameters including targets such as CPU, BSIM, FSIM, TSIM, FPGA, and other user selected knobs.

It is based on the pytest tester and integrates perfectly with the TVM python based environment, including continuous improvement in GitLab, and development, debug with Visual Studio Code with its full support for parametric tests.

The package removes the limitation of having to specify the VTA target in a single file and enables running multiple tests in parallel shortening verification time.
Running the full set of tests in gitlab-ci currently takes less than 15 minutes.

The package also provides a verification methodology at module and system (end-to-end) levels, although some components are still WIP.
In short, each test can generate a dump or trace of architectural states in each of the supported abstraction models, by default BSIM, FSIM and TSIM, which can be cross referenced using design in-variants elements, such as counters, monitors, SRAM addresses, etc.

Performance verification support is available through data collected from performance counters visualized in a roofline chart using a provided application.

## Installation

To use it, there are just a few requirements to fulfill in your environment:

```sh
$ pip install --upgrade pytest
$ pip install pytest-xdist pytest-depends pytest-testmon

# To use the roofline app you also need the following:
$ pip install matplotlib pandas
```

Make sure to make the project from the top TVM directory to generate a VTA configuration header file the package relies on for debugging purposes.

```sh
$ cd $TVM_HOME
$ make -j `nproc`
$ make chisel
```

The first time only, create the verif/work directory.

```sh
$ mkdir -p verif/work
```

If acceptable, it is advisable for both disk performance and usage to use a local disk as workspace.
```sh
$ mkdir -p /localdisk/$USER/verif/work
$ cd verif
$ ln -s /localdisk/$USER/verif/work
```

Your are now ready to use the verification package.

## List Available Tests

To verify that the package works correctly in your work space list all of the available default tests.
Note that these are also the tests that are currently run in the continuous integration flow in GitLab.

Run pytest in collect only mode.

```sh
$ cd verif
$ pytest --co
collected 94 items
Package /data/work/tvm/verif/test
  Module check_deploy_classification.py
    TVM-VTA Verification Test Module.
    Wrapper for vta/tutorials/frontend/deploy_classification.py.
    Accepts all targets.
    Function test_deploy_classification[resnet18_v1-fsim]
    Function test_deploy_classification[resnet18_v2-fsim]
    Function test_deploy_classification[resnet18_v1-tsim]
    Function test_deploy_classification[resnet18_v2-tsim]
...
```

As you can see, pytest reports the collected tests first by module, that is, the file in which the tests are found, and by test function, with each line showing the parameter permutation with which the test was run in square brackets, with each parameter separated by a hyphen.

For instance, `test_deploy_classification[resnet18_v1-fsim]` indicates that test `test_deploy_classification` was run for the FSIM target using resnet18_v1 as the selected workload parameter. We refer to this encoding, as in `test_deploy_classification[resnet18_v1-fsim]`, as the full test name.

## Command Line Options

The help argument reports all options available from pytest, including custom options added by the verification package.

```sh
$ pytest --help

custom options:
  --mode={quiet,log,prof,short,long,full}
                        Trace mode (default: log).
  --targets={cpu,fsim,tsim,de10nano,pynq} [{cpu,bsim,fsim,tsim,de10nano,pynq} ...]
                        Accelerator targets (default: ['bsim', 'fsim', 'tsim']).
```

## Test Selection by Target

Targets are the available TVM targets that corresponds to the different devices, simulation abstraction models in use. By default targets BSIM, FSIM and TSIM are used. Please do not use FPGA targets yet as today this still requires manual checking of availability.

To select specific targets and see the collection results without running any test use the --co switch as we have seen before. For instance to restrict all test selection to use FSIM:

```sh
$ pytest --co --targets fsim
collected 47 items
Package /data/work/tvm/verif/test
  Module check_deploy_classification.py
    TVM-VTA Verification Test Module.
    Wrapper for vta/tutorials/frontend/deploy_classification.py.
    Accepts all targets.
    Function test_deploy_classification[resnet18_v1-fsim]
    Function test_deploy_classification[resnet18_v2-fsim]
...
```

To select multiple targets just append them to the command line.

```sh
$ pytest --co --targets fsim tsim cpu
collected 118 items
Package /data/work/tvm/verif/test
  Module check_deploy_classification.py
    TVM-VTA Verification Test Module.
    Wrapper for vta/tutorials/frontend/deploy_classification.py.
    Accepts all targets.
    Function test_deploy_classification[resnet18_v1-fsim]
    Function test_deploy_classification[resnet18_v2-fsim]
    Function test_deploy_classification[resnet18_v1-tsim]
    Function test_deploy_classification[resnet18_v2-tsim]
    Function test_deploy_classification[resnet18_v1-cpu]
    Function test_deploy_classification[resnet18_v2-cpu]
```

## Test Selection by File

Tests can be selected by the file they are contained in to easily restrict to a particular module we are working on. For instance to see all of the tests stored in the `test_benchmark_gemm.py` file we point to the `check_test_benchmark_gemm.py` wrapper (more on this later) in the test directory.

```sh
$ pytest --co --targets fsim tsim cpu -- test/check_test_benchmark_gemm.py
collected 12 items
Package /data/work/tvm/verif/test
  Module check_test_benchmark_gemm.py
    TVM-VTA Verification Test Module.
    Wrapper for vta/tests/python/integration/test_benchmark_gemm.py.
    Accepts all targets but cpu.
    Function test_vta_alu_insn[E2E-fsim]
    Function test_vta_alu_insn[GEMM-fsim]
    Function test_vta_alu_insn[ALU-fsim]
    Function test_vta_alu_insn[LD_INP-fsim]
    Function test_vta_alu_insn[LD_WGT-fsim]
    Function test_vta_alu_insn[ST_OUT-fsim]
    Function test_vta_alu_insn[E2E-tsim]
    Function test_vta_alu_insn[GEMM-tsim]
    Function test_vta_alu_insn[ALU-tsim]
    Function test_vta_alu_insn[LD_INP-tsim]
    Function test_vta_alu_insn[LD_WGT-tsim]
    Function test_vta_alu_insn[ST_OUT-tsim]
```

Note how not all tests support all targets, with this specified in each individual test file. For instance these tests are not currently being setup to be run with the CPU target. Also note how we use two hyphens to separate options from file arguments.

## Test Selection by Keyword Matching

The most fine grained way of selecting test out of all available ones is to use the pytest keyword -k switch, with which we can match a pattern on the full test name, including an individual parametrized test. For instance try the following:

```sh
$ pytest --co -k test_vta_alu_insn[GEMM
$ pytest --co -k test_vta_alu_insn[GEMM-tsim]
```

Note that you can combine this switch with all other selection methods.

## Test Mode

Recall the list of reported test modes from the pytest help:

```sh
--mode={quiet,log,prof,short,long,full}
                        Trace mode (default: log).
```

A test mode instructs the TVM runtime to generate a dump or trace of architectural states to be used for verification, debug, profiling, etc. In the following we briefly describe the available modes. Custom modes can be easily added as explained in the advanced section later.

- **quiet**: only report success or failure return status without generating any logs
- **log**: additionally generate log files of standard out and err if present, this is the default
- **prof**: generate profiling information
- **short**: generate a short trace
- **long**: generate a longer trace
- **full**: generate a trace that includes all instrumented architectural states
- **equiv**: generate a trace that can be directly compared between models

## Work Area

Test artifacts are stored in the work area directory $ROOT/verif/work after test execution. Each test will produce files with stem equal to its full name and extension depending on the type of data. Current extensions are:

- **out**: stdout written to any file descriptor during the test run<
- **err**: stderr written to any file descriptor during the test run
- **log**: log messages issues to the python logging framework
- **fail**: full report with call stack when the test fails
- **\<mode\>**: trace produced when test is run in the specified test mode

Please note that when running in quiet mode no files are produced and out, err, log files are always produced for all other modes, but only if anything is actually written to the corresponding file streams.

For instance, running a test for both FSIM and TSIM targets yields the following:

```sh
$ pytest --mode short -k test_vta_alu_insn[ADD-
$ ls work
test_vta_alu_insn[ADD-fsim].out
test_vta_alu_insn[ADD-fsim].short
test_vta_alu_insn[ADD-tsim].out
test_vta_alu_insn[ADD-tsim].short
```

## Running Tests in Parallel
To run tests in parallel use the ```-n \<num-threads>``` switch with pytest.
In particular pytest will automatically set VTA_TARGET internally with targets passed on the command line.
For instance to run the relu test for fsim, bsim, tsim in parallel:
```sh
pytest -k relu --targets fsim tsim bsim -n3
```

## Decoupled VTA Configuration and Target Selection

VTA configurations and targets are decoupled in different JSON files:

```sh
$VTA_HW/config/\<config>.json
$VTA_HW/config/\<target>_target.json
```
where \<config> are user specified labels with default set to ```vta_config```, and
\<target> is one of the default cmake supported target like tsim, bsim, fsim, etc.

A configuration can be selected through the environmental variable ```VTA_CONFIG``` which 
is accepted by the top Makefile when building both TVM and VTA.
For example, to build with configuration block32 and fsim, tsim runtimes:

```sh
cd $TVM_HOME
rm -rf build
VTA_CONFIG=block2 make -j12
VTA_CONFIG=block2 make chisel
```

Similarly environment variable VTA_TARGET, accepted by make, pytest and the VTA code at large
can be used to select different targets during different tasks.

For FPGA target for instance, ```VTA_TARGET``` is required when building the VTA runtime to select the appropriate cmake configuration and shared libraries to build.
For instance to build the runtime for the DE10-Nano board when logged in on the target:

```sh
cd $TVM_HOME
rm -rf build
export VTA_CONFIG=block2
export VTA_TARGET=de10nano
make -j12
```

In particular pytest will automatically set VTA_TARGET internally with targets passed on the command line.
For instance to run the relu test for fsim, bsim, tsim in parallel:
```sh
pytest -k relu --targets fsim tsim bsim de10nano -n4
```

## FPGA Targets

## Roofline Performance Analysis

## Verification Methodology

## Direct and Random Testing

## Trace Based Validation

Advanced Usage
==============

Add Tests to the Test Suite
---------------------------

Add Trace Content in Runtime Code
---------------------------------

Add Custom Test Modes
---------------------

Add Test Modes
--------------
