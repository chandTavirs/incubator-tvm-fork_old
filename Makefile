# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Modified by contributors from Intel Labs

ROOTDIR = $(CURDIR)
# Specify an alternate output directory relative to ROOTDIR. Default build
OUTPUTDIR = $(if $(OUTDIR), $(OUTDIR), build)

.PHONY: clean all test doc pylint cpplint scalalint lint \
	build cython cython2 cython3 web runtime vta chisel

ifndef DMLC_CORE_PATH
  DMLC_CORE_PATH = $(ROOTDIR)/3rdparty/dmlc-core
endif

ifndef DLPACK_PATH
  DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

ifndef VTA_HW_PATH
  VTA_HW_PATH = $(ROOTDIR)/3rdparty/vta-hw
endif

INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =
VTA_CONFIG_H := $(VTA_HW_PATH)/include/vta/hw_config.h
VTA_CONFIG_PY := $(VTA_HW_PATH)/config/vta_config.py
ifneq ($(VTA_CONFIG),)
  VTA_CONFIG_JSON := $(VTA_HW_PATH)/config/$(VTA_CONFIG).json
else
  VTA_CONFIG_JSON := $(VTA_HW_PATH)/config/vta_config.json
endif
ifneq ($(VTA_TARGET),)
  VTA_CONFIG_JSON += $(VTA_HW_PATH)/config/$(VTA_TARGET)_target.json
else
  VTA_CONFIG_JSON += $(VTA_HW_PATH)/config/vta_target.json
endif
FPGA_LIST := pynq de10nano ultra96
IS_FPGA := $(filter $(VTA_TARGET), $(FPGA_LIST))

all: build
ifeq ($(IS_FPGA),)
	@$(MAKE) -C build
else
	@$(MAKE) -C build runtime vta
endif

build: vta_hw_config
	@mkdir -p build; \
	cp cmake/config.cmake build; \
	cd build; \
	rm -f CMakeCache.txt; \
	if [ -n "$(IS_FPGA)" ]; then \
	  echo "Target $(IS_FPGA) is FPGA"; \
	  sed -i 's/USE_LLVM [^)]*/USE_LLVM OFF/' config.cmake; \
	  sed -i '/USE_VTA_FPGA OFF/a set(CMAKE_CXX_FLAGS "$${CMAKE_CXX_FLAGS} -Wno-psabi")' config.cmake; \
	  sed -i 's/USE_VTA_FPGA OFF/USE_VTA_FPGA ON/' config.cmake; \
	  cp $(VTA_CONFIG_JSON) .; \
	  if [ -e "$(IS_FPGA)_target.json" ]; then \
	    mv $(IS_FPGA)_target.json vta_target.json; \
	  fi; \
	else \
	  sed -i 's/USE_LLVM [^)]*/USE_LLVM llvm-config-9/' config.cmake; \
	  sed -i 's/USE_VTA_FSIM OFF/USE_VTA_FSIM ON/' config.cmake; \
	  sed -i 's/USE_VTA_TSIM OFF/USE_VTA_TSIM ON/' config.cmake; \
	  sed -i 's/USE_VTA_BSIM OFF/USE_VTA_BSIM ON/' config.cmake; \
	fi; \
	cmake ..

relocate:
	if [ -d $(OUTPUTDIR) ]; then \
	  find $(OUTPUTDIR) \( -name flags.make -or -name DependInfo.cmake \) \
	  -exec cp -a {} {}_TSP \; \
	  -exec sed -i "s,/[^ ]*/tvm/,`pwd`/,g" {} \; \
	  -exec touch -r {}_TSP {} \; \
	  -exec rm {}_TSP \; ; \
	fi 

runtime:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) runtime

vta: build
	@$(MAKE) -C build vta_fsim
	@$(MAKE) -C build vta_bsim
	@$(MAKE) -C build vta_tsim

cpptest:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) cpptest

crttest:
	@mkdir -p build && cd build && cmake .. && $(MAKE) crttest

chisel:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel cleanall; \
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel

de10nano:
	$(MAKE) FREQ_MHZ=100 -C $(VTA_HW_PATH)/hardware/intel

program-de10nano:
	$(MAKE) FREQ_MHZ=100 -C $(VTA_HW_PATH)/hardware/intel program

pynq:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/xilinx

program-pynq:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/xilinx program

chisel-vcd:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel cleanall; \
	$(MAKE) USE_TRACE=1 USE_TRACE_PATH=/localdisk/$(USER)/tsim \
	-C $(VTA_HW_PATH)/hardware/chisel

chisel-trace:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel cleanall; \
	$(MAKE) CHISEL_TRACE=1 -C $(VTA_HW_PATH)/hardware/chisel

start-rpc-server:
	sudo -E ./apps/vta_rpc/start_rpc_server.sh
	
start-rpc-daemon:
	sudo -E nohup ./apps/vta_rpc/start_rpc_server.sh > rpc-server.log 2>&1 &

stop-rpc-daemon:
	for prog in sudo python3; do \
	  pid="`ps -C $$prog -o pid=`"; \
	  if [ -n "$$pid" ]; then \
	    echo "kill $$pid $$prog"; \
	    sudo kill $$pid; \
	  fi; \
	done

fpga-runtime:
	if [ -z "$(CI_COMMIT_SHA)" ]; then \
	  echo "Error: usage make CI_COMMIT_SHA=<hash> fpga-runtime"; \
	  exit 1; \
	fi; \
	$(MAKE) stop-rpc-daemon && \
	git fetch && \
	git -C $(VTA_HW_PATH) fetch && \
	git checkout $(CI_COMMIT_SHA) && \
	git submodule update && \
	$(MAKE) -j2 && \
	$(MAKE) start-rpc-daemon

################################################################################
# Continuous Integration Targets.
# Before merging or during a merge request please run the CI test locally with
# make ci
# To debug failing CI stages run corresponding targets like
# make ci-<stage>
# By default the ci-build target will not change a build configuration if one
# already exists to avoid removing special configurations. To run the exact ci
# build configuration using LLVM 6.0 please remove your local build directory.
################################################################################
.PHONY: ci-lint ci-buildtvm ci-rebuildvta ci-test-insn ci-test-conv2d ci-test-special ci-validation ci

ci-lint-cpp:
	python3 3rdparty/dmlc-core/scripts/lint.py vta cpp vta/include vta/src

ci-lint-python:
	python3 -m pylint vta/python/vta --rcfile=tests/lint/pylintrc

ci-lint-chisel:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel lint

ci-lint: ci-lint-cpp ci-lint-python ci-lint-chisel

ci-buildtvm:
	rm -f build/config.cmake; \
	$(MAKE) -j8; \

ci-rebuildvta:
	rm -f $(VTA_CONFIG_H); \
	$(MAKE) vta -j8; \

ci-test-insn:
	cd verif; \
	mkdir -p work; \
	pytest -n4 -k insn

ci-test-conv2d:
	cd verif; \
	mkdir -p work; \
	pytest -n4 -k conv2d

ci-test-special:
	cd verif; \
	mkdir -p work; \
	pytest -n4 -k "resnet18 or resnet50 or D1 or D9"

ci-validation:
	cd verif; \
	mkdir -p work; \
	pytest -n4 -k classification

ci-equivalence:
	cd verif; \
	mkdir -p work; \
	pytest --missing-dependency-action=run test/match_deploy_classification.py

ci: | ci-lint ci-buildtvm ci-rebuildvta ci-test-insn ci-test-conv2d ci-test-special ci-validation ci-equivalence

.PHONY: vta_hw_config #$(VTA_CONFIG_H)
vta_hw_config: $(VTA_CONFIG_H)

$(VTA_CONFIG_H): $(VTA_CONFIG_PY) $(VTA_CONFIG_JSON)
	@echo "-- Remake $@"; \
	echo "// This file was autogenerated from $(VTA_CONFIG_JSON)" > $@; \
	echo "// when running the top make." >> $@; \
	echo >> $@; \
	echo "#ifndef VTA_HW_CONFIG_H" >> $@; \
	echo "#define VTA_HW_CONFIG_H" >> $@; \
	echo >> $@; \
	python3 $< --defs | \
	sed 's/-D\(\S\+\)=\(\S\+\)\s\?/#define \1 \2\n/g' >> $@; \
	echo "#endif // VTA_HW_CONFIG_H" >> $@

# EMCC; Web related scripts
EMCC_FLAGS= -std=c++11 -DDMLC_LOG_STACK_TRACE=0\
	-Oz -s RESERVED_FUNCTION_POINTERS=2 -s MAIN_MODULE=1 -s NO_EXIT_RUNTIME=1\
	-s TOTAL_MEMORY=1073741824\
	-s EXTRA_EXPORTED_RUNTIME_METHODS="['addFunction','cwrap','getValue','setValue']"\
	-s USE_GLFW=3 -s USE_WEBGL2=1 -lglfw\
	$(INCLUDE_FLAGS)

web: $(OUTPUTDIR)/libtvm_web_runtime.js $(OUTPUTDIR)/libtvm_web_runtime.bc

$(OUTPUTDIR)/libtvm_web_runtime.bc: web/web_runtime.cc
	@mkdir -p $(OUTPUTDIR)/web
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -MM -MT $(OUTPUTDIR)/libtvm_web_runtime.bc $< >$(OUTPUTDIR)/web/web_runtime.d
	emcc $(EMCC_FLAGS) -o $@ web/web_runtime.cc

$(OUTPUTDIR)/libtvm_web_runtime.js: $(OUTPUTDIR)/libtvm_web_runtime.bc
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -o $@ $(OUTPUTDIR)/libtvm_web_runtime.bc

# Lint scripts
# NOTE: lint scripts that are executed in the CI should be in tests/lint. This allows docker/lint.sh
# to behave similarly to the CI.
cpplint:
	tests/lint/cpplint.sh

pylint:
	tests/lint/pylint.sh

jnilint:
	python3 3rdparty/dmlc-core/scripts/lint.py tvm4j-jni cpp jvm/native/src

scalalint:
	$(MAKE) -C $(VTA_HW_PATH)/hardware/chisel lint

lint: cpplint pylint jnilint

doc:
	doxygen docs/Doxyfile

javadoc:
	# build artifact is in jvm/core/target/site/apidocs
	cd jvm && mvn javadoc:javadoc -Dnotimestamp=true

# Cython build
cython:
	cd python; python3 setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/tvm/*/*/*.so python/tvm/*/*/*.dylib python/tvm/*/*/*.cpp

# JVM build rules
ifeq ($(OS),Windows_NT)
  JVM_PKG_PROFILE := windows
  SHARED_LIBRARY_SUFFIX := dll
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S), Darwin)
    JVM_PKG_PROFILE := osx-x86_64
    SHARED_LIBRARY_SUFFIX := dylib
  else
    JVM_PKG_PROFILE := linux-x86_64
    SHARED_LIBRARY_SUFFIX := so
  endif
endif

JVM_TEST_ARGS := $(if $(JVM_TEST_ARGS),$(JVM_TEST_ARGS),-DskipTests -Dcheckstyle.skip=true)

jvmpkg:
	(cd $(ROOTDIR)/jvm; \
		mvn clean package -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/$(OUTPUTDIR)" $(JVM_TEST_ARGS))
jvminstall:
	(cd $(ROOTDIR)/jvm; \
		mvn install -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/$(OUTPUTDIR)" $(JVM_TEST_ARGS))
format:
	./tests/lint/git-clang-format.sh -i origin/main
	black .
	cd rust; which cargo && cargo fmt --all; cd ..


# clean rule
clean:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) clean
	cd .. ;\
	rm -rf $(OUTPUTDIR); \
	rm -f $(VTA_CONFIG_H);
