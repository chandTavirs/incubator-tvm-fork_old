# Modified by contributors from Intel Labs

ifndef TVM_HOME
  $(error Please source configuration script)
endif

include $(TVM_HOME)/verif/make/shell.mk

SPLICE := 1
PIPE := 0
RENAME := 0
TESTS := relu shift store shl shr 1-32-32 4-32-32 mobilenet.D4

ifeq ($(VTA_CONFIG),minimal)
  CHISEL_CFG := JSONMinimal
else
  CHISEL_CFG := JSONDe10
endif

CHISEL_DIR := $(VTA_DIR)/hardware/chisel
SRC_PATH := $(VTA_DIR)/build/chisel
SRC_FILES := $(notdir $(wildcard $(SRC_PATH)/*.v))
DST_FILES := Test.$(CHISEL_CFG)Config.v VTAHostDPI.v VTAMemDPI.v VTASimDPI.v
SRC_FILES := $(addprefix $(SRC_PATH)/, $(DST_FILES))
TMP_DIR := /localdisk/$(USER)/vta/$(VTA_CONFIG)
VCD_PIPE := $(TMP_DIR)/Test.vcd
VSPLICE := $(VERIF_DIR)/apps/vsplice.py
VCDS := $(TESTS:%=%.vcd)

help:
	@echo $$'
	Run VTA build and verification tasks.
	
	Usage: make [VAR=<val>] <task>
	
	Available Tasks:
	* make build
	* make source
	* make flow
	* make browse
	* make <test>.debug
	* make clean
	
	VAR Configuration Variables:
	VTA_CONFIG	: $(VTA_CONFIG)
	TESTS		: $(TESTS)
	SPLICE		: $(SPLICE)
	PIPE		: $(PIPE)
	
	Derived Variables:
	TMP_DIR		: $(TMP_DIR)
	
	'

build $(SRC_FILES): | TVM.log VTA.log
source: VTA.f
vcd: $(VCDS)
flow: source

VTA.log:
	$(MAKE) -C $(CHISEL_DIR) cleanall
	$(strip VTA_CONFIG=$(VTA_CONFIG) \
	$(MAKE) CONFIG=$(CHISEL_CFG)Config USE_TRACE=1 USE_TRACE_PATH=$(TMP_DIR) \
	-C $(CHISEL_DIR) lib | tee $@)

TVM.log:
	rm -rf $(TVM_HOME)/build/CMakeFiles/vta_*
	VTA_CONFIG=$(VTA_CONFIG) $(MAKE) -C $(TVM_HOME) -j8 | tee $@

VTA.f: $(DST_FILES)
	echo TB.v > $@
	find . -name '*.v' ! -name TB.v -printf '%f\n' >> $@
	if [ $(RENAME) == 1 ]; then
	  cp $(RTL_DIR)/TB.v.tpl TB.v
	else
	  cp $(RTL_DIR)/VL.v.tpl TB.v
	fi

%.v: $(SRC_PATH)/%.v
	@cp $< $@;
	if [ "$*.v" == "Test.$(CHISEL_CFG)Config.v" ]; then
	  if [ "$(SPLICE)" == "1" ]; then
	    $(VSPLICE) $*.v;
	  fi
	fi

ifeq ($(PIPE),0)

$(TMP_DIR)/%.vcd: $(SRC_FILES)
	mkdir -p $(TMP_DIR)
	cd $(TVM_HOME)/verif
	VTA_CONFIG=$(VTA_CONFIG) pytest -k `basename $*` --targets tsim
	[ -r $(VCD_PIPE) ] && mv $(VCD_PIPE) $@

else # PIPE

$(TMP_DIR)/%.vcd.gz: $(SRC_FILES)
	rm -f $(VCD_PIPE)
	mkfifo $(VCD_PIPE)
	sed -e "s/module TOP/module tb/" \
	    -e "s/module Test/module vta/" $(VCD_PIPE) | \
	    gzip -c > $@ &
	cd $(TVM_HOME)/verif
	pytest -k `basename $*` --targets tsim
	if [ -r $(VCD_PIPE) ]; then
	  wait
	  rm $(VCD_PIPE)
	fi

endif # PIPE

clean:
	rm -rf *.v *.log *Log work.lib++ *.vcd VTA.f
