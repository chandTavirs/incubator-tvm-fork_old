##############################################################################
# Never called on its own - checks made in caller.
##############################################################################
# Modified by contributors from Intel Labs

ifeq ($(VTA_CONFIG),)
  VTA_CONFIG := vta_config
endif

VERIF_DIR := $(TVM_HOME)/verif
VTA_DIR := $(TVM_HOME)/3rdparty/vta-hw
FPV_DIR := $(VERIF_DIR)/fpv
MAKE_DIR := $(VERIF_DIR)/make
RTL_DIR := $(VERIF_DIR)/src/rtl
TMP_DIR := /localdisk/$(USER)/vta
#NPROC := $(shell nproc)

MAKEFLAGS += --no-builtin-rules --no-builtin-variables
#MAKEFLAGS += --silent --no-print-directory -j$(NPROC)
.ONESHELL:
.SHELLFLAGS = -e -c

red := \x1b[31m
grn := \x1b[32m
ylw := \x1b[33m
cyn := \x1b[34m
end := \x1b(B\x1b[m

pick = $(word $(1), $(subst ., ,$(2)))

source_fun := source $(MAKE_DIR)/shell.sh
fun := $(MAKE_DIR)/shell.sh
