# Modified by contributors from Intel Labs

ifndef TVM_HOME
  $(error Please source configuration script)
endif

include $(TVM_HOME)/verif/make/shell.mk

ifndef VTA_CONFIGS
  VTA_CONFIGS := vta_config
endif
VTA_DIRS := $(VTA_CONFIGS:%=VTA/%)

help:
	@echo "
	Create VTA build directories for verification.
	
	Usage: make [VAR=<val>] <task>
	
	Available Tasks:
	  make dirs
	  make flow
	
	VAR Configuration Variables:
	  VTA_CONFIGS: $(VTA_CONFIGS)
	
	Derived Variables:
	  VTA_DIRS: $(VTA_DIRS)
	
	Examples:
	Make build directories for all specified connfigurations:
	  \$$ make dirs
	Make a build directory for the minimal configuration only:
	  \$$ make VTA_CONFIGS=minimal dirs
	Make a build directory for the minimal and block2 configurations:
	  \$$ make VTA_CONFIGS=\"minimal block2\" dirs
	"

dirs: $(VTA_DIRS)

flow: dirs
	@for dir in $(VTA_DIRS); do
	  $(MAKE) -C $$dir flow
	done

VTA/%:
	@echo $@
	mkdir -p $@
	cat > $@/Makefile << EOF
	VTA_CONFIG := $*
	include \$$(TVM_HOME)/verif/make/vta.mk
	EOF
