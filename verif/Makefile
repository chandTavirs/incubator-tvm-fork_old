# Modified by contributors from Intel Labs
##############################################################################
# User Configuration.
##############################################################################

KEY := insn
TARGETS := bsim fsim tsim
MODES := equiv
SEED := 0xCafeFace

##############################################################################
# Tool Configuration.
##############################################################################

MAKEFLAGS  += --no-builtin-rules --no-builtin-variables
#MAKEFLAGS	+= --silent --no-print-directory

VERIF_HOME := $(TVM_HOME)/verif
VERIF_WORK := $(shell readlink -f $(VERIF_HOME)/work)
THIS_MAKEFILE := $(shell readlink -f $(MAKEFILE_LIST))

ifneq ($(KEY),)
	CO :=-k $(KEY) --co
else
	CO :=--co
endif
STEMS := $(strip $(shell pytest $(CO) --targets fsim | sed -ne 's/.*Function \(.*\)-fsim.*/\1/gp'))
FILES := $(foreach stem, $(STEMS), $(foreach target, $(TARGETS), $(foreach mode, $(MODES), $(stem)-$(target)].$(mode))))
TESTS := $(FILES:%=$(VERIF_WORK)/%)
NUMT  := $(words $(TARGETS))
SEQA  := $(shell seq 1 $$(($(NUMT)-1)))
SEQB  := $(shell seq 2 $(NUMT))
PAIRS := $(foreach a, $(SEQA), $(foreach b, $(wordlist $(a), $(NUM), $(SEQB)), $(word $(a), $(TARGETS)) $(word $(b), $(TARGETS))))

DIFFS := $(foreach s, $(STEMS), \
	$(foreach a, $(SEQA), \
		$(foreach b, $(wordlist $(a), $(NUMT), $(SEQB)), \
			$(foreach m, $(MODES), \
				$(VERIF_WORK)/$(s)-$(word $(a), $(TARGETS))-$(word $(b), $(TARGETS))].$(m) \
			)\
		)\
	)\
)

red := \x1b[31m
grn := \x1b[32m
ylw := \x1b[33m
end := \x1b(B\x1b[m

.PHONY:
.SUFFIXES:
.PRECIOUS: %.log
.ONESHELL:
.SHELLFLAGS = -e -c

correctness-check: $(TESTS)
equivalence-check: $(DIFFS)

tea:
	@echo "DIFFS"
	echo $(DIFFS) | tr ' ' '\n'
	echo "TESTS"
	echo $(TESTS) | tr ' ' '\n'

define TEST_RULE =
$(VERIF_WORK)/$(1)-$(2)].$(3):
	@if pytest -k $(1)- --targets $(2) --mode $(3) --seed $(SEED) >& /dev/null; then
		echo -ne "CHECK $(grn)PASS$(end)";
	else
		echo -ne "CHECK $(red)FAIL$(end)";
	fi
	echo " $(1)-$(2)].$(3)"
endef

define EQUIV_RULE =
$(VERIF_WORK)/$(1)-$(2)-$(3)].$(4): $(VERIF_WORK)/$(1)-$(2)].$(4) $(VERIF_WORK)/$(1)-$(3)].$(4)
	@if cmp $$^ > $$@; then
		echo -e "EQUIV $(grn)PASS$(end) $(1)-$(2)-$(3)].$(4)"
	else
		echo -e "EQUIV $(red)FAIL$(end) $(1)-$(2)-$(3)].$(4) - gvimdiff work/$(1)-$(2)].$(4) work/$(1)-$(3)].$(4)"
		mv $$@ $$@.fail
	fi
endef

$(foreach s,$(STEMS),\
	$(foreach t,$(TARGETS),\
		$(foreach m,$(MODES),\
			$(eval $(call TEST_RULE,$(s),$(t),$(m)))\
		)\
	)\
)

$(foreach s, $(STEMS),\
	$(foreach a, $(SEQA),\
		$(foreach b, $(wordlist $(a), $(NUMT), $(SEQB)),\
			$(foreach m, $(MODES),\
				$(eval $(call EQUIV_RULE,$(s),$(word $(a),$(TARGETS)),$(word $(b),$(TARGETS)),$(m)))\
			)\
		)\
	)\
)
