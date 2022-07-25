FUNCTIONS = MORTONBITSHIFT MORTONPDEP PITCHEDNAIVE PITCHEDPRECALC PITCHEDFAST
DIMENSIONS = 1 2 3 4 5 6 7 8 9 10
COMPILERS = CLANG GCC

COMBINATIONS = $(foreach c,$(COMPILERS),$(addprefix $(c)_,$(foreach f,$(FUNCTIONS),$(addprefix $(f)_,$(DIMENSIONS)))))

JSON_DIR = out/json
ASM_DIR = out/asm

CLANG=clang++-14
GCC=g++-11
MCA=llvm-mca-14

table: make_table.py all_results
	python3 $< $@

$(JSON_DIR):
	mkdir -p $@

$(ASM_DIR):
	mkdir -p $@

all_results: $(JSON_DIR) $(addprefix $(JSON_DIR)/,$(addsuffix .json,$(COMBINATIONS)))

$(ASM_DIR)/CLANG_%.s: layouts.cpp $(ASM_DIR)
	$(CLANG) -S -DBUILD_$(firstword $(subst _, ,$*)) -DNDIMS=$(lastword $(subst _, ,$*)) -march=alderlake -O3 $< -o $@

$(ASM_DIR)/GCC_%.s: layouts.cpp $(ASM_DIR)
	$(GCC) -S -DBUILD_$(firstword $(subst _, ,$*)) -DNDIMS=$(lastword $(subst _, ,$*)) -march=alderlake -O3 $< -o $@

$(JSON_DIR)/%.json: $(ASM_DIR)/%.s
	sed "s/ret/# ret/g" $< | $(MCA) - --iterations=10000 --json > $@

.SECONDARY:
