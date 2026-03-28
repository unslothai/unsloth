# Unsloth Studio Setup — Makefile
# Targets:
#   make                 → Linux native binary  (./setup)
#   make windows         → Windows binary via MinGW cross-compiler  (setup.exe)
#   make all             → both
#   make clean           → remove build artefacts

# ── Toolchain ────────────────────────────────────────────────────────────────

CC_LINUX   := gcc
CC_WINDOWS := x86_64-w64-mingw32-gcc   # adjust to i686-w64-mingw32-gcc for 32-bit

# ── Source ────────────────────────────────────────────────────────────────────

SRC := UnslothStudio.c

# ── Output names ─────────────────────────────────────────────────────────────

OUT_LINUX   := UnslothStudio
OUT_WINDOWS := UnslothStudio.exe

# ── Flags shared by both targets ─────────────────────────────────────────────

CFLAGS_COMMON := \
    -std=c11 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Wno-unused-parameter \
    -O2

# Linux — link nothing extra (POSIX is in libc)
CFLAGS_LINUX := $(CFLAGS_COMMON) -D_GNU_SOURCE
LDFLAGS_LINUX :=

# Windows — subsystem console; link ws2_32 for any future socket use
CFLAGS_WINDOWS := $(CFLAGS_COMMON) -D_WIN32_WINNT=0x0601
LDFLAGS_WINDOWS := -lkernel32 -lshell32 -Wl,-subsystem,console

# ── Default target ────────────────────────────────────────────────────────────

.PHONY: all linux windows clean

all: linux windows

linux: $(OUT_LINUX)

windows: $(OUT_WINDOWS)

# ── Linux build ───────────────────────────────────────────────────────────────

$(OUT_LINUX): $(SRC)
	$(CC_LINUX) $(CFLAGS_LINUX) -o $@ $< $(LDFLAGS_LINUX)
	@echo "Built: $@"

# ── Windows cross-compile build ───────────────────────────────────────────────

$(OUT_WINDOWS): $(SRC)
	$(CC_WINDOWS) $(CFLAGS_WINDOWS) -o $@ $< $(LDFLAGS_WINDOWS)
	@echo "Built: $@"

# ── Compile-only check (no link) ─────────────────────────────────────────────

check:
	$(CC_LINUX) $(CFLAGS_LINUX) -fsyntax-only $(SRC)
	@echo "Syntax OK"

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(OUT_LINUX) $(OUT_WINDOWS)
	@echo "Cleaned."