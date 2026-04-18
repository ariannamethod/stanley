# stanley 2.1 — weightless organism, pure C.
# no pytorch. no python. libc + libm + libpthread only.

CC       ?= cc
CFLAGS   ?= -std=c11 -Wall -Wextra -O2 -I.
LDFLAGS  ?= -lm -lpthread

# On macOS, cc -lpthread is implicit and sometimes complained about;
# keep it for Linux compatibility.
UNAME_S := $(shell uname -s)

SRC      := stanley.c graze.c main.c
HDR      := stanley.h graze.h
OBJ      := $(SRC:.c=.o)
BIN      := stanley

# Per-feature test suites. Each is a standalone binary linked against the
# organism + graze. New features land with their own file under tests/.
TEST_SRC  := tests/test_core.c tests/test_graze.c tests/test_maturity.c \
             tests/test_shimmer.c tests/test_refused.c tests/test_integration.c
TEST_BINS := $(TEST_SRC:.c=)

.PHONY: all clean test test-build run demo

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BIN)
	./$(BIN)

# REPL demo — drives Stanley through a short scripted dialogue.
demo: $(BIN)
	@printf 'hello stanley\n/stats\nare you there\n/stats\n/quit\n' | ./$(BIN) --no-origin | head -40

# Build + run every suite under tests/. Each suite reports PASS/FAIL.
test: test-build
	@set -e; for t in $(TEST_BINS); do \
	    printf "\n--- %s ---\n" $$t; \
	    ./$$t; \
	done

test-build: $(TEST_BINS)

tests/test_%: tests/test_%.c stanley.c graze.c $(HDR) tests/check.h
	$(CC) $(CFLAGS) -o $@ $< stanley.c graze.c $(LDFLAGS)

clean:
	rm -f $(OBJ) $(BIN) $(TEST_BINS)
