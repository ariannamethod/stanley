# stanley 2.0 — weightless organism, pure C.
# no pytorch. no python. libc + libm + libpthread only.

CC       ?= cc
CFLAGS   ?= -std=c11 -Wall -Wextra -O2 -I.
LDFLAGS  ?= -lm -lpthread

# On macOS, cc -lpthread is implicit and sometimes complained about;
# keep it for Linux compatibility.
UNAME_S := $(shell uname -s)

SRC      := stanley.c main.c
HDR      := stanley.h
OBJ      := $(SRC:.c=.o)
BIN      := stanley

.PHONY: all clean test run

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BIN)
	./$(BIN)

test: $(BIN)
	@printf 'hello stanley\n/stats\nare you there\n/stats\n/quit\n' | ./$(BIN) --no-origin | head -40

clean:
	rm -f $(OBJ) $(BIN)
