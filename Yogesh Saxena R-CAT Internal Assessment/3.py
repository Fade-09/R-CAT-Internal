def fib_seq(n):
    a, b = 0, 1
    while a <= n:
        yield a
        a, b = b, a + b

n = 6 
fib_gen = list(fib_seq(n))
print(fib_gen)
