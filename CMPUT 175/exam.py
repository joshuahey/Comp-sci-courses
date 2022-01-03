# Constant: time = 2.2E-05 (sec)
# Linear: time = 2.9E-05 + -1.3E-10*n (sec)
# Quadratic: time = 2.4E-05 + -6.2E-16*n^2 (sec)
# Cubic: time = 2.3E-05 + -3.6E-21*n^3 (sec)
# Polynomial: time = -8.9 * x^-0.19 (sec)
# Logarithmic: time = 9.1E-05 + -6.7E-06*log(n) (sec)
# Linearithmic: time = 2.8E-05 + -1E-11*n*log(n) (sec)
# Exponential: time = -11 * -3.7E-06^n (sec)
# import big_o
# best, others = big_o.big_o(non_repetitive, sample_strings,n_measures=20)
# print(best)


x=[8,6,7,8,9,9,4]
for i in range(0,len(x)):
    x.append("hey")
print(x)