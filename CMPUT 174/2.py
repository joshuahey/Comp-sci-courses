def hours_to_legal_limit(bal):
    c=0
    while (bal>0.05):
        bal=bal-0.015
        c=c+1
    return c

x=float(input())
print(hours_to_legal_limit(x))