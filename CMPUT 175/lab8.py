def length(a):
    if a==[]:
        return 0     
    else:
        return 1+length(a[1:])

alist=[]
print(length(alist))


def intDivision(dividend, divisor):
    assert dividend>=0, ("negative {}".format(n))
    assert divisor>0, ("negative n 0 {}".format(n))
    if dividend < divisor:
        return 0
    else:
        return 1+intDivision(dividend-divisor,divisor)
def main():
    n = int(input('Enter an integer dividend: '))
    m = int(input('Enter an integer divisor: '))
    print('Integer division', n, '//', m, '=', intDivision(n,m))
main() 

def sumDigits(n):
    assert n>=0, ("negative {}".format(n))
    if n==0:
        return 0
    else :
        return (n%10)+sumDigits(n//10)
def main():
    number = int(input('Enter a number:'))
    print(sumDigits(number))
main() 

def reverseDisplay(n):
    if n==0:
        print(0)
        return 0
    elif n<10:
        print(n)
        return n
    else:
        print((n%10),end="")
        return reverseDisplay(n//10)

def main():
    number = int(input('Enter a number:'))
    reverseDisplay(number)
main() 