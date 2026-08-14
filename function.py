
'''

def addNum (a,b):
    c=a+b
    print("Sum is",c)

addNum (10,20)



def multipy (a,b):
    c=a*b
    print("multipy is",c)

multipy (10,20)




def multipy (a,b):
    c=a*b
    return c

res=multipy(10,20)

print("res =" ,res)


def avg (a,b,c):
    d=(a+b+c)/3
    
    return d



res = avg (10,20,30)
print ("res=", res)



def isEven (num):
    rem= num % 2 
    if (rem==0):

        print("Even")

    else:
        print ("odd")

a=int(input("Enter a number:"))
isEven(a)



def isEven (num):
    rem= num % 2 
    return (rem==0)

a=int(input("Enter a number:"))
if (isEven (a)):
    print("Even")

else:
    print("Odd")


    


def avg (a,b,c):

  d=(a+b+c)/3

  return d 

res = avg (10,20,30)


print ("res=", res)




'''

def isEven (num):
    rem= num % 2

    if (rem==0):

        print("Even")

    else:
        print("Odd")

a=int(input("enter the number:"))
isEven (a)









