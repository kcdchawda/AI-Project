
'''''
#Function 
def addnum (a,b):
    c= a+b
    print ("sum is" ,c)
addnum  (10,20)

def avg (a,b):
    c= (a+b)/2
    return c
res= avg (10,20)
print ("res=",res)

def isEven (a,b):
    rem = num % 2
    if (rem==0):
        print ("Even")
    else:
        print ("odd")
    a=int(input("Enter a number"))
    isEven (a)

def isEven (a,b):
    rem = num % 2
    return (rem==0)
a=int(input("Enter a number"))
if (isEven (a)):
        print ("Even")
else :
     print ("odd")

def multipy (a,b):
     c= a*b
     print ("multipy", c)
     multipy (10,20)

def multipy (a,b):
     c= a*b
     return c
res=multipy(10,20)
print ("res=", res)



#Age

a=int(input("Enter a number: "))
b=int(input("Enter other number: "))
c=int(input("Enter other number: "))

c=a+b+c
d=c/3
print ("sum is ", c)
print ("avg ", d)



#sum of  number in a  range 
sum=0


for i in range (10,21):


      sum=sum+i


print(sum)

Sum=0 

for i in range (10,21):

      sum= sum+i
print (sum)

#sum of  number in a  range 
sum=0

for i in range (10,21):

      sum=sum+i

print (sum)

#While Loop

i=1
while (i<=5):
    print(i)
    i+=1


i=5
while (i>=1):
    print(i)
    i-=1


#While Loop
i=1
while (i<=5):
    print(i)
    i+=1

i=5
while (i>=5):
    print(i)
    i-=1


#while loop

i=1
while (i<=5):
    print(i)
i+=1

i=5
while (i>=5):
    print(i)
i-=1

# sum of  number in a  range 

sum=0 
for i in range (10,21):
    sum=sum+1
    print (sum)


#While Loop
i=1
while(i<=5):
    print(i)

i=5
while(i>=1):
    print(i)
    i-=1

#Additon 

i=1
sum=0
while (i<=5):
    sum=sum+1
    i+=1
print (sum)

#Reverse
for i in range (10,1,-1):
    print(i)

'''
def addnum (a,b):
    c=a+b
    print ("sum is", c)

addnum(1,2)
