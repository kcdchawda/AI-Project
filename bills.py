'''


input: bill

bill: upto 2000 => 5% discount
bill: 2000 - 5000 => 10%
bill: 5000 - 8000 => 15%
bill: more than 8000 => 20%

'''

bill =int(input("Enter the bill:"))
if (bill>=0 and bill<=2000):
    print("discout 5%")


elif(bill>=2000 and bill<=5000):
    print("discout 10%")

elif(bill>=5000 and bill<=8000):
    print("discout 15%")

else :
    print("discout 20%")